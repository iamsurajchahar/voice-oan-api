import asyncio
from contextlib import nullcontext
from functools import lru_cache
import json
import time
from typing import AsyncGenerator, Optional, Literal
import re
from fastapi import Request

import regex
# from fastapi import BackgroundTasks
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart, SystemPromptPart

from pydantic_ai.usage import UsageLimits

from agents.voice import voice_agent, voice_agent_signed_in, STATIC_VOICE_SYSTEM_PROMPT
from agents.tools.farmer import normalize_phone_to_mobile
from agents.services.farmer_cache import (
    get_farmer_data_cached_only,
    is_fetch_inflight,
    refresh_farmer_data_bounded,
    enqueue_farmer_refresh,
    should_refresh_farmer_data,
    exceeds_max_serve_stale,
)
from app.models.union import (
    UNION_BANNED_MESSAGE,
    is_ai_call_banned_union,
    resolve_supported_unions,
    union_banned_message_for_lang,
)
from app.services.scheme_ingestion import (
    SUPPORTED_SCHEME_UNIONS,
    SchemeCacheError,
    SchemeDependencyError,
    get_cached_scheme_records_for_union,
)
from agents.tools.common import (
    get_timeout_nudge_message,
    get_tool_nudge_message,
    send_nudge_message_raya,
    set_tool_call_nudge_event,
)
from agents.tools.conversation_state import set_conversation_closing_flag
from agents.tools.terms import get_ambiguity_hints_for_query
from helpers.gujarati_numbers import mask_tag_identifier
from helpers.utils import get_logger, clean_output_by_language, get_today_date_str
from app.config import settings
from app.utils import (
    update_message_history,
    trim_history,
    format_message_pairs,
    clean_message_history_for_openai,
    SessionRequestOwner,
    is_session_request_owner,
    refresh_session_request_ownership,
    release_session_request_ownership,
)
from app.model_boundary_capture import boundary_capture_context
from app.services.stt_signals import (
    detect_stt_signal,
    generate_stt_signal_response,
    count_consecutive_stt_signals,
)
from app.services.moderation import ModerationVerdict, check_moderation
from app.services.fallback import AGENT_ACTIVITY, classify, execute_with_fallback, stream_with_fallback, with_first_token_deadline
from app.services.non_meaningful import NonMeaningfulVerdict, check_non_meaningful_streak
from app.services import outbound as _outbound
from app.services.outbound_consent import ConsentVerdict, classify_consent
from app.services.translation import (
    INDIAN_LANGUAGES,
    OPENAI_PRETRANSLATION_MODEL,
    translate_text,
    translate_text_stream_fast,
    translate_to_english_with_gpt5_mini,
    translate_to_english_with_oss_vllm,
    translate_to_english_with_structured_fallback,
)
from app.services.voice_trace import VoiceTrace, create_voice_trace, sanitize_text
# NOTE: Removing telemetry for now.
# from app.tasks.telemetry import send_telemetry
from agents.deps import FarmerAccount, FarmerContext
from agents.services.farmer_identity import (
    identity_state_for_envelope,
    identity_tool_groups,
    unavailable_capability_lines,
)
from app.llm_core import resolver as _llm_resolver
from app.llm_core.config_model import Step as _LlmStep
from agents.models.farmer import FarmerDataEnvelope, FarmerRecord
try:  # Langfuse is optional at import time
    from langfuse import get_client as _get_langfuse_client
except ImportError:  # pragma: no cover
    _get_langfuse_client = None

logger = get_logger(__name__)


class SentenceSegmenter:
    sep = 'ŽžŽžSentenceSeparatorŽžŽž'
    latin_terminals = '!?.:;'
    jap_zh_terminals = '。！？'
    terminals = latin_terminals + jap_zh_terminals

    def __init__(self):
        terminals = self.terminals
        self._re = [
            (regex.compile(r'(\P{N})([' + terminals + r'])(\p{Z}*)'), r'\1\2\3' + self.sep),
            (regex.compile(r'(' + terminals + r')(\P{N})'), r'\1' + self.sep + r'\2'),
        ]

    @lru_cache(maxsize=2**16)
    def __call__(self, line: str):
        for (_re, repl) in self._re:
            line = _re.sub(repl, line)
        return [t for t in line.split(self.sep) if t != '']


sentence_segmenter = SentenceSegmenter()
VOICE_TRANSLATION_BATCH_CHAR_LIMIT = 600
VOICE_TRANSLATION_SOFT_SPLIT_MIN_CHARS = 180


def extract_complete_sentences(text: str):
    if not text:
        return [], ""
    inline_structural_match = re.search(r"(?=\s#{1,6}\s)|(?=\n#{1,6}\s)|(?=\n\d+\.\s)|(?=\n[-*•]\s)", text)
    if inline_structural_match and inline_structural_match.start() > 0:
        split_at = inline_structural_match.start()
        head = text[:split_at]
        tail = text[split_at:].lstrip("\n")
        if head:
            return [head], tail
    structural_match = re.search(r"\n(?=(?:#{1,6}\s|[-*•]\s|\d+\.\s))", text)
    if structural_match:
        split_at = structural_match.start()
        head = text[:split_at]
        tail = text[split_at:].lstrip("\n")
        if head:
            return [head], tail
    sentences = sentence_segmenter(text)
    if len(sentences) <= 1:
        return [], text
    return sentences[:-1], sentences[-1]


def _split_voice_batch_text(text: str, max_chars: int = VOICE_TRANSLATION_BATCH_CHAR_LIMIT) -> tuple[str, str]:
    if len(text) <= max_chars:
        return text, ""

    window = text[:max_chars]
    split_at = -1
    for pattern in ("\n\n", "\n", ". ", "? ", "! ", ": ", "; ", "। ", "。 ", "### ", "## ", "# "):
        idx = window.rfind(pattern)
        if idx >= VOICE_TRANSLATION_SOFT_SPLIT_MIN_CHARS:
            split_at = idx + len(pattern.rstrip())
            break

    if split_at < 0:
        structural_markers = (
            r"\n(?=#{1,6}\s)",
            r"\n(?=\d+\.\s)",
            r"\n(?=[-*•]\s)",
            r"(?<=:)\s+",
            r"(?<=;)\s+",
        )
        for pattern in structural_markers:
            matches = list(re.finditer(pattern, window))
            if matches:
                idx = matches[-1].start()
                if idx >= VOICE_TRANSLATION_SOFT_SPLIT_MIN_CHARS:
                    split_at = idx
                    break

    if split_at < 0:
        # Last resort: split at the latest word boundary so an unpunctuated
        # run-on still flushes for voice delivery instead of stalling until
        # the stream ends.
        idx = window.rfind(" ")
        if idx >= VOICE_TRANSLATION_SOFT_SPLIT_MIN_CHARS:
            split_at = idx

    if split_at < 0:
        return text, ""

    return text[:split_at], text[split_at:]


def extract_translation_units(text: str):
    if not text:
        return [], ""

    ready_sentences, remaining = extract_complete_sentences(text)
    ready_units = [unit for unit in ready_sentences if unit and unit.strip()]

    while remaining and len(remaining) >= VOICE_TRANSLATION_BATCH_CHAR_LIMIT:
        head, tail = _split_voice_batch_text(remaining)
        if not head or head == remaining:
            break
        ready_units.append(head)
        remaining = tail

    return ready_units, remaining


def _batch_starts_new_line_or_list(text: str) -> bool:
    if not text or not text.strip():
        return False
    stripped = text.lstrip()
    if text != stripped:
        return True
    if stripped.startswith(("-", "•")) and (len(stripped) == 1 or stripped[1:2].isspace() or stripped[1:2] == "."):
        return True
    if stripped.startswith("*") and (len(stripped) == 1 or stripped[1:2].isspace() or stripped[1:2] == "."):
        return True
    return bool(re.match(r"^\d+\.\s", stripped))


def _prepare_text_for_voice_translation(text: str) -> str:
    """Make English text more translation-safe for voice rendering."""
    if not text:
        return text

    out = text
    # Flatten markdown list structure into spoken separators before translation.
    out = re.sub(r"\s*\n\s*[-*•]\s*", ", ", out)
    out = re.sub(r"\s*\n+\s*", " ", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r":,\s*", ": ", out)
    return out.strip()


# ── Greeting short-circuit helpers ─────────────────────────────────────
_GREETING_TOKENS = {
    # English
    "hello", "hi", "hey", "hlo",
    # Gujarati
    "હલો", "હેલો", "નમસ્તે", "નમસ્કાર",
    # Hindi
    "नमस्ते", "हेलो", "हलो",
    # Transliteration
    "namaste", "halo", "helo",
    # Multi-word greeting combos
    "ha hello", "હા હલો", "ji", "જી", "bolo", "બોલો",
    "ha bolo", "હા બોલો", "ji bolo", "જી બોલો",
}


def _is_bare_greeting(query: str) -> bool:
    """Return True if the query is just a greeting with no real content."""
    cleaned = re.sub(r"[*\s]+", " ", query).strip().lower()
    if not cleaned:
        return False
    # Strip punctuation for matching
    cleaned = re.sub(r"[.,!?।]+$", "", cleaned).strip()
    if cleaned in _GREETING_TOKENS:
        return True
    # Collapse repeated words: "hello hello" → "hello"
    words = cleaned.split()
    if len(words) <= 4:
        deduped = " ".join(dict.fromkeys(words))
        if deduped in _GREETING_TOKENS:
            return True
    return False


_GREETING_RESPONSES = {
    "gu": "નમસ્તે, હું સરલાબેન છું. તમારા પશુ વિશે કોઈ સમસ્યા હોય તો મને જણાવો.",
    "en": "Hello, I am Sarlaben. Please tell me what issue you are facing with your animal.",
}

# ── Fragment detection (garbled / too-short input) ────────────────────────
_FRAGMENT_RESPONSES = {
    "gu": "મને તમારો પ્રશ્ન સમજાયો નથી. કૃપા કરીને તમારો પ્રશ્ન ફરીથી પૂછો.",
    "en": "I could not understand your question. Please ask your question again.",
}

_HISTORY_MARKERS = {
    "greeting": "hello",
    "fragment": "[fragment]",
    "low_confidence": "[unclear-user-input]",
    "pretranslation_failed": "[pretranslation-failed]",
    "stt_no_audio": "[stt:no-audio]",
    "stt_unclear": "[stt:unclear-speech]",
    "moderation_reject": "[moderation-rejected]",
    "outbound_intro": "[outbound-call-started]",
}

# Markers that are dropped from the non-meaningful window entirely (neither
# counted nor streak-breaking). Excluded turns are skipped, so "5 consecutive
# non-meaningful turns" means consecutive among *eligible* turns — a
# pretranslation/moderation system turn in the middle does not reset the streak.
# fragment / unclear / no-audio / stt markers are intentionally NOT excluded:
# they represent genuinely unclear caller turns and should count toward a
# hangup (the prompt documents them as non-meaningful system markers).
_NON_MEANINGFUL_EXCLUDED_TURNS = frozenset(
    {
        #_HISTORY_MARKERS["fragment"],
        #_HISTORY_MARKERS["low_confidence"],
        _HISTORY_MARKERS["pretranslation_failed"],
        #_HISTORY_MARKERS["stt_no_audio"],
        #_HISTORY_MARKERS["stt_unclear"],
        _HISTORY_MARKERS["moderation_reject"],
        # Legacy marker from the removed in-app intro (Raya owns the opener now).
        # Not a caller turn at all, so it must neither count toward a hangup
        # streak nor break one, for as long as such histories survive trimming.
        _HISTORY_MARKERS["outbound_intro"],
    }
)


def _is_fragment_query(query: str) -> bool:
    """Return True if query is too short/garbled to be a real question."""
    cleaned = re.sub(r"[*\s.,!?।]+", " ", query).strip()
    if not cleaned:
        return True
    # Single character or very short (≤3 chars) — likely noise
    if len(cleaned) <= 3:
        return True
    return False


# ── Hold message detection ─────────────────────────────────────────────
# Carrier IVR "your call is on hold" messages get picked up by STT and
# sent as user input, creating runaway loops. Detect them and respond
# with "goodbye" so the STT provider cuts the call.
_HOLD_MSG_PATTERNS_GU = [
    "હોલ્ડ પર",            # "on hold" in Gujarati
    "લાઇન પર રહો",        # "stay on the line"
    "લાઈન પર રહો",        # variant spelling
]
_HOLD_MSG_PATTERNS_EN = [
    "put your call on hold",
    "call has been put on hold",
    "call on hold",
    "please stay on the line",
    "please remain on the line",
]
TELEPHONY_TERMINATE_CALL_TOKEN = {
    "gu": "Goodbye.",
    "en": "Goodbye.",
}

TRANSLATION_TROUBLE_MESSAGE = {
    "gu": "માફ કરશો, હાલમાં તમારા સવાલનો જવાબ આપવામાં તકલીફ થઈ રહી છે. કૃપા કરીને થોડા સમય પછી ફરી કોલ કરો.",
    "en": "I'm having some trouble answering your question right now, please call in some time.",
}


def _has_meaningful_history(history: list) -> bool:
    """Return True when the session already contains non-trivial conversation."""
    for msg in reversed(history or []):
        for part in getattr(msg, "parts", []) or []:
            content = getattr(part, "content", None)
            if not isinstance(content, str):
                continue
            text = content.strip()
            if not text:
                continue
            if detect_stt_signal(text) is not None:
                continue
            return True
    return False


def _is_hold_message(query: str) -> bool:
    """Return True if the query looks like a carrier hold/IVR message."""
    lower = query.lower()
    for pat in _HOLD_MSG_PATTERNS_GU:
        if pat in lower:
            return True
    for pat in _HOLD_MSG_PATTERNS_EN:
        if pat in lower:
            return True
    return False


def _greeting_response(target_lang: str) -> str:
    return _GREETING_RESPONSES.get(target_lang, _GREETING_RESPONSES["gu"])


# ── Identity fast-path ────────────────────────────────────────────────────
_IDENTITY_PHRASES_GU = {
    "તમારું નામ શું છે", "તારું નામ શું છે", "તમે કોણ છો", "આ સેવા શું છે",
    "આ કઈ સેવા છે", "તમે ક્યાંથી બોલો છો", "ક્યાંથી બોલો",
}
_IDENTITY_PHRASES_EN = {
    "what is your name", "who are you", "what is this service", "what service is this",
    "where are you calling from",
}

_IDENTITY_RESPONSE_EN = (
    f"I am Sarlaben from Amul AI. I was created on {settings.voice_profile_creation_date_words}, "
    "and I help dairy farmers with animal health, feed, and breeding guidance."
)

_WAIT_MESSAGES = {
    "gu": "રાહ જુઓ, હું તમારો જવાબ શોધી રહી છું.",
    "en": "Please wait a moment while I find the answer for you.",
}

_COMPARISON_PATTERNS = (
    r"\bdifference between\b",
    r"\bwhat is the difference\b",
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bversus\b",
    r"\bvs\b",
    r"\bvs\.\b",
    r"\bdifference\b",
)

_EXPLAINER_PATTERNS = (
    r"\bwhat is\b",
    r"\bwhat are\b",
    r"\btell me about\b",
    r"\bexplain\b",
    r"\bmeaning of\b",
)

_SYMPTOM_PATTERNS = (
    r"\bfever\b",
    r"\bnot eating\b",
    r"\bnot come in heat\b",
    r"\bnot coming in heat\b",
    r"\bbleeding\b",
    r"\bdiarrhea\b",
    r"\bloose motion\b",
    r"\bcough\b",
    r"\bbloat\b",
    r"\bmastitis\b",
    r"\bpregnant\b",
    r"\bcalving\b",
    r"\bsick\b",
)

_IDENTITY_DRIFT_PATTERN = re.compile(
    r"\b(?:OpenAI|ChatGPT|GPT|Claude|Anthropic|large language model|"
    r"I am an AI assistant made by|I am an AI made by|created by OpenAI|"
    r"made by Anthropic)\b",
    re.IGNORECASE,
)


def _fast_path_kind_for_query(text: str) -> Optional[Literal["identity"]]:
    """Return 'identity' if the query is an identity or social-greeting query, else None."""
    cleaned = re.sub(r"[.,!?।\s]+", " ", text).strip().lower()
    if not cleaned:
        return None
    if cleaned in _IDENTITY_PHRASES_GU or cleaned in _IDENTITY_PHRASES_EN:
        return "identity"
    return None


def render_in_flight_wait_message(lang: str) -> str:
    """Return the localized in-flight wait message for the given language code."""
    key = (lang or "en").strip().lower()
    return _WAIT_MESSAGES.get(key, _WAIT_MESSAGES["en"])


def _guard_identity_drift(text: str) -> str:
    """Replace any sentence that leaks a non-Sarlaben AI identity with the canonical line."""
    if not _IDENTITY_DRIFT_PATTERN.search(text):
        return text
    sentences = sentence_segmenter(text.strip())
    fixed = []
    replaced = False
    for s in sentences:
        if _IDENTITY_DRIFT_PATTERN.search(s):
            if not replaced:
                fixed.append(_IDENTITY_RESPONSE_EN)
                replaced = True
        else:
            fixed.append(s)
    return " ".join(fixed).strip()


def _voice_answer_mode_for_query(text: str) -> Optional[str]:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip().lower()
    if not cleaned:
        return None
    if any(re.search(pattern, cleaned) for pattern in _COMPARISON_PATTERNS):
        return "compact_comparison"
    if any(re.search(pattern, cleaned) for pattern in _EXPLAINER_PATTERNS):
        return "compact_explainer"
    if any(re.search(pattern, cleaned) for pattern in _SYMPTOM_PATTERNS):
        return "action_first_symptom"
    return None


def _prepare_voice_output(text: str, lang_code: str) -> str:
    """Normalize model output for voice delivery."""
    return clean_output_by_language(text, lang_code)


def _canonical_history_user_text(kind: str, fallback: str = "") -> str:
    return _HISTORY_MARKERS.get(kind, fallback or kind)


def _is_eligible_non_meaningful_turn(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if cleaned in _NON_MEANINGFUL_EXCLUDED_TURNS:
        return False
    return True


def _normalize_user_turn_for_non_meaningful(text: str) -> str:
    """Normalize stored user turn text before heuristic/classifier checks.

    History can contain wrapped forms like:
      **User:** "Correct"
    Strip wrappers/quotes so comparisons operate on caller content only.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    match = re.match(r'^\*\*User:\*\*\s*"?(.*?)"?$', cleaned, flags=re.IGNORECASE)
    if match:
        cleaned = (match.group(1) or "").strip()
    # Remove balanced outer quotes if still present.
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _collect_recent_user_turns_for_non_meaningful(history: list, current_query: str, limit: int = 5) -> list[str]:
    turns: list[str] = []
    for msg in reversed(history or []):
        for part in getattr(msg, "parts", []) or []:
            if getattr(part, "part_kind", "") != "user-prompt":
                continue
            content = getattr(part, "content", None)
            if not isinstance(content, str):
                continue
            normalized = _normalize_user_turn_for_non_meaningful(content)
            if not _is_eligible_non_meaningful_turn(normalized):
                continue
            turns.append(normalized)
            if len(turns) >= limit - 1:
                break
        if len(turns) >= limit - 1:
            break
    turns.reverse()
    normalized_current = _normalize_user_turn_for_non_meaningful(current_query)
    if _is_eligible_non_meaningful_turn(normalized_current):
        turns.append(normalized_current)
    return turns[-limit:]


def _should_gate_non_meaningful_llm(turns: list[str]) -> bool:
    """Gate on LLM only when we have a full 5-turn window."""
    return len(turns) >= 5


def _canned_union_ban_translation(text_en: str, target_lang: str) -> str | None:
    """Pinned Gujarati/Hindi union-ban copy when the English batch is that line."""
    if (text_en or "").strip() != UNION_BANNED_MESSAGE:
        return None
    return union_banned_message_for_lang(target_lang)


async def _render_text_for_caller(text_en: str, target_lang: str) -> str:
    """Render English loop text for the caller's language outside the agent loop."""
    normalized_target = (target_lang or "en").strip().lower()
    if normalized_target in {"en", "english"}:
        return _prepare_voice_output(text_en, "en")

    canned_ban = _canned_union_ban_translation(text_en, normalized_target)
    if canned_ban is not None:
        return _prepare_voice_output(canned_ban, normalized_target)

    try:
        translated = await translate_text(
            text=text_en,
            source_lang="english",
            target_lang=normalized_target,
        )
        return _prepare_voice_output(translated, normalized_target)
    except Exception as e:
        logger.error(
            "Caller render translation failed; target_lang=%s text=%r error=%s",
            normalized_target,
            text_en[:120],
            e,
        )
        return TRANSLATION_TROUBLE_MESSAGE.get(
            normalized_target,
            TRANSLATION_TROUBLE_MESSAGE["en"],
        )


async def _canned_for_caller(text_en: str, target_lang: str, canned: dict[str, str]) -> str:
    """Return a pre-written caller string for the target language when one exists,
    skipping the TranslateGemma round-trip on fixed fast-path replies. Falls back to
    live translation for languages that have no canned variant."""
    key = (target_lang or "en").strip().lower()
    if key in canned:
        return _prepare_voice_output(canned[key], key)
    return await _render_text_for_caller(text_en, target_lang)


def _history_pair(user_text: str, assistant_text: str) -> tuple[ModelRequest, ModelResponse]:
    return (
        ModelRequest(parts=[UserPromptPart(content=user_text)]),
        ModelResponse(parts=[TextPart(content=assistant_text)]),
    )


def _is_signed_in_session(user_info: Optional[dict], user_id: str) -> bool:
    if user_id and user_id != "anonymous":
        return True
    return bool(user_info)


async def get_or_fetch_farmer_data(mobile: str):
    """Voice read policy (stale-while-revalidate).

    Serve the cached envelope immediately when present (fresh or stale — the
    caller enqueues a background refresh for stale records). On a cold/never-
    cached (or hard-expired/deleted) miss, do a bounded blocking fetch so the
    first turn has data, capped by FARMER_COLD_FETCH_TIMEOUT so a slow upstream
    never hangs the call.

    Still patchable by tests that stub this symbol.
    """
    cached = await get_farmer_data_cached_only(mobile)
    if cached is not None:
        if exceeds_max_serve_stale(cached):
            # Too stale to serve (e.g. background refresh has been failing):
            # block on a bounded API call, falling back to the stale record
            # only if the API also fails.
            fresh = await refresh_farmer_data_bounded(mobile)
            return fresh if fresh is not None else cached
        return cached
    if await is_fetch_inflight(mobile):
        # A recent cold fetch was cancelled and a worker is still on it. Don't
        # pay the same budget again — this turn is unresolved.
        return None
    return await refresh_farmer_data_bounded(mobile)


def _build_runtime_context_request(deps: FarmerContext) -> ModelRequest:
    """Stable per-call context (constant across a call's turns: date, farmer
    profile, signed-in state, tool groups). Placed BEFORE history so the token
    sequence [system][stable-context][history] stays a single growing prefix that
    vLLM prefix caching can reuse across turns. Per-query content that changes
    every turn lives in _build_query_hints_request() and is appended AFTER history
    so it never breaks this prefix."""
    # Derived from the same predicate that drives the tool gates, so this line
    # can never advertise a group the model was not given. It previously
    # hardcoded "booking", announcing it on exactly the turns where booking was
    # impossible (issue #282).
    tool_groups = identity_tool_groups(deps)
    runtime_context = deps.get_runtime_context_message()
    context_lines = [
        "Runtime context for this turn:",
        f"- Today date: {get_today_date_str()}",
        runtime_context.replace("Runtime context for this turn:\n", "", 1),
        f"- Tool groups in this run: {', '.join(tool_groups)}",
    ]
    return ModelRequest(parts=[UserPromptPart(content="\n".join(context_lines))])


def _build_query_hints_request(deps: FarmerContext) -> Optional[ModelRequest]:
    """Per-query hints derived from the caller's current utterance: disambiguation
    rules (for ambiguous terms) and the voice answer mode. Kept OUT of the stable
    pre-history context — these change every turn, so placing them before history
    would break the cacheable prefix. Appended right before the user message
    instead, where the instructions also sit closest to the query they describe.
    Returns None when the query triggers neither hint."""
    hint_lines: list[str] = []
    # Inject ambiguity hints for the agent so it can decide to clarify vs. answer
    ambiguity_hints = get_ambiguity_hints_for_query(
        deps.query or "",
        threshold=settings.ambiguity_match_threshold,
    )
    if ambiguity_hints:
        hint_lines.append(f"- Disambiguation rules for terms in this query:\n{ambiguity_hints}")
    answer_mode = _voice_answer_mode_for_query(deps.query or "")
    if answer_mode == "compact_comparison":
        hint_lines.append(
            "- Voice answer mode: compact comparison. Give one short contrast sentence, then at most one short practical takeaway. Do not enumerate. Do not use labels, colons, or list structure. Do not append an extra follow-up question unless required."
        )
    elif answer_mode == "compact_explainer":
        hint_lines.append(
            "- Voice answer mode: compact explainer. Give one short plain-language definition or explanation, then at most one short practical takeaway. Do not teach the full topic. Do not enumerate. Do not use labels, colons, or list structure. Do not append an extra follow-up question unless required."
        )
    elif answer_mode == "action_first_symptom":
        hint_lines.append(
            "- Voice answer mode: action-first symptom response. Start with the most useful immediate action in one short sentence. Add at most one short safety or escalation sentence. Do not give long background, multiple causes, or a symptom checklist unless asked."
        )
    if not hint_lines:
        return None
    return ModelRequest(parts=[UserPromptPart(content="\n".join(["Hints for the current user query:", *hint_lines]))])


def _extract_farmer_tags(records: list[FarmerRecord]) -> list[str]:
    tags: list[str] = []
    for record in records:
        raw = record.tagNumbers or record.tagNo or ""
        if not raw:
            continue
        for tag in str(raw).split(","):
            cleaned = tag.strip()
            masked = mask_tag_identifier(cleaned)
            if masked and masked not in tags:
                tags.append(masked)
    return tags


def _render_breeding_value(value) -> str:
    """Compact, model-readable form of lastBreedingActivity (amulpashudhan returns
    a nested object with the AI date + bull id, or a flat string)."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _append_animal_records(lines: list[str], envelope: FarmerDataEnvelope) -> None:
    """Per-animal records (incl. last AI date + bull id) for breeding/AI-history
    questions. Tags are masked (last 4 digits) so TTS never reads a full tag aloud.
    Populated by the background cache refresh — absent on a brand-new caller's
    first turn, present from the next turn on."""
    blocks: list[str] = []
    for record in envelope.farmers:
        for animal in getattr(record, "animals", None) or []:
            data = animal.model_dump()
            masked = mask_tag_identifier(data.get("tagNumber") or "")
            if not masked:
                continue
            parts = [f"tag ending {masked}"]
            if data.get("animalType"):
                parts.append(f"type={data['animalType']}")
            breeding = data.get("lastBreedingActivity")
            if breeding:
                parts.append(f"last AI/breeding={_render_breeding_value(breeding)}")
            else:
                parts.append("no AI records available")
            blocks.append("- " + ", ".join(parts))
    if blocks:
        lines.append("")
        lines.append("### Per-animal AI / breeding history")
        lines.extend(blocks)


def _normalize_farmer_name(value) -> str:
    """Names carry stray dots and double spaces (`PATEL..`, `A  B`)."""
    # \w excludes Unicode Mn/Mc, so a plain [^\w\s] strips Gujarati matras and
    # collapses distinct names and villages (કડી and કડા both became કડ).
    cleaned = regex.sub(r"[^\p{L}\p{M}\p{N}\s]", "", str(value or ""))
    return regex.sub(r"\s+", " ", cleaned).strip().casefold()


def _build_compact_farmer_summary(envelope: Optional[FarmerDataEnvelope]) -> str:
    """Farmer block for the runtime context, or an explicit statement of what is
    unavailable and why.

    This used to return "" for both "no record exists" and "we have not resolved
    this caller yet", so the model received no farmer block at all and could not
    distinguish the two — or tell the caller anything useful about either. It
    then invented identifiers and called the tools anyway. The tools are now
    withheld on those turns (agents.services.farmer_identity); these lines are
    what lets the model say something true rather than "that feature does not
    exist" or claiming a booking it never made. See issue #282.
    """
    state = identity_state_for_envelope(envelope)
    if state != "found":
        return "\n".join(unavailable_capability_lines(state))

    first = envelope.farmers[0]
    tags = _extract_farmer_tags(envelope.farmers)
    societies = sorted({r.societyName for r in envelope.farmers if r.societyName})

    lines = [
        f"- Farmer records matched: {len(envelope.farmers)}",
        f"- Farmer data source: {envelope.source or 'unknown'}",
        f"- Farmer cache state: {'stale' if envelope.stale else 'fresh'}",
    ]
    if envelope.refreshAfter:
        lines.append(f"- Farmer refresh after: {envelope.refreshAfter}")
    if first.farmerName:
        lines.append(f"- Farmer name: {first.farmerName}")
    if societies:
        lines.append(f"- Societies: {', '.join(societies[:3])}")
    if first.farmerCode:
        lines.append(f"- Farmer code available: yes")
    union_code = first.model_dump().get("unionCode") or first.model_dump().get("union_code")
    society_code = first.model_dump().get("societyCode") or first.model_dump().get("society_code")
    if union_code:
        lines.append(f"- Union code: {union_code}")
    if society_code:
        lines.append(f"- Society code: {society_code}")
    if first.farmerCode:
        lines.append(f"- Farmer code: {first.farmerCode}")
    # Herd counts: always surface what we have. The agent answers from this
    # context (the brittle get_herd_summary / list_animal_tags / get_farmer_profile
    # tools were dropped — they read the same cache and returned "not available"
    # when the upstream record omitted totalAnimals even though tags were present).
    first_data = first.model_dump()
    total_animals = first.totalAnimals
    if total_animals is None and tags:
        total_animals = len(tags)  # fallback when upstream omits the count
    if total_animals is not None:
        lines.append(f"- Total animals: {total_animals}")
    cow = first_data.get("cow") or first_data.get("Cow")
    if cow is not None:
        lines.append(f"- Cows: {cow}")
    buffalo = first_data.get("buffalo") or first_data.get("Buffalo")
    if buffalo is not None:
        lines.append(f"- Buffaloes: {buffalo}")
    milking = first_data.get("totalMilkingAnimals") or first_data.get("Milking Animal")
    if milking is not None:
        lines.append(f"- Milking animals: {milking}")
    if tags:
        # All tags inline — no truncation, since the list-tags tool was dropped.
        lines.append(f"- Known animal tags: {', '.join(tags)}")
    # Which farmer to book for is only a real question when the answer changes
    # the visit. Technicians come from (unionCode, societyCode) alone, so when
    # every record sits in one society they all yield the same technician, the
    # same village and the same visit — 90% of multi-account mobiles. Asking
    # anyway is what killed the call: the mobile is shared by a household and
    # the names are not separable by ear (of 364 real pairs the agent offered,
    # 250 share a name token and 19 are byte identical), so the caller answers,
    # the answer fits both, and the agent asks again. 71 of the 265 calls in
    # "AI booking not done" died in that loop — the largest single remaining
    # source of failed bookings. See issue #282.
    if len(envelope.farmers) > 1:
        # Option numbers here must match the "Farmer option N" lines emitted below.
        _numbered = list(enumerate((r.model_dump() for r in envelope.farmers), start=1))

        def _codes(d: dict) -> tuple:
            return (
                str(d.get("unionCode") or d.get("union_code") or "").strip(),
                str(d.get("societyCode") or d.get("society_code") or "").strip(),
            )

        def _village(d: dict) -> tuple:
            """The village a record belongs to, for grouping.

            (unionCode, societyCode) is exactly the key the technician lookup
            uses — GetAITechniciansBySocietyQueryParams takes nothing else — so
            two records share a village iff they share this pair. A union-only
            key would merge two societies of one union.

            Only ever called on `bookable` records, which carry both codes by
            construction, so there is no missing-code case to handle here; that
            is decided once, above, and those records are excluded.
            """
            return _codes(d)

        def _village_label(d: dict) -> Optional[str]:
            """A village name the caller could say out loud, or None.

            Deliberately never a society code or a farmer name: a caller cannot
            read out "00731", and offering farmer names is the byte-similar-name
            question this block exists to remove.
            """
            return str(d.get("societyName") or "").strip() or None

        # _fetch_ai_technicians returns None unless BOTH codes are present, and
        # create_ai_call needs all three, so a record missing either has no
        # technician group and cannot be booked at all.
        bookable = [(i, d) for i, d in _numbered if all(_codes(d))]

        lines.append("- Multiple farmer records are registered on this mobile number.")
        if not bookable:
            lines.append(
                "- None of these records carry the society and union codes needed to book, "
                "so an AI booking cannot be made from them."
            )
            lines.append(
                "- Do NOT ask which farmer name for the AI booking. Tell the caller their "
                "details are not available right now."
            )
        else:
            if len(bookable) < len(_numbered):
                lines.append(
                    "- Only these options can be used for an AI booking (the rest are missing "
                    f"society or union codes): {', '.join(f'Farmer option {i}' for i, _ in bookable)}."
                )
            # One entry per village, carrying the first speakable label and the
            # lowest option number in it — the model should never have to infer
            # the mapping from the option lines' society names.
            by_village: dict = {}
            for i, d in bookable:
                key = _village(d)
                entry = by_village.setdefault(key, {"label": None, "option": i})
                if entry["label"] is None:
                    entry["label"] = _village_label(d)
                entry["option"] = min(entry["option"], i)
            first_option = min(i for i, _ in bookable)
            labels = [e["label"] for e in by_village.values()]
            spoken = {_normalize_farmer_name(l) for l in labels if l}
            if len(by_village) == 1:
                names = [_normalize_farmer_name(d.get("farmerName")) for _, d in bookable]
                all_named = len(set(names)) == 1 and all(names)
                lines.append(
                    "- For AI booking, they are all in the same village, served by the same "
                    "technicians, so the visit is identical whichever record is used."
                    + (" They are duplicate records of one farmer." if all_named else "")
                )
                lines.append(
                    f"- Do NOT ask which farmer name for the AI booking. Use Farmer option {first_option}."
                )
            elif all(labels) and len(spoken) == len(labels):
                # Distinctness judged as the caller hears it: "RAMOS" and
                # "Ramos " are two schema entries but one spoken choice.
                choices = ", ".join(
                    f"{e['label']} (Farmer option {e['option']})" for e in by_village.values()
                )
                lines.append(
                    "- For AI booking, they are in different villages, which means different technicians."
                )
                lines.append(
                    f"- Ask which village the animal is in, then use the option named here: {choices}. "
                    "Do NOT ask which farmer name for the AI booking."
                )
            else:
                lines.append(
                    "- For AI booking, these records may be in different villages, but they "
                    "cannot be told apart by village name."
                )
                lines.append(
                    f"- Do NOT ask which farmer name for the AI booking. Use Farmer option {first_option}."
                )

    # No cap: a farmer omitted here cannot be selected for booking, and its
    # technician group in _build_ai_technician_summary becomes unreachable.
    for index, record in enumerate(envelope.farmers, start=1):
        record_data = record.model_dump()
        farmer_name = record_data.get("farmerName") or "Unknown farmer"
        society_name = record_data.get("societyName") or "Unknown society"
        farmer_code = record_data.get("farmerCode")
        union_code = record_data.get("unionCode") or record_data.get("union_code")
        society_code = record_data.get("societyCode") or record_data.get("society_code")
        lines.append(
            f"- Farmer option {index}: name={farmer_name}, society_name={society_name}, "
            f"farmer_code={farmer_code}, union_code={union_code}, society_code={society_code}"
        )

    _append_animal_records(lines, envelope)

    return "\n".join(lines)


SUPPORTED_SCHEME_CONTEXT_UNIONS = SUPPORTED_SCHEME_UNIONS


def _collect_farmer_unions(envelope: Optional[FarmerDataEnvelope]) -> list[str]:
    if envelope is None:
        return []

    seen: set[str] = set()
    unions: list[str] = []
    for farmer in envelope.farmers:
        record = farmer.model_dump()
        raw_union = record.get("unionName") or record.get("union_name")
        normalized_union = str(raw_union or "").strip().lower()
        if not normalized_union or normalized_union in seen:
            continue
        seen.add(normalized_union)
        unions.append(normalized_union)
    return unions


def _collect_farmer_accounts(envelope: Optional[FarmerDataEnvelope]) -> list[FarmerAccount]:
    """Extract every (union, society, farmer) account on the caller's mobile.

    A mobile can map to multiple PashuGPT accounts (e.g. a cow account and a
    buffalo account). The milk-collection tool fans out over all of these so
    a farmer's data is never missed because the agent picked one account.
    Deduplicated on (union_code, society_code, farmer_code).
    """
    if envelope is None:
        return []

    seen: set[tuple] = set()
    accounts: list[FarmerAccount] = []
    for farmer in envelope.farmers:
        record = farmer.model_dump()
        union_code = record.get("unionCode") or record.get("union_code")
        society_code = record.get("societyCode") or record.get("society_code")
        farmer_code = record.get("farmerCode") or record.get("farmer_code")
        if not (union_code and society_code and farmer_code):
            continue
        key = (str(union_code), str(society_code), str(farmer_code))
        if key in seen:
            continue
        seen.add(key)
        accounts.append(
            FarmerAccount(
                union_code=str(union_code),
                society_code=str(society_code),
                farmer_code=str(farmer_code),
                farmer_name=record.get("farmerName") or record.get("farmer_name"),
                society_name=record.get("societyName") or record.get("society_name"),
            )
        )
    return accounts


async def _build_union_scheme_summary(farmer_unions: list[str]) -> str:
    scheme_unions = resolve_supported_unions(farmer_unions, SUPPORTED_SCHEME_CONTEXT_UNIONS)
    if not scheme_unions:
        return ""

    lines = [
        "",
        "## Union schemes available",
        "- The following scheme titles are available from the union scheme cache. Use these titles and links for scheme-related questions. Retrieve full cached scheme details when the user asks about a specific scheme.",
    ]
    for union_name in scheme_unions:
        try:
            records = await get_cached_scheme_records_for_union(union_name)
        except SchemeDependencyError:
            logger.warning("Union scheme summary skipped because Redis dependency is unavailable union=%s", union_name)
            lines.append(f"- **{union_name.title()}**: Scheme cache dependency is unavailable.")
            continue
        except SchemeCacheError:
            logger.warning("Union scheme summary skipped because scheme cache could not be read union=%s", union_name)
            lines.append(f"- **{union_name.title()}**: Scheme cache could not be read.")
            continue
        except Exception as exc:
            logger.warning("Union scheme summary skipped because of unexpected error union=%s error=%s", union_name, exc)
            lines.append(f"- **{union_name.title()}**: Scheme list is temporarily unavailable.")
            continue

        if not records:
            lines.append(f"- **{union_name.title()}**: No cached scheme list is available yet.")
            continue

        lines.append(f"- **{union_name.title()} union schemes:**")
        seen_links: set[tuple[str, str]] = set()
        for record in records:
            title = record.get("scheme_title")
            link = record.get("scheme_url")
            if not title or not link:
                continue
            dedupe_key = (str(title).casefold(), str(link))
            if dedupe_key in seen_links:
                continue
            seen_links.add(dedupe_key)
            lines.append(f"  - {title}: {link}")
    return "\n".join(lines)


def _dedupe_technicians(technicians: list[dict]) -> list[dict]:
    """Drop duplicate technician rows, preserving order.

    The upstream GetAITUserDetailsBySocietyCode endpoint can return the same
    technician more than once; chat dedupes identically in
    agents/farmer_context.py. Keyed on userId, falling back to name+mobile when
    the id is absent.
    """
    unique: dict[str, dict] = {}
    for technician in technicians:
        key = technician.get("userId") or (
            f"{technician.get('fullName')}|{technician.get('mobileNumber')}"
        )
        unique.setdefault(key, technician)
    return list(unique.values())


def _farmer_record_union_name(record: FarmerRecord) -> str | None:
    data = record.model_dump()
    union_name = data.get("unionName") or data.get("union_name")
    return union_name if isinstance(union_name, str) else None


def _farmer_record_identity(data: dict) -> tuple[str, str, str]:
    return (
        str(data.get("farmerCode") or data.get("farmer_code") or ""),
        str(data.get("societyCode") or data.get("society_code") or ""),
        str(data.get("unionCode") or data.get("union_code") or ""),
    )


def _technician_group_is_banned(
    group: dict,
    banned_identities: set[tuple[str, str, str]],
    banned_union_codes: set[str],
) -> bool:
    """True when this cached group belongs to a union banned from AI-call booking.

    Identity (farmerCode, societyCode, unionCode) is the primary match so a mixed
    mobile keeps Kaira technicians. Farmer+society and union-code fallbacks hide
    leftover Kutch groups when some codes on the group are incomplete.
    """
    identity = (
        str(group.get("farmerCode") or group.get("farmer_code") or ""),
        str(group.get("societyCode") or group.get("society_code") or ""),
        str(group.get("unionCode") or group.get("union_code") or ""),
    )
    if identity != ("", "", "") and identity in banned_identities:
        return True
    farmer_society = (identity[0], identity[1])
    if farmer_society != ("", "") and any(
        (farmer_code, society_code) == farmer_society
        for farmer_code, society_code, _union_code in banned_identities
    ):
        return True
    union_code = identity[2]
    return bool(union_code) and union_code in banned_union_codes


def _append_ai_call_union_ban_lines(lines: list[str], farmer: FarmerRecord) -> None:
    data = farmer.model_dump()
    farmer_name = data.get("farmerName") or data.get("farmer_name") or "Unknown farmer"
    society_name = data.get("societyName") or data.get("society_name") or "Unknown society"
    _, society_code, union_code = _farmer_record_identity(data)
    lines.append(
        f"- Technician group: farmer_name={farmer_name}, society_name={society_name}, "
        f"union_code={union_code or None}, society_code={society_code or None}"
    )
    lines.append("- AI call booking is not allowed for this union.")
    lines.append(f"- Tell the farmer: `{UNION_BANNED_MESSAGE}`")
    lines.append("- Do not ask which technician they want. Do not call `create_ai_call`.")


def _build_ai_technician_summary(envelope: Optional[FarmerDataEnvelope]) -> str:
    if envelope is None:
        return ""

    banned_farmers: list[FarmerRecord] = []
    banned_identities: set[tuple[str, str, str]] = set()
    banned_union_codes: set[str] = set()
    for farmer in envelope.farmers or []:
        if not is_ai_call_banned_union(_farmer_record_union_name(farmer)):
            continue
        banned_farmers.append(farmer)
        data = farmer.model_dump()
        identity = _farmer_record_identity(data)
        if identity != ("", "", ""):
            banned_identities.add(identity)
        if identity[2]:
            banned_union_codes.add(identity[2])

    if banned_farmers:
        logger.info(
            "Skipping AI technician context; union is banned from AI-call booking unions=%s",
            [_farmer_record_union_name(farmer) for farmer in banned_farmers],
        )

    all_farmers_banned = bool(envelope.farmers) and len(banned_farmers) == len(envelope.farmers)
    technician_groups = [] if all_farmers_banned else [
        group
        for group in (envelope.aiTechnicians or [])
        if not _technician_group_is_banned(group, banned_identities, banned_union_codes)
    ]
    lines: list[str] = []
    for farmer in banned_farmers:
        _append_ai_call_union_ban_lines(lines, farmer)

    if technician_groups:
        lines.append("- AI technician options for booking are internal context, not user-provided information.")
        lines.append("- The caller does not know which AI technicians are available unless you tell them by technician name.")
        lines.append("- AI technician options for booking are grouped by farmer and society.")
        lines.append("- Each technician option only has these fields: id, full_name, mobile_number.")
        lines.append("- When asking the farmer to choose a technician, use the technician full name in natural spoken form.")
        lines.append("- Do not ask by technician position, number, option index, or ordinal words such as first, second, or third.")
        lines.append("- Mention phone only if a disambiguating mobile number is needed.")
        # Every group and every technician is listed. A cap here is silent: the
        # model cannot offer a technician it never saw, nor match one the caller
        # names, and nothing marks the list as partial (AMUL-39).
        for group in technician_groups:
            farmer_name = group.get("farmerName") or "Unknown farmer"
            society_name = group.get("societyName") or "Unknown society"
            society_code = group.get("societyCode")
            union_code = group.get("unionCode")
            lines.append(
                f"- Technician group: farmer_name={farmer_name}, society_name={society_name}, "
                f"union_code={union_code}, society_code={society_code}"
            )
            technicians = _dedupe_technicians(group.get("technicians") or [])
            if not technicians:
                if group.get("lookupFailed"):
                    # Distinct from "none exist": the lookup errored, so the
                    # agent must not assert the society has no technicians.
                    lines.append(
                        "- AI technician option: could not be retrieved for this farmer group "
                        "right now; say technician details are temporarily unavailable and ask "
                        "the caller to try again later. Do not say the society has no technicians."
                    )
                else:
                    lines.append("- AI technician option: none available for this farmer group.")
                continue
            for technician in technicians:
                name = technician.get("fullName")
                mobile = technician.get("mobileNumber")
                user_id = technician.get("userId")
                option = "- AI technician option:"
                if user_id:
                    option += f" id={user_id}"
                if name:
                    option += f" full_name={name}"
                if mobile:
                    option += f", mobile_number={mobile}"
                lines.append(option)
    elif not banned_farmers:
        # Say what NOT to do, like the per-group branches above: with only the
        # terse line the model offered the prompt's own example names as if real.
        lines.append("- AI technician options for booking are not available in the current signed-in context.")
        lines.append("- Do NOT name any technician, and do not use a technician name from the instructions or an example.")
    return "\n".join(lines)


def should_translate_batch(
    batch_text: str,
    word_count: int,
    is_first_batch: bool = False,
) -> bool:
    """Decide whether the accumulated batch should be flushed for translation."""
    text_end = batch_text.rstrip()
    ends_sentence = text_end.endswith(('.', '!', '?', ':'))

    # Phase 1: first batch — get first audio to the caller fast.
    if is_first_batch:
        return ends_sentence and word_count >= 3

    # Phase 2: subsequent batches — balance quality vs latency.
    if len(batch_text) >= VOICE_TRANSLATION_BATCH_CHAR_LIMIT:
        return True
    if word_count >= 40:
        return True  # force flush, don't hoard

    if word_count < 8:
        return ends_sentence and word_count >= 5

    # 8-40 words: flush on any natural boundary.
    if ends_sentence:
        return True
    if text_end.endswith('\n\n'):
        return True
    if text_end.endswith('\n') and len(batch_text.split('\n')) > 1:
        last_line = batch_text.rstrip('\n').split('\n')[-1].strip()
        if last_line.startswith(('-', '*', '•')) or re.match(r'^\d+\.', last_line):
            return True
    return False

# Langfuse Sessions: same session_id groups all traces for one conversation (session replay, session-level metrics).
def _langfuse_session_context(session_id: str, user_id: str, process_id: Optional[str] = None):
    """Set Langfuse session_id so all agent runs for this conversation appear under one Session."""
    try:
        from app.observability import langfuse_client
        from langfuse import propagate_attributes
        if langfuse_client is None:
            return nullcontext()
        # Langfuse Sessions: session_id ≤200 chars (US-ASCII); same ID = one Session in Langfuse UI
        safe_session_id = (session_id or "").strip()[:200]
        kwargs = dict(
            session_id=safe_session_id or None,
            user_id=(user_id or "anonymous")[:200],
        )
        if process_id:
            kwargs["metadata"] = {"process_id": str(process_id)[:200]}
        return propagate_attributes(**kwargs)
    except Exception:
        return nullcontext()


async def stream_voice_message(
    query: str,
    session_id: str,
    source_lang: str,
    target_lang: str,
    user_id: str,
    history: list,
    provider: Optional[Literal['RAYA']] = None,
    process_id: Optional[str] = None,
    user_info: dict = None,
    owner: Optional[SessionRequestOwner] = None,
    http_request: Optional[Request] = None,
    trace: Optional[VoiceTrace] = None,
    pipeline_profile: str = "managed",
    call_type: str = "inbound",
#    background_tasks: BackgroundTasks,

) -> AsyncGenerator[str, None]:
    """Async generator for streaming chat messages."""
    request_started_at = time.monotonic()
    # Model selection is resolved by the unified pipeline (the only path): the agent
    # handle, provider, and display model name all come from the resolved primary
    # AGENT tier for this session's profile NAME. For the current env this is the same
    # provider/base_url/model the removed get_model_for_variant/provider_for_variant
    # returned, generalized to the weighted-profile split.
    _agent_tier = _llm_resolver.primary_tier(_LlmStep.AGENT, pipeline_profile)
    # oss-vs-managed behavioural split from the resolved AGENT primary tier KIND, not
    # a variant string: a vllm/self-hosted primary (gemma, qwen, ...) -> kind "oss";
    # a managed provider (openai/anthropic/gemini) -> "managed". With the 2-way
    # env-shim (profile named oss/managed) this equals the old
    # ``pipeline_variant == "oss"`` bit exactly. Retained downstream for tier-kind
    # telemetry labels + pretranslation dispatch (it no longer selects the model).
    is_oss = _agent_tier.kind == "oss"
    request_model = _agent_tier.handle
    request_provider = _agent_tier.provider
    request_model_name = _agent_tier.model_name
    last_owner_refresh_at = 0.0
    last_emitted_sig_char: str | None = None
    trace = trace or create_voice_trace(
        session_id=session_id,
        user_id=user_id,
        query=query,
        source_lang=source_lang,
        target_lang=target_lang,
        provider=provider,
        process_id=process_id,
    )
    # Tag the trace with the resolved pipeline variant so Langfuse dashboards
    # can filter sessions by variant. The categorical *score* is emitted
    # below from inside `trace.request_context()`, where a Langfuse trace
    # context is active — emitting it here would silently no-op
    # (Langfuse v4: "Operations that depend on an active span will be
    # skipped"; mirror of amul-oan-api#70).
    call_type = _outbound.normalize_call_type(call_type)
    try:
        trace.metadata["pipeline_profile"] = pipeline_profile
        trace.metadata["request_model"] = request_model_name
        trace.metadata["request_provider"] = request_provider
        trace.metadata["call_type"] = call_type
    except Exception:  # pragma: no cover - never break the call
        pass
    # Serialize the resolved pipeline config into COMPACT flat keys and merge them
    # into trace.metadata (which rides along in metadata=self.metadata on the root
    # observation — the path that lands; this SDK has no update_current_trace, and a
    # big nested blob is OTEL-attribute size-capped). Adds `pipeline_profile`,
    # `pipeline_flags`, and one `pc_<step>` per step. Full static config is in the
    # `llm_core.full_config` boot log. Best-effort — never breaks the call.
    try:
        from app.llm_core import trace as _pipeline_trace
        from app.llm_core import resolver as _lr, runtime as _lrt
        from app.llm_core.config_model import Step as _LS
        _pt = _pipeline_trace.begin(pipeline_profile)
        _pipeline_trace.populate(
            _pt, _lrt.get_pipeline(), _lr.primary_tier, pipeline_profile,
            (_LS.PRE_TRANSLATION, _LS.MODERATION, _LS.NON_MEANINGFUL, _LS.AGENT, _LS.POST_TRANSLATION),
        )
        _pipeline_trace.add_compact_metadata(_pt, trace.metadata)
    except Exception as _pt_exc:  # pragma: no cover - tracing must never break the call
        logger.debug("pipeline_config populate skipped: %s", _pt_exc)
    logger.info(
        "voice request_variant session_id=%s variant=%s model=%s provider=%s",
        session_id,
        pipeline_profile,
        request_model_name,
        request_provider,
    )

    async def _request_is_stale(reason: str) -> bool:
        nonlocal last_owner_refresh_at
        if http_request is not None and await http_request.is_disconnected():
            trace.set_outcome("client_disconnected")
            logger.info(
                "Stopping request due to client disconnect - session_id=%s process_id=%s reason=%s",
                session_id,
                process_id,
                reason,
            )
            return True

        now = time.monotonic()
        if owner is not None and (
            last_owner_refresh_at == 0.0
            or now - last_owner_refresh_at >= settings.session_owner_refresh_interval_seconds
        ):
            refreshed = await refresh_session_request_ownership(owner)
            last_owner_refresh_at = now
            if not refreshed:
                trace.set_outcome("stale_request")
                logger.info(
                    "Stopping stale request after ownership lost during refresh - session_id=%s process_id=%s epoch=%s reason=%s",
                    session_id,
                    process_id,
                    owner.epoch,
                    reason,
                )
                return True

        if owner is not None and not await is_session_request_owner(owner):
            trace.set_outcome("stale_request")
            logger.info(
                "Stopping stale request because a newer request owns the session - session_id=%s process_id=%s epoch=%s reason=%s",
                session_id,
                process_id,
                owner.epoch,
                reason,
            )
            return True

        return False

    def _first_sig_char(text: str) -> str | None:
        for ch in text or "":
            if not ch.isspace():
                return ch
        return None

    def _last_sig_char(text: str) -> str | None:
        for ch in reversed(text or ""):
            if not ch.isspace():
                return ch
        return None

    def _prepare_translated_emit(text: str) -> str:
        nonlocal last_emitted_sig_char
        if not isinstance(text, str) or not text:
            return text

        first_sig = _first_sig_char(text)
        if (
            last_emitted_sig_char in {".", "!", "?", "।"}
            and first_sig is not None
            and re.match(r"[A-Za-z\u0A80-\u0AFF]", first_sig)
            and not text[0].isspace()
        ):
            text = " " + text

        last_sig = _last_sig_char(text)
        if last_sig is not None:
            last_emitted_sig_char = last_sig
        return text

    def _emit(text: str, *, kind: str = "assistant") -> str:
        trace.record_emit(text, kind=kind)
        return text

    try:
        # Keep the Langfuse root observation open for the full streaming
        # generator so downstream model calls and pydantic-ai spans nest under
        # this agent_journey.
        with trace.request_context():
            # Emit the per-session pipeline_profile categorical score from
            # *inside* the trace context (chat #70 fix). score_id is
            # deterministic per session so subsequent voice turns in the
            # same session upsert the same score (no duplicates).
            if _get_langfuse_client is not None:
                try:
                    _lf = _get_langfuse_client()
                    _lf.score_current_trace(
                        name="pipeline_profile",
                        value=pipeline_profile,
                        data_type="CATEGORICAL",
                        score_id=f"voice-variant-{(session_id or '')[:180]}",
                        comment="Sticky pipeline variant for this voice session",
                    )
                except Exception as e:  # pragma: no cover
                    logger.debug("Langfuse: voice pipeline_profile score failed: %s", e)
            requested_source_lang = (source_lang or "gu").strip().lower()
            requested_target_lang = (target_lang or "gu").strip().lower()
            trace.set_language(requested_source_lang, requested_target_lang)
            needs_output_translation = requested_target_lang in INDIAN_LANGUAGES and requested_target_lang not in {"en", "english"}
            nudge_lang = (requested_target_lang or "en").strip().lower()
            has_meaningful_history = _has_meaningful_history(history)

            # ── Outbound consent gate ─────────────────────────────────────
            # RAYA speaks the opening line on a call we placed, not us. By the
            # time the first request reaches this service the farmer has already
            # heard the intro and answered it — so the FIRST outbound turn
            # carries the consent reply. We classify that reply; we never send an
            # intro of our own, or the farmer hears two.
            #
            # The stage is read from Redis (not history) so it survives trimming,
            # and is read on every turn while the feature flag is on because Raya
            # is not guaranteed to re-stamp call_type after the first request.
            outbound_stage = (
                await _outbound.get_stage(session_id)
                if settings.outbound_intro_enabled
                else None
            )
            # The turn answering Raya's opening question is owned by the consent
            # gate: the greeting / fragment / identity fast paths must not preempt
            # it. "હા બોલો" — the single most likely affirmative reply — is itself a
            # _GREETING_TOKENS entry, so without this an affirmative reply could be
            # answered with the generic greeting and the readout lost.
            outbound_consent_turn = settings.outbound_intro_enabled and (
                (
                    _outbound.is_outbound(call_type)
                    and outbound_stage is None
                    and not history
                )
                # Sessions opened by our own (now removed) intro that are still
                # mid-call across the deploy — they are already past the intro, so
                # their next turn is still the consent reply.
                or outbound_stage == _outbound.STAGE_INTRO_SENT
            )
            if outbound_consent_turn:
                logger.info(
                    "Outbound consent turn; session_id=%s process_id=%s user_id=%s query=%r",
                    session_id, process_id, user_id, (query or "")[:60],
                )
                # Warm the milk summary alongside the consent classifier so an
                # affirmative reply does not pay the upstream lookup serially.
                # There is no longer a turn of our own to hide it behind. Failure
                # just means the agent fetches it itself.
                _prefetch_mobile = normalize_phone_to_mobile(user_id)
                if _prefetch_mobile:
                    async def _warm_outbound_milk(mobile_number: str = _prefetch_mobile):
                        envelope = await get_or_fetch_farmer_data(mobile_number)
                        await _outbound.prefetch_milk_summary(
                            session_id, _collect_farmer_accounts(envelope),
                        )
                    _outbound.spawn(_warm_outbound_milk(), label="outbound_milk_prefetch")

            # ── STT signal handling (no-audio / unclear speech) ─────────────
            # These are not real user messages — skip translation & agent,
            # generate a short contextual "please repeat" via GPT-5-mini.
            stt_signal = detect_stt_signal(query)
            if stt_signal is not None:
                trace.set_route("stt_signal")
                logger.info(
                    "STT signal detected; session_id=%s process_id=%s signal=%s",
                    session_id,
                    process_id,
                    stt_signal,
                )
                recent_text = "\n\n".join(format_message_pairs(history, 3))
                if await _request_is_stale("before_stt_signal_response"):
                    return
                prior_stt_failures = count_consecutive_stt_signals(history)
                final_attempt = (prior_stt_failures + 1) >= max(1, settings.stt_signal_retry_ceiling)
                with trace.stage("stt_signal_response", as_type="generation"):
                    stt_response = await generate_stt_signal_response(
                        signal=stt_signal,
                        target_lang=requested_target_lang,
                        recent_history_text=recent_text,
                        final_attempt=final_attempt,
                    )
                history_signal = (
                    _canonical_history_user_text("stt_no_audio")
                    if stt_signal == "No audio/User is speaking softly"
                    else _canonical_history_user_text("stt_unclear")
                )
                history_response = _FRAGMENT_RESPONSES["en"] if not final_attempt else "Sorry, I still could not hear you clearly. Please try again later."
                stt_req, stt_resp = _history_pair(history_signal, history_response)
                with trace.stage("history_write"):
                    await update_message_history(session_id, [*history, stt_req, stt_resp])
                trace.set_outcome("stt_signal")
                yield _emit(_prepare_voice_output(stt_response, requested_target_lang))
                return

            # ── Hold message short-circuit ────────────────────────────────
            # Carrier IVR "your call is on hold" messages get transcribed by
            # STT and sent as user input, creating runaway loops of 20+ traces.
            # Respond with "goodbye" so the STT provider disconnects the call.
            if _is_hold_message(query):
                trace.set_route("hold_message")
                logger.info(
                    "Hold message detected; responding with goodbye to cut call - session_id=%s process_id=%s query=%r",
                    session_id, process_id, query[:100],
                )
                goodbye = TELEPHONY_TERMINATE_CALL_TOKEN.get(
                    requested_target_lang,
                    TELEPHONY_TERMINATE_CALL_TOKEN["en"],
                )
                trace.set_outcome("hold_message")
                # Telephony hangup token must remain exact ASCII "Goodbye.".
                # Do not pass through language cleanup (Gujarati filter would
                # strip Latin letters and leave only ".").
                yield _emit(goodbye)
                return

            # ── Greeting short-circuit ────────────────────────────────────
            # Bare greetings ("hello", "હલો", "હા") should not trigger the
            # full agent pipeline or a nudge.  Respond immediately.
            # When translation pipeline is active, let greetings flow through
            # the normal agent pipeline so history stays in English.
            if _is_bare_greeting(query) and not has_meaningful_history and not outbound_consent_turn:
                trace.set_route("greeting_fast_path")
                logger.info(
                    "Bare greeting detected; short-circuiting - session_id=%s process_id=%s query=%r",
                    session_id, process_id, query,
                )
                greeting_history = _GREETING_RESPONSES["en"]
                with trace.stage("greeting_fast_path"):
                    greeting_response = await _canned_for_caller(greeting_history, requested_target_lang, _GREETING_RESPONSES)
                greet_req, greet_resp = _history_pair(_canonical_history_user_text("greeting"), greeting_history)
                with trace.stage("history_write"):
                    await update_message_history(session_id, [*history, greet_req, greet_resp])
                trace.set_outcome("greeting_fast_path")
                yield _emit(_prepare_voice_output(greeting_response, requested_target_lang))
                return

            # ── Identity fast-path ────────────────────────────────────────
            # Pure identity queries ("What is your name?", "What is this service?")
            # should return the canonical Sarlaben identity line directly
            # without running the full agent pipeline.
            if _fast_path_kind_for_query(query) == "identity" and not has_meaningful_history and not outbound_consent_turn:
                trace.set_route("identity_fast_path")
                logger.info(
                    "Identity fast-path triggered; session_id=%s process_id=%s query=%r",
                    session_id, process_id, query,
                )
                identity_resp_en = _IDENTITY_RESPONSE_EN
                with trace.stage("identity_fast_path"):
                    identity_resp_for_caller = await _render_text_for_caller(identity_resp_en, requested_target_lang)
                id_req, id_resp = _history_pair(_canonical_history_user_text("greeting"), identity_resp_en)
                with trace.stage("history_write"):
                    await update_message_history(session_id, [*history, id_req, id_resp])
                trace.set_outcome("identity_fast_path")
                yield _emit(_prepare_voice_output(identity_resp_for_caller, requested_target_lang))
                return

            # ── Fragment short-circuit ────────────────────────────────────
            # Very short / garbled input (≤3 chars) that isn't a greeting or
            # STT signal — ask the farmer to repeat instead of routing to agent.
            if _is_fragment_query(query) and not has_meaningful_history and not outbound_consent_turn:
                trace.set_route("fragment_fast_path")
                logger.info(
                    "Fragment query detected; short-circuiting - session_id=%s process_id=%s query=%r",
                    session_id, process_id, query,
                )
                frag_response_for_history = _FRAGMENT_RESPONSES["en"]
                with trace.stage("fragment_fast_path"):
                    frag_response_for_caller = await _canned_for_caller(frag_response_for_history, requested_target_lang, _FRAGMENT_RESPONSES)
                frag_req, frag_resp = _history_pair(_canonical_history_user_text("fragment"), frag_response_for_history)
                with trace.stage("history_write"):
                    await update_message_history(session_id, [*history, frag_req, frag_resp])
                trace.set_outcome("fragment_fast_path")
                yield _emit(_prepare_voice_output(frag_response_for_caller, requested_target_lang))
                return

            # ── Nudge: arm BEFORE any pre-processing ────────────────────────
            # Fires on whichever happens first:
            #   (a) the configured timer expires, OR
            #   (b) the LLM invokes a tool (signalled via tool_call_event).
            # Cancelled if first text/translated chunk reaches the client first.
            nudge_task = None
            if settings.enable_voice_nudges:
                trace.set_nudge(armed=True, sent=False)
                nudge_sent = False
                tool_call_event = asyncio.Event()
                set_tool_call_nudge_event(tool_call_event)

                async def send_nudge_on_trigger() -> None:
                    nonlocal nudge_sent
                    try:
                        elapsed = max(0.0, time.monotonic() - request_started_at)
                        remaining = max(0.0, float(settings.nudge_timeout_seconds) - elapsed)
                        logger.info(
                            "Nudge armed; session_id=%s process_id=%s elapsed=%.3fs remaining=%.3fs timeout=%.3fs",
                            session_id,
                            process_id,
                            elapsed,
                            remaining,
                            settings.nudge_timeout_seconds,
                        )

                        # Wait for EITHER the timer OR a tool-call signal
                        timer_task = asyncio.create_task(asyncio.sleep(remaining))
                        event_task = asyncio.create_task(tool_call_event.wait())
                        done, pending = await asyncio.wait(
                            {timer_task, event_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for t in pending:
                            t.cancel()

                        trigger_reason = "tool_call" if event_task in done else "timeout"
                        if await _request_is_stale("before_nudge_send"):
                            return
                        if nudge_sent:
                            return
                        nudge_sent = True
                        trace.set_nudge(
                            sent=True,
                            trigger=trigger_reason,
                            sent_ms=round((time.monotonic() - request_started_at) * 1000.0, 2),
                        )
                        nudge_msg = (
                            get_tool_nudge_message(nudge_lang)
                            if trigger_reason == "tool_call"
                            else get_timeout_nudge_message(nudge_lang)
                        )
                        await send_nudge_message_raya(nudge_msg, session_id, process_id)
                        elapsed = max(0.0, time.monotonic() - request_started_at)
                        logger.info(
                            "Nudge sent (%s); session_id=%s process_id=%s total_elapsed=%.3fs",
                            trigger_reason,
                            session_id,
                            process_id,
                            elapsed,
                        )
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.warning(
                            "Nudge task failed; session_id=%s process_id=%s error=%s",
                            session_id,
                            process_id,
                            e,
                        )

                nudge_task = asyncio.create_task(send_nudge_on_trigger())
                logger.info(
                    "Nudge initiated; session_id=%s process_id=%s",
                    session_id,
                    process_id,
                )
            else:
                trace.set_nudge(armed=False, sent=False)
                logger.info(
                    "Voice nudges disabled by config; session_id=%s process_id=%s",
                    session_id,
                    process_id,
                )
            # ── End nudge setup ─────────────────────────────────────────────

            processing_query = query
            processing_lang = "en"
            history_user_text = query
            moderation_recent_history = "\n\n".join(format_message_pairs(history, 2))
            non_meaningful_recent_turns = _collect_recent_user_turns_for_non_meaningful(history, query, limit=5)
            mobile = normalize_phone_to_mobile(user_id)
            signed_in = _is_signed_in_session(user_info, user_id)
            # Fails closed: a turn whose farmer lookup never ran (no resolvable
            # mobile) or threw stays "unresolved", so the identity-taking tools
            # stay hidden. The context lines are seeded here too — otherwise the
            # model would find the tools simply absent, with nothing to tell the
            # caller, which is how a small model ends up claiming a booking it
            # never made. Both are replaced below once the lookup resolves.
            farmer_identity = "unresolved"
            farmer_info = "\n".join(unavailable_capability_lines(farmer_identity))
            farmer_unions: list[str] = []
            farmer_accounts: list[FarmerAccount] = []
            ai_technician_info = ""
            farmer_cache_task = (
                asyncio.create_task(get_or_fetch_farmer_data(mobile))
                if mobile
                else None
            )

            # Kick off content moderation in parallel with pretranslation.
            # Moderation receives the raw native-language text so it does
            # not need to wait for pretranslation to finish.
            moderation_started_at = time.monotonic()
            # Stamp the true completion time so deferring the await (below) does
            # not inflate the reported moderation latency: the verdict is now
            # consumed after the agent's prefill, which can be later than when the
            # moderation call actually finished.
            moderation_done_at: dict = {"t": None}
            moderation_task = asyncio.create_task(
                check_moderation(
                    text=query,
                    source_lang=requested_source_lang,
                    recent_history_text=moderation_recent_history,
                    profile_name=pipeline_profile,
                    session_id=session_id,
                    user_id=user_id,
                    process_id=process_id or "",
                    pipeline_profile=pipeline_profile,
                )
            )
            moderation_task.add_done_callback(
                lambda _t: moderation_done_at.__setitem__("t", time.monotonic())
            )
            non_meaningful_started_at = time.monotonic()
            non_meaningful_done_at: dict = {"t": None}
            non_meaningful_task = asyncio.create_task(
                check_non_meaningful_streak(
                    user_turns=non_meaningful_recent_turns,
                    source_lang=requested_source_lang,
                )
            )
            non_meaningful_task.add_done_callback(
                lambda _t: non_meaningful_done_at.__setitem__("t", time.monotonic())
            )

            # Outbound consent gate. Started here so the classifier runs under
            # pretranslation and the farmer-context fetch rather than after them;
            # it reads the raw native-language reply, so it needs neither. The
            # verdict is consumed before the agent input is built (below),
            # because an affirmative reply changes that input.
            consent_task = (
                asyncio.create_task(
                    classify_consent(reply=query, source_lang=requested_source_lang)
                )
                if outbound_consent_turn
                else None
            )

            if requested_source_lang not in {"en", "english"}:
                # Pretranslation tier decision (provider label + model + tier) comes
                # from the resolved PRE_TRANSLATION primary tier (the only path). For
                # the current env this equals the removed `is_oss` toggle exactly (an
                # OSS session's primary is the vLLM/OSS-model tier, else the
                # managed/OpenAI-model tier). The per-attempt dispatch below routes
                # to the OSS vs managed pretranslation twin by the tier kind.
                _pre_mt = _llm_resolver.primary_tier(_LlmStep.PRE_TRANSLATION, pipeline_profile)
                _pretrans_provider_label = "vllm" if _pre_mt.kind == "oss" else "openai"
                _pretrans_model = _pre_mt.model_name
                _pretrans_requested_tier = _pre_mt.kind
                _pretranslation_attempts: list[dict[str, object]] = []
                _pretranslation_actual_tier = _pretrans_requested_tier
                _pretranslation_actual_provider = _pretrans_provider_label
                _pretranslation_actual_model = _pretrans_model
                _pretranslation_fallback_used = False
                logger.info(
                    "Translation pipeline enabled; pretranslating %s -> en with %s (variant=%s)",
                    requested_source_lang,
                    _pretrans_model,
                    pipeline_profile,
                )
                if await _request_is_stale("before_query_pretranslation"):
                    moderation_task.cancel()
                    non_meaningful_task.cancel()
                    return
                if settings.fallback_enabled:
                    # Standard OSS -> managed fallback. Drops the legacy TranslateGemma
                    # stopgap (decision #7): TranslateGemma is also self-hosted vLLM, so
                    # it shared a failure domain with the OSS pretranslation it backed up.
                    # The managed tier is the OpenAI pretranslation (translate_to_english_with_gpt5_mini),
                    # which was the pre-OSS primary.
                    try:
                        async def _run_pretranslation_attempt(a):
                            nonlocal _pretranslation_actual_tier, _pretranslation_actual_provider, _pretranslation_actual_model
                            attempt_info: dict[str, object] = {
                                "tier": a.kind,
                                "provider": a.provider,
                                "model": a.model_name,
                                "endpoint": a.endpoint,
                            }
                            _pretranslation_attempts.append(attempt_info)
                            try:
                                if a.kind == "oss":
                                    translated = await translate_to_english_with_oss_vllm(
                                        text=query,
                                        source_lang=requested_source_lang,
                                        session_id=session_id,
                                        user_id=user_id,
                                        process_id=process_id or "",
                                        pipeline_profile=pipeline_profile,
                                    )
                                else:
                                    translated = await translate_to_english_with_gpt5_mini(
                                        text=query,
                                        source_lang=requested_source_lang,
                                        session_id=session_id,
                                        user_id=user_id,
                                        process_id=process_id or "",
                                        pipeline_profile=pipeline_profile,
                                    )
                            except Exception as _attempt_exc:
                                attempt_info["status"] = "error"
                                attempt_info["error_class"] = type(_attempt_exc).__name__
                                attempt_info["error_reason"] = classify(_attempt_exc).value
                                raise
                            attempt_info["status"] = "ok"
                            _pretranslation_actual_tier = a.kind
                            _pretranslation_actual_provider = a.provider
                            _pretranslation_actual_model = a.model_name
                            return translated

                        with trace.stage(
                            "pretranslation",
                            as_type="generation",
                            input=trace.metadata.get("query"),
                            metadata={
                                "provider": _pretrans_provider_label,
                                "source_lang": requested_source_lang,
                                "pipeline_profile": pipeline_profile,
                            },
                            model=_pretrans_model,
                        ):
                            processing_query = await execute_with_fallback(
                                pipeline="pretranslation",
                                session_id=session_id,
                                profile_name=pipeline_profile,
                                run=_run_pretranslation_attempt,
                            )
                        _pretranslation_fallback_used = (
                            len(_pretranslation_attempts) > 1
                            and _pretranslation_attempts[0].get("status") == "error"
                        )
                        trace.set_pretranslation(
                            text=processing_query,
                            provider=_pretrans_provider_label,
                            fallback_used=_pretranslation_fallback_used,
                            requested_tier=_pretrans_requested_tier,
                            requested_provider=_pretrans_provider_label,
                            requested_model=_pretrans_model,
                            actual_tier=_pretranslation_actual_tier,
                            actual_provider=_pretranslation_actual_provider,
                            actual_model=_pretranslation_actual_model,
                            attempts=_pretranslation_attempts,
                        )
                        history_user_text = processing_query or _canonical_history_user_text("low_confidence")
                    except Exception as e:
                        logger.error(
                            "pretranslation failed (all tiers) for session_id=%s source_lang=%s error=%s",
                            session_id,
                            requested_source_lang,
                            e,
                        )
                        processing_query = ""
                        _pretranslation_actual_tier = "failed"
                        _pretranslation_actual_provider = "failed"
                        _pretranslation_actual_model = "failed"
                        _pretranslation_fallback_used = (
                            len(_pretranslation_attempts) > 1
                            and _pretranslation_attempts[0].get("status") == "error"
                        )
                        trace.set_pretranslation(
                            text=processing_query,
                            provider="failed",
                            fallback_used=_pretranslation_fallback_used,
                            requested_tier=_pretrans_requested_tier,
                            requested_provider=_pretrans_provider_label,
                            requested_model=_pretrans_model,
                            actual_tier=_pretranslation_actual_tier,
                            actual_provider=_pretranslation_actual_provider,
                            actual_model=_pretranslation_actual_model,
                            attempts=_pretranslation_attempts,
                        )
                        history_user_text = _canonical_history_user_text("pretranslation_failed")
                else:
                    try:
                        _legacy_primary_attempt = {
                            "tier": _pretrans_requested_tier,
                            "provider": _pretrans_provider_label,
                            "model": _pretrans_model,
                            "status": "started",
                        }
                        _pretranslation_attempts.append(_legacy_primary_attempt)
                        with trace.stage(
                            "pretranslation",
                            as_type="generation",
                            input=trace.metadata.get("query"),
                            metadata={
                                "provider": _pretrans_provider_label,
                                "source_lang": requested_source_lang,
                                "pipeline_profile": pipeline_profile,
                            },
                            model=_pretrans_model,
                        ):
                            if _pretrans_requested_tier == "oss":
                                processing_query = await translate_to_english_with_oss_vllm(
                                    text=query,
                                    source_lang=requested_source_lang,
                                    session_id=session_id,
                                    user_id=user_id,
                                    process_id=process_id or "",
                                    pipeline_profile=pipeline_profile,
                                )
                            else:
                                processing_query = await translate_to_english_with_gpt5_mini(
                                    text=query,
                                    source_lang=requested_source_lang,
                                    session_id=session_id,
                                    user_id=user_id,
                                    process_id=process_id or "",
                                    pipeline_profile=pipeline_profile,
                                )
                        _legacy_primary_attempt["status"] = "ok"
                        _pretranslation_actual_tier = _pretrans_requested_tier
                        _pretranslation_actual_provider = _pretrans_provider_label
                        _pretranslation_actual_model = _pretrans_model
                        trace.set_pretranslation(
                            text=processing_query,
                            provider=_pretrans_provider_label,
                            fallback_used=False,
                            requested_tier=_pretrans_requested_tier,
                            requested_provider=_pretrans_provider_label,
                            requested_model=_pretrans_model,
                            actual_tier=_pretranslation_actual_tier,
                            actual_provider=_pretranslation_actual_provider,
                            actual_model=_pretranslation_actual_model,
                            attempts=_pretranslation_attempts,
                        )
                        history_user_text = processing_query or _canonical_history_user_text("low_confidence")
                    except Exception as e:
                        _legacy_primary_attempt["status"] = "error"
                        _legacy_primary_attempt["error_class"] = type(e).__name__
                        _legacy_primary_attempt["error_reason"] = classify(e).value
                        logger.error(
                            "OpenAI pretranslation failed for session_id=%s source_lang=%s model=%s error=%s",
                            session_id,
                            requested_source_lang,
                            OPENAI_PRETRANSLATION_MODEL,
                            e,
                        )
                        try:
                            logger.info("Falling back to TranslateGemma pretranslation for session_id=%s", session_id)
                            _legacy_fallback_attempt = {
                                "tier": "translategemma",
                                "provider": "translategemma",
                                "model": "translategemma",
                                "status": "started",
                            }
                            _pretranslation_attempts.append(_legacy_fallback_attempt)
                            with trace.stage(
                                "pretranslation_fallback",
                                as_type="generation",
                                input=trace.metadata.get("query"),
                                metadata={"provider": "translategemma", "source_lang": requested_source_lang},
                            ):
                                processing_query = await translate_to_english_with_structured_fallback(
                                    text=query,
                                    source_lang=requested_source_lang,
                                )
                            _legacy_fallback_attempt["status"] = "ok"
                            _pretranslation_actual_tier = "translategemma"
                            _pretranslation_actual_provider = "translategemma"
                            _pretranslation_actual_model = "translategemma"
                            trace.set_pretranslation(
                                text=processing_query,
                                provider="translategemma",
                                fallback_used=True,
                                requested_tier=_pretrans_requested_tier,
                                requested_provider=_pretrans_provider_label,
                                requested_model=_pretrans_model,
                                actual_tier=_pretranslation_actual_tier,
                                actual_provider=_pretranslation_actual_provider,
                                actual_model=_pretranslation_actual_model,
                                attempts=_pretranslation_attempts,
                            )
                            history_user_text = processing_query or _canonical_history_user_text("low_confidence")
                        except Exception as fallback_error:
                            _legacy_fallback_attempt["status"] = "error"
                            _legacy_fallback_attempt["error_class"] = type(fallback_error).__name__
                            _legacy_fallback_attempt["error_reason"] = classify(fallback_error).value
                            logger.error(
                                "TranslateGemma pretranslation fallback failed for session_id=%s error=%s",
                                session_id,
                                fallback_error,
                            )
                            processing_query = ""
                            _pretranslation_actual_tier = "failed"
                            _pretranslation_actual_provider = "failed"
                            _pretranslation_actual_model = "failed"
                            trace.set_pretranslation(
                                text=processing_query,
                                provider="failed",
                                fallback_used=True,
                                requested_tier=_pretrans_requested_tier,
                                requested_provider=_pretrans_provider_label,
                                requested_model=_pretrans_model,
                                actual_tier=_pretranslation_actual_tier,
                                actual_provider=_pretranslation_actual_provider,
                                actual_model=_pretranslation_actual_model,
                                attempts=_pretranslation_attempts,
                            )
                            history_user_text = _canonical_history_user_text("pretranslation_failed")

            else:
                history_user_text = query
                trace.set_pretranslation(
                    text=query,
                    provider="none",
                    fallback_used=False,
                    requested_tier="none",
                    requested_provider="none",
                    requested_model="none",
                    actual_tier="none",
                    actual_provider="none",
                    actual_model="none",
                    attempts=[],
                )

            # ── Content moderation: deferred gate (runs with the agent) ──────
            # check_moderation() was kicked off at the top of the turn and runs
            # concurrently with pretranslation, the farmer-context load, AND the
            # answer agent's prefill/generation below. We deliberately do NOT
            # block on it here. The verdict is resolved lazily — via
            # _resolve_moderation() — only at the points that can emit
            # caller-facing output for a non-fast-path turn:
            #   1. the empty-pretranslation short-circuit, and
            #   2. just before the agent's first streamed chunk is emitted.
            # This takes the ~1.5s moderation call off the critical path on
            # warm-cache turns (cold farmer fetches already hid it). A rejected
            # query is still declined before any answer reaches the caller, and
            # side-effecting booking tools self-gate on the same verdict via
            # deps.ensure_in_scope(), so optimistic agent execution can never turn
            # a rejected query into a real booking write. Fail-open on any
            # unexpected moderation exception — a flaky check must never drop a
            # real farmer call.
            _moderation_resolved = False
            _moderation_verdict: Optional[ModerationVerdict] = None
            _non_meaningful_resolved = False
            _non_meaningful_verdict: Optional[NonMeaningfulVerdict] = None
            _should_gate_non_meaningful = _should_gate_non_meaningful_llm(non_meaningful_recent_turns)

            async def _resolve_moderation() -> Optional[ModerationVerdict]:
                nonlocal _moderation_resolved, _moderation_verdict
                if _moderation_resolved:
                    return _moderation_verdict
                _moderation_resolved = True
                moderation_status = "ok"
                moderation_status_message: Optional[str] = None
                try:
                    _moderation_verdict = await moderation_task
                    done_t = moderation_done_at["t"] or time.monotonic()
                    trace.attach_stage_timing(
                        "moderation",
                        (done_t - moderation_started_at) * 1000.0,
                        source_lang=requested_source_lang,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as moderation_error:
                    moderation_status = "error"
                    moderation_status_message = str(moderation_error)[:300]
                    done_t = moderation_done_at["t"] or time.monotonic()
                    trace.attach_stage_timing(
                        "moderation",
                        (done_t - moderation_started_at) * 1000.0,
                        status="error",
                        source_lang=requested_source_lang,
                    )
                    logger.error(
                        "Moderation task raised unexpectedly for session_id=%s error=%s",
                        session_id,
                        moderation_error,
                    )
                    _moderation_verdict = None
                moderation_duration_ms = (done_t - moderation_started_at) * 1000.0
                trace.set_moderation(_moderation_verdict)
                moderation_payload = trace.metadata.get("moderation", {})
                trace.record_child_observation(
                    name="moderation",
                    as_type="generation",
                    input={
                        "source_lang": requested_source_lang,
                        "text": sanitize_text(query),
                        "recent_history_text": sanitize_text(moderation_recent_history),
                    },
                    output=(
                        {
                            "category": getattr(_moderation_verdict, "category", None),
                            "reason": getattr(_moderation_verdict, "reason", None),
                            "rejected": getattr(_moderation_verdict, "rejected", None),
                            "failed_open": getattr(_moderation_verdict, "failed_open", None),
                            "failed_closed": getattr(_moderation_verdict, "failed_closed", None),
                        }
                        if _moderation_verdict is not None
                        else {"available": False}
                    ),
                    metadata={
                        "duration_ms": round(moderation_duration_ms, 2),
                        "status": moderation_status,
                        "source_lang": requested_source_lang,
                        "pipeline_profile": pipeline_profile,
                        "requested_tier": moderation_payload.get("requested_tier"),
                        "requested_provider": moderation_payload.get("requested_provider"),
                        "requested_model": moderation_payload.get("requested_model"),
                        "actual_tier": moderation_payload.get("actual_tier"),
                        "actual_provider": moderation_payload.get("actual_provider"),
                        "actual_model": moderation_payload.get("actual_model"),
                        "fallback_used": moderation_payload.get("fallback_used"),
                        "attempts": moderation_payload.get("attempts"),
                    },
                    model=moderation_payload.get("actual_model") or moderation_payload.get("requested_model"),
                    level="ERROR" if moderation_status == "error" else "DEFAULT",
                    status_message=moderation_status_message,
                )
                if _moderation_verdict is not None:
                    logger.info(
                        "Moderation verdict: category=%s rejected=%s failed_open=%s reason=%r session_id=%s process_id=%s",
                        _moderation_verdict.category,
                        _moderation_verdict.rejected,
                        _moderation_verdict.failed_open,
                        _moderation_verdict.reason,
                        session_id,
                        process_id,
                    )
                return _moderation_verdict

            async def _resolve_non_meaningful() -> Optional[NonMeaningfulVerdict]:
                nonlocal _non_meaningful_resolved, _non_meaningful_verdict
                if _non_meaningful_resolved:
                    return _non_meaningful_verdict
                _non_meaningful_resolved = True
                try:
                    if not _should_gate_non_meaningful and not non_meaningful_task.done():
                        non_meaningful_task.cancel()
                        try:
                            await non_meaningful_task
                        except asyncio.CancelledError:
                            pass
                        _non_meaningful_verdict = NonMeaningfulVerdict(
                            five_consecutive_non_meaningful=False,
                            reason="gate skipped by heuristic",
                            failed_open=False,
                        )
                        done_t = time.monotonic()
                        trace.attach_stage_timing(
                            "non_meaningful",
                            (done_t - non_meaningful_started_at) * 1000.0,
                            source_lang=requested_source_lang,
                            turn_count=len(non_meaningful_recent_turns),
                            gate_skipped=True,
                        )
                    elif non_meaningful_task.done():
                        _non_meaningful_verdict = await non_meaningful_task
                    else:
                        done, pending = await asyncio.wait(
                            {non_meaningful_task},
                            timeout=max(0.0, settings.voice_non_meaningful_gate_timeout_seconds),
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if not done:
                            for pending_task in pending:
                                pending_task.cancel()
                            # Reap the cancellation. asyncio.wait() does not await
                            # the pending task for us, so without this the classifier
                            # task lingers cancelled-but-unawaited on the normal agent
                            # success path (which never calls the reaper), risking a
                            # "Task was destroyed but it is pending" warning.
                            for pending_task in pending:
                                try:
                                    await pending_task
                                except asyncio.CancelledError:
                                    pass
                            _non_meaningful_verdict = NonMeaningfulVerdict(
                                five_consecutive_non_meaningful=False,
                                reason="gate timeout",
                                failed_open=True,
                            )
                            logger.info(
                                "Non-meaningful gate timed out; fail-open session_id=%s process_id=%s timeout=%.2fs",
                                session_id,
                                process_id,
                                settings.voice_non_meaningful_gate_timeout_seconds,
                            )
                        else:
                            _non_meaningful_verdict = await non_meaningful_task
                    done_t = non_meaningful_done_at["t"] or time.monotonic()
                    trace.attach_stage_timing(
                        "non_meaningful",
                        (done_t - non_meaningful_started_at) * 1000.0,
                        source_lang=requested_source_lang,
                        turn_count=len(non_meaningful_recent_turns),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as non_meaningful_error:
                    done_t = non_meaningful_done_at["t"] or time.monotonic()
                    trace.attach_stage_timing(
                        "non_meaningful",
                        (done_t - non_meaningful_started_at) * 1000.0,
                        status="error",
                        source_lang=requested_source_lang,
                    )
                    logger.error(
                        "Non-meaningful task raised unexpectedly for session_id=%s error=%s",
                        session_id,
                        non_meaningful_error,
                    )
                    _non_meaningful_verdict = NonMeaningfulVerdict(
                        five_consecutive_non_meaningful=False,
                        reason=f"task error: {type(non_meaningful_error).__name__}",
                        failed_open=True,
                    )
                trace.metadata["non_meaningful"] = {
                    "available": _non_meaningful_verdict is not None,
                    "five_consecutive_non_meaningful": bool(
                        getattr(_non_meaningful_verdict, "five_consecutive_non_meaningful", False)
                    ),
                    "failed_open": bool(getattr(_non_meaningful_verdict, "failed_open", False)),
                    "reason": getattr(_non_meaningful_verdict, "reason", ""),
                    "turn_count": len(non_meaningful_recent_turns),
                    "gate_skipped": not _should_gate_non_meaningful,
                }
                if _non_meaningful_verdict is not None:
                    logger.info(
                        "Non-meaningful verdict: five_consecutive_non_meaningful=%s failed_open=%s reason=%r session_id=%s process_id=%s",
                        _non_meaningful_verdict.five_consecutive_non_meaningful,
                        _non_meaningful_verdict.failed_open,
                        _non_meaningful_verdict.reason,
                        session_id,
                        process_id,
                    )
                return _non_meaningful_verdict

            async def _cancel_non_meaningful_task_if_pending() -> None:
                if non_meaningful_task.done():
                    return
                non_meaningful_task.cancel()
                try:
                    await non_meaningful_task
                except asyncio.CancelledError:
                    pass

            async def _cancel_consent_task_if_pending() -> None:
                """Reap the consent classifier on paths that short-circuit before
                the consent gate. The outbound stage is deliberately left at
                ``intro_sent`` on those paths, so the next turn re-classifies."""
                if consent_task is None or consent_task.done():
                    return
                consent_task.cancel()
                try:
                    await consent_task
                except asyncio.CancelledError:
                    pass

            async def _moderation_decline_stream(verdict: ModerationVerdict):
                """Emit the canned decline and write history for a rejected query."""
                trace.set_route("moderation_rejected")
                if nudge_task and not nudge_task.done():
                    nudge_task.cancel()
                    trace.set_nudge(cancel_reason="moderation_rejected")
                    logger.info(
                        "Nudge canceled (moderation rejected); session_id=%s process_id=%s",
                        session_id,
                        process_id,
                    )
                    try:
                        await nudge_task
                    except asyncio.CancelledError:
                        pass
                decline_en = (
                    verdict.decline_text_en()
                    or "This helpline only handles dairy farming and animal husbandry questions."
                )
                decline_for_caller = await _render_text_for_caller(decline_en, requested_target_lang)
                decline_user_text = _canonical_history_user_text("moderation_reject")
                decl_req, decl_resp = _history_pair(decline_user_text, decline_en)
                with trace.stage("history_write"):
                    await update_message_history(session_id, [*history, decl_req, decl_resp])
                trace.set_outcome("moderation_rejected")
                yield _emit(_prepare_voice_output(decline_for_caller, requested_target_lang))

            async def _non_meaningful_hangup_stream(verdict: NonMeaningfulVerdict):
                """Emit goodbye and persist history when five non-meaningful turns are detected."""
                trace.set_route("non_meaningful_hangup")
                if nudge_task and not nudge_task.done():
                    nudge_task.cancel()
                    trace.set_nudge(cancel_reason="non_meaningful_hangup")
                    logger.info(
                        "Nudge canceled (non-meaningful hangup); session_id=%s process_id=%s",
                        session_id,
                        process_id,
                    )
                    try:
                        await nudge_task
                    except asyncio.CancelledError:
                        pass
                goodbye = TELEPHONY_TERMINATE_CALL_TOKEN.get(
                    requested_target_lang,
                    TELEPHONY_TERMINATE_CALL_TOKEN["en"],
                )
                nm_req, nm_resp = _history_pair(history_user_text or query, TELEPHONY_TERMINATE_CALL_TOKEN["en"])
                with trace.stage("history_write"):
                    await update_message_history(session_id, [*history, nm_req, nm_resp])
                trace.set_outcome("non_meaningful_hangup")
                logger.info(
                    "Non-meaningful hangup emitted; session_id=%s process_id=%s reason=%r",
                    session_id,
                    process_id,
                    verdict.reason,
                )
                # Keep the exact telephony termination token.
                yield _emit(goodbye)

            async def _outbound_decline_stream(verdict: ConsentVerdict):
                """Emit the scripted farewell and hang up after a declined outbound call."""
                trace.set_route("outbound_declined")
                if nudge_task and not nudge_task.done():
                    nudge_task.cancel()
                    trace.set_nudge(cancel_reason="outbound_declined")
                    try:
                        await nudge_task
                    except asyncio.CancelledError:
                        pass
                farewell_en = _outbound.OUTBOUND_DECLINE_FAREWELL["en"]
                farewell_for_caller = await _canned_for_caller(
                    farewell_en, requested_target_lang, _outbound.OUTBOUND_DECLINE_FAREWELL,
                )
                goodbye = TELEPHONY_TERMINATE_CALL_TOKEN.get(
                    requested_target_lang, TELEPHONY_TERMINATE_CALL_TOKEN["en"],
                )
                dec_req, dec_resp = _history_pair(
                    history_user_text or query, f"{farewell_en} {TELEPHONY_TERMINATE_CALL_TOKEN['en']}",
                )
                with trace.stage("history_write"):
                    await update_message_history(session_id, [*history, dec_req, dec_resp])
                trace.set_outcome("outbound_declined")
                logger.info(
                    "Outbound consent declined; emitting farewell + hangup - session_id=%s process_id=%s reason=%r",
                    session_id, process_id, verdict.reason,
                )
                yield _emit(_prepare_voice_output(farewell_for_caller, requested_target_lang))
                # Termination token stays exact ASCII "Goodbye." — passing it
                # through the Gujarati output filter would strip it to ".".
                yield _emit(" " + goodbye)

            # ── Empty-pretranslation guard ───────────────────────────────
            # Only short-circuit when pretranslation produced no usable text
            # at all (i.e. both primary and fallback failed). True noise still
            # routes to the agent, which is better at asking for
            # clarification in context than a canned global retry.
            if (
                requested_source_lang not in {"en", "english"}
                and not (processing_query or "").strip()
            ):
                # This short-circuits the agent, so resolve moderation here: a
                # rejected query must be declined rather than asked to repeat.
                _verdict = await _resolve_moderation()
                if _verdict is not None and _verdict.rejected:
                    if await _request_is_stale("after_moderation_reject"):
                        await _cancel_non_meaningful_task_if_pending()
                        await _cancel_consent_task_if_pending()
                        return
                    async for _c in _moderation_decline_stream(_verdict):
                        yield _c
                    await _cancel_non_meaningful_task_if_pending()
                    await _cancel_consent_task_if_pending()
                    return
                trace.set_route("pretranslation_empty")
                logger.info(
                    "Pretranslation produced no usable text; asking to repeat - session_id=%s process_id=%s query=%r",
                    session_id, process_id, query,
                )
                low_conf_resp_for_history = _FRAGMENT_RESPONSES["en"]
                low_conf_resp_for_caller = await _canned_for_caller(low_conf_resp_for_history, requested_target_lang, _FRAGMENT_RESPONSES)
                low_conf_req, low_conf_rsp = _history_pair(
                    history_user_text or _canonical_history_user_text("low_confidence"),
                    low_conf_resp_for_history,
                )
                with trace.stage("history_write"):
                    await update_message_history(session_id, [*history, low_conf_req, low_conf_rsp])
                trace.set_outcome("pretranslation_empty")
                yield _emit(_prepare_voice_output(low_conf_resp_for_caller, requested_target_lang))
                await _cancel_non_meaningful_task_if_pending()
                await _cancel_consent_task_if_pending()
                return

            if farmer_cache_task is not None:
                try:
                    with trace.stage("farmer_context"):
                        envelope = await farmer_cache_task
                    if envelope is None:
                        # A concurrent fetch (e.g. the outbound prefetch) may have
                        # landed while ours was giving up. Cheap Redis re-read
                        # before we commit to an unresolved turn.
                        envelope = await get_farmer_data_cached_only(mobile)
                        if envelope is not None:
                            logger.info("Farmer context resolved on re-read for mobile %s", mobile)
                    # Scored AFTER the re-read: a recovered envelope must not be
                    # graded UNRESOLVED and have the identity tools withheld.
                    farmer_identity = identity_state_for_envelope(envelope)
                    farmer_info = _build_compact_farmer_summary(envelope)
                    farmer_unions = _collect_farmer_unions(envelope)
                    farmer_accounts = _collect_farmer_accounts(envelope)
                    with trace.stage("scheme_summary"):
                        scheme_summary = await _build_union_scheme_summary(farmer_unions)
                    if scheme_summary:
                        farmer_info = f"{farmer_info}\n{scheme_summary}" if farmer_info else scheme_summary
                    ai_technician_info = _build_ai_technician_summary(envelope)
                    trace.set_farmer_context(
                        source=getattr(envelope, "source", None) if envelope else None,
                        stale=getattr(envelope, "stale", None) if envelope else None,
                        unions=farmer_unions,
                        farmer_info_chars=len(farmer_info),
                        technician_info_chars=len(ai_technician_info),
                    )
                    logger.info(
                        "Farmer summary loaded from cache for mobile %s source=%s stale=%s unions=%s summary_chars=%s technician_chars=%s",
                        mobile,
                        getattr(envelope, "source", None) if envelope else None,
                        getattr(envelope, "stale", None) if envelope else None,
                        farmer_unions,
                        len(farmer_info),
                        len(ai_technician_info),
                    )
                    if mobile and should_refresh_farmer_data(envelope):
                        await enqueue_farmer_refresh(mobile)
                        logger.info(
                            "Farmer cache refresh scheduled in background for mobile %s stale=%s status=%s",
                            mobile,
                            getattr(envelope, "stale", None) if envelope else None,
                            getattr(envelope, "lookupStatus", None) if envelope else None,
                        )
                except Exception as e:
                    logger.warning(f"Failed to load farmer summary for mobile {mobile}: {e}")

            # ── Outbound consent gate ─────────────────────────────────────
            # Turn 2 of an outbound call: the farmer's first reply to the scripted
            # consent question. Three-way — a reply that is neither yes nor no
            # (typically the farmer asking their own question) simply falls through
            # to a normal agent turn, which is the outcome we most want to protect.
            outbound_milk_hint: Optional[str] = None
            if consent_task is not None:
                _consent_wait_started = time.monotonic()
                consent_verdict = await consent_task
                trace.attach_stage_timing(
                    "outbound_consent",
                    (time.monotonic() - _consent_wait_started) * 1000.0,
                    intent=consent_verdict.intent,
                    failed_open=consent_verdict.failed_open,
                )
                try:
                    trace.metadata["outbound_consent_intent"] = consent_verdict.intent
                except Exception:  # pragma: no cover - tracing must never break the call
                    pass
                logger.info(
                    "Outbound consent verdict - session_id=%s process_id=%s intent=%s reason=%r failed_open=%s",
                    session_id, process_id, consent_verdict.intent,
                    consent_verdict.reason, consent_verdict.failed_open,
                )
                # The opener is done either way: never re-classify on later turns.
                await _outbound.set_stage(session_id, _outbound.STAGE_RESOLVED)

                if consent_verdict.is_negative:
                    await _cancel_non_meaningful_task_if_pending()
                    if not moderation_task.done():
                        moderation_task.cancel()
                    if not await _request_is_stale("after_outbound_decline"):
                        async for _c in _outbound_decline_stream(consent_verdict):
                            yield _c
                    return

                if consent_verdict.is_affirmative:
                    prefetched = await _outbound.get_prefetched_milk_summary(session_id)
                    if prefetched:
                        outbound_milk_hint = _outbound.milk_answer_hint(prefetched)
                        trace.set_route("outbound_milk_readout")
                    elif signed_in and mobile and farmer_accounts:
                        # Prefetch missed (cold cache or slow upstream) — the agent
                        # fetches it itself against the same pinned window.
                        _from, _to = _outbound.milk_window(settings.outbound_milk_window_days)
                        outbound_milk_hint = _outbound.milk_fetch_hint(_from, _to)
                        trace.set_route("outbound_milk_readout_cold")
                    else:
                        # Consented, but there is nothing to read out (no account on
                        # this number). Say so and leave the call open rather than
                        # reading an upstream failure message aloud.
                        await _cancel_non_meaningful_task_if_pending()
                        if not moderation_task.done():
                            moderation_task.cancel()
                        trace.set_route("outbound_no_data")
                        no_data_en = _outbound.OUTBOUND_NO_DATA["en"]
                        no_data_for_caller = await _canned_for_caller(
                            no_data_en, requested_target_lang, _outbound.OUTBOUND_NO_DATA,
                        )
                        nd_req, nd_resp = _history_pair(history_user_text or query, no_data_en)
                        with trace.stage("history_write"):
                            await update_message_history(session_id, [*history, nd_req, nd_resp])
                        trace.set_outcome("outbound_no_data")
                        logger.info(
                            "Outbound consent affirmative but no milk data available - "
                            "session_id=%s process_id=%s signed_in=%s accounts=%s",
                            session_id, process_id, signed_in, len(farmer_accounts),
                        )
                        yield _emit(_prepare_voice_output(no_data_for_caller, requested_target_lang))
                        return

            logger.info(f"User info: {user_info}")
            deps = FarmerContext(
                query=processing_query,
                lang_code=processing_lang,
                target_lang=requested_target_lang,
                provider=provider,
                session_id=session_id,
                process_id=process_id,
                farmer_info=farmer_info,
                farmer_unions=farmer_unions,
                ai_technician_info=ai_technician_info,
                signed_in=signed_in,
                mobile=mobile,
                farmer_identity=farmer_identity,
                farmer_accounts=farmer_accounts,
            )
            # Let side-effecting tools (bookings) self-gate on the concurrent
            # moderation verdict before performing any write.
            deps.set_moderation_task(moderation_task)

            message_pairs = "\n\n".join(format_message_pairs(history, 3))
            logger.info(f"Message pairs: {message_pairs}")
            user_message = deps.get_user_message()
            runtime_context_request = _build_runtime_context_request(deps)
            logger.info(f"Running agent with user message: {user_message}")

            cleaned_history = clean_message_history_for_openai(history)
            if len(cleaned_history) != len(history):
                logger.warning(f"Cleaned {len(history) - len(cleaned_history)} orphaned tool calls from history")
                if not await _request_is_stale("before_cleaned_history_write"):
                    await update_message_history(session_id, cleaned_history)
                history = cleaned_history

            trimmed_history = trim_history(
                history,
                max_tokens=32_000,
                include_system_prompts=False,
                include_tool_calls=True,
                # Older turns keep their replies but lose their retrieved
                # documents, so this turn cannot be answered out of a previous
                # turn's search results (issue #271).
                tool_return_turns=(
                    settings.history_tool_return_turns
                    if settings.history_tool_return_turns >= 0
                    else None
                ),
            )
            logger.info(f"Trimmed history length: {len(trimmed_history)} messages")
            # pydantic-ai's Agent(instructions=STATIC_VOICE_SYSTEM_PROMPT) already
            # emits the system prompt on every run. Prepending another
            # SystemPromptPart here produced two identical role=system messages
            # (~33 KB each) per turn, which both inflates context and dilutes
            # attention to the actual runtime context. Keep only the runtime
            # context request, which carries the per-turn deps (today's date,
            # farmer profile, ambiguity hints, voice answer mode).
            # Stable context first → [system][stable-context][history] is a single
            # growing prefix vLLM can cache across turns. Per-query hints (if any)
            # go last, right before the user message, so they never break it.
            model_input_history = [runtime_context_request, *trimmed_history]
            query_hints_request = _build_query_hints_request(deps)
            if query_hints_request is not None:
                model_input_history.append(query_hints_request)
            if outbound_milk_hint is not None:
                # Appended after the per-query hints, immediately before the user
                # message: this turn's instruction is the readout, and it must sit
                # closest to the reply it acts on.
                model_input_history.append(
                    ModelRequest(parts=[UserPromptPart(content=(
                        "Hints for the current user query:\n" + outbound_milk_hint
                    ))])
                )
            active_agent = voice_agent_signed_in if (signed_in and mobile) else voice_agent
            usage_limits = UsageLimits(request_limit=6 if (signed_in and mobile) else 4)

            if settings.retrieval_audit_log:
                logger.info(
                    "RETRIEVAL_AUDIT query=%r session_id=%s process_id=%s target_lang=%s",
                    processing_query,
                    session_id,
                    process_id,
                    requested_target_lang,
                )

            with boundary_capture_context(
                session_id=session_id,
                process_id=process_id,
                user_query=processing_query,
            ):
                # Restored token streaming on pydantic-ai 1.x. run_stream now drives
                # the full tool-call loop past a tool-call-only first response (the
                # 0.2.4 stall that previously forced a blocking run()), then streams
                # the final English text. We pipe those en deltas straight into the
                # en->gu batch translator below, so agent generation and output
                # translation overlap instead of running strictly back-to-back.
                agent_started_at = time.monotonic()
                _agent_output = ""
                first_text_chunk_received = False
                sentence_buffer = ""
                translation_batch: list[str] = []
                batch_word_count = 0
                _agent_requested_tier = "oss" if is_oss else "managed"
                _agent_actual_tier = _agent_requested_tier
                _agent_actual_provider = request_provider
                _agent_actual_model = request_model_name
                _agent_committed_tier: Optional[str] = None
                _agent_committed_provider: Optional[str] = None
                _agent_committed_model: Optional[str] = None
                _agent_fallback_used = False
                _agent_attempts: list[dict[str, object]] = []

                async def _yield_translated_text(text_to_translate: str) -> AsyncGenerator[str, None]:
                    if not text_to_translate:
                        return
                    text_to_translate = _guard_identity_drift(text_to_translate)
                    canned_ban = _canned_union_ban_translation(
                        text_to_translate, requested_target_lang,
                    )
                    if canned_ban is not None:
                        yield _prepare_voice_output(canned_ban, requested_target_lang)
                        return
                    try:
                        with trace.stage(
                            "output_translation",
                            as_type="generation",
                            input={"chars": len(text_to_translate)},
                            metadata={"target_lang": requested_target_lang},
                        ):
                            async for chunk in translate_text_stream_fast(
                                text=text_to_translate,
                                source_lang="english",
                                target_lang=requested_target_lang,
                            ):
                                if await _request_is_stale("during_output_translation"):
                                    return
                                cleaned = (
                                    _prepare_voice_output(chunk, requested_target_lang)
                                    if isinstance(chunk, str) and chunk
                                    else chunk
                                )
                                if isinstance(cleaned, str) and cleaned.strip():
                                    trace.mark("first_translation_chunk_ms")
                                yield cleaned
                    except Exception as e:
                        trace.increment("output_translation_errors")
                        logger.error(
                            "Translation pipeline output translation failed for session_id=%s error=%s",
                            session_id,
                            e,
                        )
                        trouble = TRANSLATION_TROUBLE_MESSAGE.get(
                            requested_target_lang,
                            TRANSLATION_TROUBLE_MESSAGE["en"],
                        )
                        yield trouble


                # Token consumer (nudge / staleness / batch-translation) — shared by both
                # the fallback and legacy source paths so the proven loop isn't duplicated.
                async def _consume_agent_text(stream_iter):
                    nonlocal _agent_output, first_text_chunk_received, sentence_buffer, translation_batch, batch_word_count
                    try:
                        async for chunk in stream_iter:
                            if await _request_is_stale("during_agent_stream"):
                                break

                            if isinstance(chunk, str) and chunk:
                                if not _agent_output and chunk.strip():
                                    trace.mark("first_agent_text_ms")
                                _agent_output += chunk

                            if not needs_output_translation:
                                if (
                                    not first_text_chunk_received
                                    and isinstance(chunk, str)
                                    and chunk
                                    and chunk.strip()
                                ):
                                    first_text_chunk_received = True
                                    if nudge_task:
                                        nudge_task.cancel()
                                        logger.info(
                                            "Nudge canceled (first text chunk received); session_id=%s process_id=%s chunk_preview=%s",
                                            session_id,
                                            process_id,
                                            chunk[:50] if len(chunk) > 50 else chunk,
                                        )
                                        try:
                                            await nudge_task
                                        except asyncio.CancelledError:
                                            pass
                                    trace.set_nudge(cancel_reason="first_text_chunk_received")

                                cleaned_chunk = (
                                    _prepare_voice_output(chunk, requested_target_lang)
                                    if isinstance(chunk, str) and chunk
                                    else chunk
                                )
                                if await _request_is_stale("before_direct_yield"):
                                    break
                                yield _emit(cleaned_chunk)
                                continue

                            sentence_buffer += chunk
                            ready_units, remaining = extract_translation_units(sentence_buffer)
                            if ready_units:
                                for unit in ready_units:
                                    candidate_units = [unit]
                                    if len(unit) >= VOICE_TRANSLATION_BATCH_CHAR_LIMIT:
                                        candidate_units = []
                                        remaining_unit = unit
                                        while remaining_unit:
                                            head, tail = _split_voice_batch_text(remaining_unit)
                                            if not tail or head == remaining_unit:
                                                candidate_units.append(remaining_unit)
                                                break
                                            candidate_units.append(head)
                                            remaining_unit = tail

                                    for candidate in candidate_units:
                                        translation_batch.append(candidate)
                                        batch_word_count += len(candidate.split())
                                        batch_text = "".join(translation_batch)

                                        if should_translate_batch(batch_text, batch_word_count, is_first_batch=not first_text_chunk_received):
                                            async for translated_chunk in _yield_translated_text(batch_text):
                                                if (
                                                    not first_text_chunk_received
                                                    and isinstance(translated_chunk, str)
                                                    and translated_chunk
                                                    and translated_chunk.strip()
                                                ):
                                                    first_text_chunk_received = True
                                                    if nudge_task:
                                                        nudge_task.cancel()
                                                        logger.info(
                                                            "Nudge canceled (first translated chunk received); session_id=%s process_id=%s",
                                                            session_id,
                                                            process_id,
                                                        )
                                                        try:
                                                            await nudge_task
                                                        except asyncio.CancelledError:
                                                            pass
                                                    trace.set_nudge(cancel_reason="first_translated_chunk_received")
                                                if await _request_is_stale("before_translated_yield"):
                                                    break
                                                yield _emit(_prepare_translated_emit(translated_chunk))
                                            translation_batch = []
                                            batch_word_count = 0

                                sentence_buffer = remaining

                        if needs_output_translation and not await _request_is_stale("before_translation_flush"):
                            if translation_batch:
                                batch_text = "".join(translation_batch)
                                async for translated_chunk in _yield_translated_text(batch_text):
                                    if (
                                        not first_text_chunk_received
                                        and isinstance(translated_chunk, str)
                                        and translated_chunk
                                        and translated_chunk.strip()
                                    ):
                                        first_text_chunk_received = True
                                        if nudge_task: nudge_task.cancel()
                                        logger.info(
                                            "Nudge canceled (final translated batch); session_id=%s process_id=%s",
                                            session_id,
                                            process_id,
                                        )
                                        try:
                                            if nudge_task:
                                                await nudge_task
                                        except asyncio.CancelledError:
                                            pass
                                        trace.set_nudge(cancel_reason="final_translated_batch")
                                    if await _request_is_stale("before_final_translated_yield"):
                                        break
                                    yield _emit(_prepare_translated_emit(translated_chunk))

                            if sentence_buffer.strip():
                                async for translated_chunk in _yield_translated_text(sentence_buffer):
                                    if (
                                        not first_text_chunk_received
                                        and isinstance(translated_chunk, str)
                                        and translated_chunk
                                        and translated_chunk.strip()
                                    ):
                                        first_text_chunk_received = True
                                        if nudge_task: nudge_task.cancel()
                                        logger.info(
                                            "Nudge canceled (tail translated fragment); session_id=%s process_id=%s",
                                            session_id,
                                            process_id,
                                        )
                                        try:
                                            if nudge_task:
                                                await nudge_task
                                        except asyncio.CancelledError:
                                            pass
                                        trace.set_nudge(cancel_reason="tail_translated_fragment")
                                    if await _request_is_stale("before_tail_translated_yield"):
                                        break
                                    yield _emit(_prepare_translated_emit(translated_chunk))
                    except StopAsyncIteration:
                        pass
                    except RuntimeError as e:
                        if "StopAsyncIteration" in str(e) or "anext()" in str(e):
                            # anext() errors occur on superseded processes during
                            # teardown — the final process_id has its own generator
                            # and is unaffected, so this is just cleanup noise.
                            logger.debug(
                                "Suppressed stream runtime error (superseded process teardown) - session_id=%s process_id=%s error=%s",
                                session_id,
                                process_id,
                                e,
                            )
                        else:
                            raise
                    finally:
                        if nudge_task and not nudge_task.done():
                            if nudge_task: nudge_task.cancel()
                            trace.set_nudge(cancel_reason="stream_ended")
                            logger.info(
                                "Nudge canceled (stream ended); session_id=%s process_id=%s",
                                session_id,
                                process_id,
                            )
                            try:
                                await nudge_task
                            except asyncio.CancelledError:
                                pass


                if settings.fallback_enabled:
                    # OSS -> managed first-token commit. A pre-first-token OSS failure
                    # silently swaps to managed; a post-first-token failure can't swap
                    # (caller already heard audio) -> canned line. The agent stream is
                    # driven from a single task (see below) for anyio task-affinity.
                    _fb_holder: dict = {}

                    async def _raw_stream(attempt, _attempt_info):
                        # (D) COMMIT-ON-FIRST-ACTIVITY. Iterate via agent.iter()+node.stream()
                        # so the FIRST pydantic-ai model event (a tool-call part, emitted
                        # BEFORE the tools run and long before the first TEXT delta) is
                        # surfaced once as the AGENT_ACTIVITY sentinel.
                        # with_first_token_deadline treats that as the first-token commit,
                        # so the slow 20s milk-collection tool can no longer trip the TTFT
                        # deadline and force a cross-tier re-run of side-effecting tools
                        # (CreateAICall booking / SMS -> duplicate bookings). The sentinel
                        # is swallowed by the deadline wrapper and never heard by the
                        # caller. Liveness is preserved: a hung endpoint emits no event, so
                        # the deadline still fires -> swap. _attempt_had_chunk / committed
                        # stay tied to real TEXT (unchanged caller-visible commit).
                        _activity_signaled = False
                        async with active_agent.iter(
                            user_prompt=user_message,
                            message_history=model_input_history,
                            deps=deps,
                            usage_limits=usage_limits,
                            model=attempt.model,
                        ) as agent_run:
                            _attempt_had_chunk = False
                            async for node in agent_run:
                                if type(node).__name__ == 'ModelRequestNode':
                                    async with node.stream(agent_run.ctx) as request_stream:
                                        async for event in request_stream:
                                            if not _activity_signaled:
                                                _activity_signaled = True
                                                yield AGENT_ACTIVITY
                                            event_type = type(event).__name__
                                            _c = None
                                            if event_type == 'PartStartEvent' and hasattr(event, 'part'):
                                                if type(event.part).__name__ == 'TextPart' and hasattr(event.part, 'content'):
                                                    _c = event.part.content
                                            elif event_type == 'PartDeltaEvent' and hasattr(event, 'delta'):
                                                if type(event.delta).__name__ == 'TextPartDelta':
                                                    _c = event.delta.content_delta
                                            if _c:
                                                if not _attempt_had_chunk:
                                                    _attempt_had_chunk = True
                                                    _attempt_info["committed"] = True
                                                    _attempt_info["status"] = "committed"
                                                yield _c
                            if _attempt_had_chunk and _attempt_info.get("status") != "error":
                                _attempt_info["status"] = "ok"
                            elif _attempt_info.get("status") != "error":
                                _attempt_info["status"] = "ok_no_output"
                            _fb_holder["new_messages"] = agent_run.result.new_messages()

                    async def _make_stream(attempt):
                        nonlocal _agent_actual_tier, _agent_actual_provider, _agent_actual_model
                        nonlocal _agent_committed_tier, _agent_committed_provider, _agent_committed_model
                        # Bound time-to-first-token (attempt.timeout) so a silent OSS
                        # hang swaps to managed before the caller hears anything; the
                        # deadline disarms after the first token, so a long mid-stream
                        # gap (tool round-trip) keeps the model's 600s read-timeout.
                        _attempt_info: dict[str, object] = {
                            "tier": attempt.kind,
                            "provider": attempt.provider,
                            "model": attempt.model_name,
                            "endpoint": attempt.endpoint,
                            "status": "started",
                        }
                        _agent_attempts.append(_attempt_info)
                        try:
                            async for _c in with_first_token_deadline(attempt, _raw_stream(attempt, _attempt_info)):
                                if _agent_committed_tier is None:
                                    _agent_committed_tier = attempt.kind
                                    _agent_committed_provider = attempt.provider
                                    _agent_committed_model = attempt.model_name
                                    _agent_actual_tier = attempt.kind
                                    _agent_actual_provider = attempt.provider
                                    _agent_actual_model = attempt.model_name
                                yield _c
                        except Exception as _attempt_exc:
                            _attempt_info["status"] = "error"
                            _attempt_info["error_class"] = type(_attempt_exc).__name__
                            _attempt_info["error_reason"] = classify(_attempt_exc).value
                            raise

                    _src = stream_with_fallback(
                        pipeline="chat",
                        session_id=session_id,
                        profile_name=pipeline_profile,
                        make_stream=_make_stream,
                    ).__aiter__()

                    # Pull the first token in THIS task — pydantic-ai's anyio stream
                    # must be advanced from a single task (a separate create_task would
                    # cross task boundaries and break its cancel scope). Parallelism is
                    # preserved because the moderation check is already running on its
                    # own background task (moderation_task), overlapping the agent.
                    _NO_FIRST = object()
                    _first_chunk = _NO_FIRST
                    _stream_error = None
                    try:
                        _first_chunk = await _src.__anext__()
                    except StopAsyncIteration:
                        _first_chunk = _NO_FIRST  # empty stream (no tokens), not an error
                    except Exception as _e:  # pre-first-token: OSS + managed both failed
                        _stream_error = _e

                    # Moderation gate — resolve before emitting anything to the caller.
                    _verdict = await _resolve_moderation()
                    if _verdict is not None and _verdict.rejected:
                        try:
                            await _src.aclose()
                        except Exception:
                            pass
                        if not await _request_is_stale("after_moderation_reject"):
                            async for _c in _moderation_decline_stream(_verdict):
                                yield _c
                        await _cancel_non_meaningful_task_if_pending()
                        return

                    # Non-meaningful gate — hang up after five consecutive
                    # unclear/gibberish turns. Resolving here also reaps the
                    # background classifier task on the normal agent path.
                    _non_meaningful = await _resolve_non_meaningful()
                    if (
                        _non_meaningful is not None
                        and _non_meaningful.five_consecutive_non_meaningful
                    ):
                        try:
                            await _src.aclose()
                        except Exception:
                            pass
                        if not await _request_is_stale("after_non_meaningful_hangup"):
                            async for _c in _non_meaningful_hangup_stream(_non_meaningful):
                                yield _c
                        return

                    if _stream_error is not None:
                        _agent_actual_tier = "failed"
                        _agent_actual_provider = "failed"
                        _agent_actual_model = "failed"
                        logger.error(
                            "Voice agent stream failed before first token; session_id=%s process_id=%s error=%s",
                            session_id, process_id, _stream_error,
                        )
                        if not await _request_is_stale("after_stream_error"):
                            _trouble = TRANSLATION_TROUBLE_MESSAGE.get(
                                requested_target_lang, TRANSLATION_TROUBLE_MESSAGE["en"],
                            )
                            yield _emit(_trouble)
                        new_messages = _fb_holder.get("new_messages", [])
                    else:
                        async def _committed_stream():
                            if _first_chunk is not _NO_FIRST:
                                yield _first_chunk
                            async for _c in _src:
                                yield _c

                        try:
                            async for _out in _consume_agent_text(_committed_stream()):
                                yield _out
                        except Exception as _post_err:
                            logger.error(
                                "Voice agent stream failed after first token; session_id=%s process_id=%s error=%s",
                                session_id, process_id, _post_err,
                            )
                            if not await _request_is_stale("after_stream_error"):
                                _trouble = TRANSLATION_TROUBLE_MESSAGE.get(
                                    requested_target_lang, TRANSLATION_TROUBLE_MESSAGE["en"],
                                )
                                yield _emit(_trouble)
                        _agent_output = _agent_output.strip()
                        new_messages = _fb_holder.get("new_messages", [])
                else:
                    async with active_agent.run_stream(
                        user_prompt=user_message,
                        message_history=model_input_history,
                        deps=deps,
                        usage_limits=usage_limits,
                        model=request_model,
                    ) as response_stream:
                        stream_iter = response_stream.stream_text(delta=True, debounce_by=0)
                        _verdict = await _resolve_moderation()
                        if _verdict is not None and _verdict.rejected:
                            if not await _request_is_stale("after_moderation_reject"):
                                async for _c in _moderation_decline_stream(_verdict):
                                    yield _c
                            await _cancel_non_meaningful_task_if_pending()
                            return
                        # Non-meaningful gate — hang up after five consecutive
                        # unclear/gibberish turns. Resolving here also reaps the
                        # background classifier task on the normal agent path.
                        _non_meaningful = await _resolve_non_meaningful()
                        if (
                            _non_meaningful is not None
                            and _non_meaningful.five_consecutive_non_meaningful
                        ):
                            if not await _request_is_stale("after_non_meaningful_hangup"):
                                async for _c in _non_meaningful_hangup_stream(_non_meaningful):
                                    yield _c
                            return
                        _attempt_info: dict[str, object] = {
                            "tier": _agent_requested_tier,
                            "provider": request_provider,
                            "model": request_model_name,
                            "status": "started",
                        }
                        _agent_attempts.append(_attempt_info)

                        async def _legacy_stream_with_attempt():
                            nonlocal _agent_committed_tier, _agent_committed_provider, _agent_committed_model
                            _attempt_had_chunk = False
                            try:
                                async for _chunk in stream_iter:
                                    if not _attempt_had_chunk:
                                        _attempt_had_chunk = True
                                        _attempt_info["committed"] = True
                                        _attempt_info["status"] = "committed"
                                        _agent_committed_tier = _agent_requested_tier
                                        _agent_committed_provider = request_provider
                                        _agent_committed_model = request_model_name
                                    yield _chunk
                            except Exception as _attempt_exc:
                                _attempt_info["status"] = "error"
                                _attempt_info["error_class"] = type(_attempt_exc).__name__
                                _attempt_info["error_reason"] = classify(_attempt_exc).value
                                raise
                            if _attempt_had_chunk and _attempt_info.get("status") != "error":
                                _attempt_info["status"] = "ok"
                            elif _attempt_info.get("status") != "error":
                                _attempt_info["status"] = "ok_no_output"

                        async for _out in _consume_agent_text(_legacy_stream_with_attempt()):
                            yield _out
                        _agent_output = _agent_output.strip()
                        new_messages = response_stream.new_messages()

                _agent_fallback_used = (
                    len(_agent_attempts) > 1
                    and _agent_attempts[0].get("status") == "error"
                )

                trace.attach_stage_timing(
                    "agent",
                    (time.monotonic() - agent_started_at) * 1000.0,
                    signed_in=bool(signed_in and mobile),
                    request_limit=usage_limits.request_limit,
                    pipeline_profile=pipeline_profile,
                    requested_tier=_agent_requested_tier,
                    requested_model=request_model_name,
                    requested_provider=request_provider,
                    actual_tier=_agent_actual_tier,
                    actual_model=_agent_actual_model,
                    actual_provider=_agent_actual_provider,
                    fallback_used=_agent_fallback_used,
                    first_token_committed_tier=_agent_committed_tier,
                    first_token_committed_provider=_agent_committed_provider,
                    first_token_committed_model=_agent_committed_model,
                    attempts=_agent_attempts,
                )
                trace.set_agent(
                    signed_in=bool(signed_in and mobile),
                    output=_agent_output,
                    new_messages=new_messages,
                    requested_tier=_agent_requested_tier,
                    requested_provider=request_provider,
                    requested_model=request_model_name,
                    actual_tier=_agent_actual_tier,
                    actual_provider=_agent_actual_provider,
                    actual_model=_agent_actual_model,
                    first_token_committed_tier=_agent_committed_tier,
                    first_token_committed_provider=_agent_committed_provider,
                    first_token_committed_model=_agent_committed_model,
                    fallback_used=_agent_fallback_used,
                    attempts=_agent_attempts,
                )

            # If the LLM called signal_conversation_state("conversation_closing"),
            # append the termination token so RAYA disconnects the call.
            # We scan the agent's new messages for the tool call rather than
            # using contextvars, because pydantic-ai runs tools in child tasks
            # whose contextvar writes don't propagate back to the caller.
            closing = any(
                getattr(part, "tool_name", None) == "signal_conversation_state"
                and "conversation_closing" in (getattr(part, "args_as_json_str", lambda: "")() if callable(getattr(part, "args_as_json_str", None)) else str(getattr(part, "args", "")))
                for msg in new_messages
                for part in (getattr(msg, "parts", None) or [])
            )
            if closing and not await _request_is_stale("before_goodbye"):
                goodbye = TELEPHONY_TERMINATE_CALL_TOKEN.get(
                    requested_target_lang,
                    TELEPHONY_TERMINATE_CALL_TOKEN["en"],
                )
                logger.info(
                    "Appending goodbye after conversation_closing signal; session_id=%s process_id=%s",
                    session_id, process_id,
                )
                yield _emit(" " + goodbye)

            if await _request_is_stale("before_history_write"):
                return

            messages = [*history, *new_messages]
            logger.info(f"Updating message history for session {session_id} with {len(messages)} messages")
            with trace.stage("history_write"):
                await update_message_history(session_id, messages)
            if trace.outcome is None:
                trace.set_outcome("success")
    except Exception as exc:
        trace.finish(trace.outcome or "error", error=exc)
        raise
    finally:
        release_started_at = time.monotonic()
        released = await release_session_request_ownership(owner)
        trace.attach_stage_timing(
            "ownership_release",
            (time.monotonic() - release_started_at) * 1000.0,
            released=released,
        )
        if owner is not None:
            logger.info(
                "Session ownership released - session_id=%s process_id=%s epoch=%s released=%s",
                session_id,
                process_id,
                owner.epoch,
                released,
            )
        trace.finish(trace.outcome or "success")
