"""
Marqo client implementation for vector search.
The Marqo Python client is synchronous; we run it in asyncio.to_thread() to avoid
blocking the event loop when serving many concurrent requests.
"""
import asyncio
import os
import re
from typing import Any, Dict, FrozenSet, List, Literal, Optional

import marqo
from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry, RunContext

from agents.deps import FarmerContext
from agents.services.species import (
    hit_species,
    off_species_penalty,
    partition_hits_by_species,
    resolve_caller_species,
)
from agents.tools.terms import normalize_text_with_glossary
from app.observability import start_observation
from helpers.utils import get_logger

logger = get_logger(__name__)
_index_capabilities_cache: Dict[str, Dict[str, Any]] = {}
_TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)
_GUJARATI_CHAR_RE = re.compile(r"[\u0A80-\u0AFF]")
_REFUSAL_OR_META_PATTERNS = [
    "i can only answer",
    "your query appears to be",
    "would you like to ask about",
    "not within the agricultural scope",
    "out of scope",
    "not related to",
    "i cannot help",
    "i'm unable to",
    "as an ai",
    "search results for",
    "based on the provided documents",
]
_WRONG_INTENT_HINTS = [
    "hf receipts",
    "tracking numbers",
    "track number",
]

# Second-chance vocabulary, used only when the first search came back with
# nothing to show. Callers say "not coming into heat"; the corpus writes
# "anestrus". The first query is left untouched so a search that already works
# keeps working — this only runs when the alternative is telling the farmer we
# have no information on a core topic of this helpline (issue #271).
_RETRY_EXPANSIONS: List[tuple] = [
    (
        re.compile(r"\b(heat|estrus|oestrus|anestrus|anoestrus|bulling|come?s? into heat)\b", re.I),
        "anestrus estrus induction silent heat cycle",
    ),
    (
        re.compile(r"\b(repeat breed\w*|not conceiv\w*|conception failure|repeat breeder)\b", re.I),
        "repeat breeder conception rate infertility",
    ),
    (
        re.compile(r"\b(heat detection|detect\w* heat|signs of heat|standing heat)\b", re.I),
        "estrus detection signs standing heat timing insemination",
    ),
    (
        re.compile(r"\b(shed|cowshed|housing|gaushala|byre)\b", re.I),
        "cattle shed construction subsidy animal husbandry scheme assistance",
    ),
    (
        re.compile(r"\b(subsidy|subsidies|sahay|scheme)\b", re.I),
        "government assistance eligibility application animal husbandry department",
    ),
]

# What the tool says when it has nothing on-topic. The wording is the contract:
# the agent used to receive a bare "No results found" and fill the silence with
# whatever it had retrieved a turn earlier, which is how a cattle-shed subsidy
# question came back as diarrhoea advice (issue #271). Naming the gap and the
# next step leaves it nothing to substitute.
_RETRIEVAL_GAP = (
    "No results found for `{query}`.\n\n"
    "RETRIEVAL_GAP{reason}. Do not answer this from a neighbouring topic, from "
    "memory, or from documents retrieved earlier in this call. Tell the caller "
    "plainly that you do not have this information, then offer one concrete next "
    "step: booking a veterinary health call, or their dairy society or nearest "
    "government veterinary dispensary."
)
_GAP_REASON_EMPTY = ": the indexed documents have nothing on this topic"
_GAP_REASON_SPECIES = (
    ": every document that matched was about a different animal than this "
    "caller's, so none of it applies"
)


def _validate_search_query(query: str) -> str:
    normalized = re.sub(r"\s+", " ", (query or "").strip())

    if not normalized:
        logger.warning("Search query validation failed: empty query")
        raise ModelRetry("INVALID_QUERY: EMPTY_QUERY. Provide a focused agricultural search query.")

    lowered = normalized.lower()
    if any(p in lowered for p in _REFUSAL_OR_META_PATTERNS):
        logger.warning("Search query validation failed: refusal/meta leakage query=%s", normalized)
        raise ModelRetry(
            "INVALID_QUERY: REFUSAL_TEXT_LEAK. "
            "Provide only concise domain keywords, never policy/refusal/meta text."
        )

    if any(p in lowered for p in _WRONG_INTENT_HINTS):
        logger.warning("Search query validation failed: known wrong-intent leakage query=%s", normalized)
        raise ModelRetry(
            "INVALID_QUERY: OFF_TOPIC_QUERY. "
            "Regenerate query aligned to user intent and agricultural topic."
        )

    token_count = len(_TOKEN_RE.findall(lowered))
    if token_count > 20:
        logger.warning("Search query validation failed: too long token_count=%s query=%s", token_count, normalized)
        raise ModelRetry(
            "INVALID_QUERY: QUERY_TOO_LONG. "
            "Use 2-12 concise keywords capturing entity/problem/task."
        )

    sentence_markers = ("?", ".", "!", " because ", " please ", " should ", " would ")
    if token_count >= 12 and any(marker in lowered for marker in sentence_markers):
        logger.warning("Search query validation failed: narrative query=%s", normalized)
        raise ModelRetry(
            "INVALID_QUERY: NARRATIVE_QUERY. "
            "Use compact keyword query, not a sentence or explanation."
        )

    logger.info("Search query validation passed: query=%s", normalized)
    return normalized


def _marqo_search_sync(endpoint_url: str, index_name: str, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    client = marqo.Client(url=endpoint_url)
    result = client.index(index_name).search(**search_params)
    return result.get("hits", [])


def _get_index_capabilities_sync(endpoint_url: str, index_name: str) -> Dict[str, Any]:
    cache_key = f"{endpoint_url}::{index_name}"
    cached = _index_capabilities_cache.get(cache_key)
    if cached is not None:
        return cached

    client = marqo.Client(url=endpoint_url)
    try:
        index_info = client.get_index(index_name)
        tensor_fields = set(index_info.get("tensorFields", []) if isinstance(index_info, dict) else [])
        all_fields = index_info.get("allFields", []) if isinstance(index_info, dict) else []
        field_names = {f.get("name") for f in all_fields if isinstance(f, dict) and f.get("name")}
        capabilities = {
            "exists": True,
            "tensor_fields": sorted(tensor_fields),
            "has_text_tensor": "text" in tensor_fields,
            "has_text_for_embedding_tensor": "text_for_embedding" in tensor_fields,
            "has_is_reference_filter": "is_reference" in field_names,
            "field_names": sorted(field_names),
        }
    except Exception as e:
        capabilities = {
            "exists": False,
            "error": str(e),
            "tensor_fields": [],
            "has_text_tensor": False,
            "has_text_for_embedding_tensor": False,
            "has_is_reference_filter": False,
            "field_names": [],
        }

    _index_capabilities_cache[cache_key] = capabilities
    return capabilities


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _prepare_query_for_e5(query: str) -> str:
    cleaned = query.strip()
    if cleaned.lower().startswith("query:"):
        return cleaned
    return f"query: {cleaned}"


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid int for %s=%r; using default=%s", name, raw, default)
        return default


def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid float for %s=%r; using default=%s", name, raw, default)
        return default


def _resolve_final_top_k(requested_top_k: int) -> int:
    default_final = max(1, _parse_int_env("MARQO_DEFAULT_FINAL_CHUNKS", 12))
    env_cap = max(1, _parse_int_env("MARQO_MAX_FINAL_CHUNKS", 20))
    hard_cap = min(env_cap, 20)

    try:
        requested = int(requested_top_k)
    except (TypeError, ValueError):
        requested = default_final

    if requested <= 0:
        requested = default_final

    return max(1, min(requested, hard_cap))


def _expand_query_by_profile(query: str, profile: str) -> str:
    profile_norm = (profile or "gu-v1").strip().lower()
    cleaned = re.sub(r"\s+", " ", query.strip())
    if profile_norm in {"off", "none", "disabled"}:
        return cleaned

    if profile_norm == "gu-v1":
        if _GUJARATI_CHAR_RE.search(cleaned):
            return cleaned
        return cleaned

    logger.warning("Unknown MARQO_QUERY_EXPANSION_PROFILE=%s; using raw normalized query", profile)
    return cleaned


def _expand_query_for_retry(query: str) -> Optional[str]:
    """One widened query to try when the first one found nothing.

    Returns None when no expansion applies, or when the expansion would not
    actually change the query — there is no point paying for the same search
    twice.
    """
    extras: List[str] = []
    for pattern, expansion in _RETRY_EXPANSIONS:
        if pattern.search(query):
            for term in expansion.split():
                if term not in extras:
                    extras.append(term)
    if not extras:
        return None
    lowered = query.lower()
    novel = [term for term in extras if term.lower() not in lowered]
    if not novel:
        return None
    return f"{query} {' '.join(novel)}"


def _doc_key(hit: Dict[str, Any]) -> str:
    return (
        str(hit.get("doc_id") or "").strip()
        or str(hit.get("filename") or "").strip()
        or str(hit.get("name_en") or "").strip()
        or str(hit.get("name") or "").strip()
        or str(hit.get("_id") or "").strip()
    )


def _apply_doc_diversity(hits: List[Dict[str, Any]], top_k: int, max_per_doc: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    per_doc_counts: Dict[str, int] = {}

    for hit in hits:
        key = _doc_key(hit)
        count = per_doc_counts.get(key, 0)
        if count >= max_per_doc:
            continue
        per_doc_counts[key] = count + 1
        selected.append(hit)
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        for hit in hits:
            if hit in selected:
                continue
            selected.append(hit)
            if len(selected) >= top_k:
                break
    return selected


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _tokenize(value: str) -> List[str]:
    return _TOKEN_RE.findall(_normalize_text(value))


def _token_overlap_score(query: str, text: str) -> float:
    q_tokens = set(_tokenize(query))
    t_tokens = set(_tokenize(text))
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def _metadata_blob(hit: Dict[str, Any]) -> str:
    return " ".join(
        str(hit.get(k) or "")
        for k in (
            "name",
            "name_en",
            "name_gu",
            "filename",
            "title_en",
            "title_gu",
            "category_tags",
            "description",
            "doc_short_description",
            "doc_llm_description",
        )
    )


def _rerank_hits(
    query: str,
    hits: List[Dict[str, Any]],
    allowed_species: Optional[FrozenSet[str]] = None,
) -> List[Dict[str, Any]]:
    if not hits:
        return hits

    # Same switch that governs dropping, so turning the filter off leaves both
    # the result set and its order exactly as they were before issue #271.
    species_weight = (
        _parse_float_env("MARQO_OFF_SPECIES_PENALTY", 0.15)
        if _env_bool("VOICE_SPECIES_FILTER", True)
        else 0.0
    )

    raw_scores = [float(h.get("_score", h.get("score", 0.0)) or 0.0) for h in hits]
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    denom = (max_score - min_score) if max_score > min_score else 1.0

    rescored: List[Dict[str, Any]] = []
    for hit, raw in zip(hits, raw_scores):
        semantic = (raw - min_score) / denom
        text = str(hit.get("text") or "")
        metadata_text = _metadata_blob(hit)
        lexical_text = _token_overlap_score(query, text)
        lexical_meta = _token_overlap_score(query, metadata_text)
        lexical = max(lexical_text, lexical_meta)

        metadata_boost = 0.08 * lexical_meta
        reference_penalty = -0.12 if bool(hit.get("is_reference", False)) else 0.0
        # Only mixed-species documents reach this — a document about nothing but
        # another animal was already dropped before reranking.
        species_penalty = off_species_penalty(
            hit_species(hit, metadata_text), allowed_species, species_weight
        )
        rerank_score = (
            (0.62 * semantic) + (0.30 * lexical) + metadata_boost + reference_penalty + species_penalty
        )

        enriched = dict(hit)
        enriched["_rerank_score"] = rerank_score
        rescored.append(enriched)

    rescored.sort(key=lambda x: float(x.get("_rerank_score", 0.0)), reverse=True)
    return rescored


DocumentType = Literal['video', 'document']


class SearchHit(BaseModel):
    name: str = ""
    text: str = ""
    doc_id: str = ""
    type: str = "document"
    source: str = ""
    score: float = Field(default=0.0)
    id: str = Field(default="")

    class Config:
        extra = "ignore"
        populate_by_name = True

    @property
    def processed_text(self) -> str:
        cleaned = re.sub(r'\n{2,}', '\n\n', self.text)
        cleaned = re.sub(r'\t+', '\t', cleaned)
        cleaned = normalize_text_with_glossary(cleaned)
        return cleaned

    def __str__(self) -> str:
        return f"**{self.name}**\n```\n{self.processed_text}\n```\n"


async def search_documents(
    ctx: RunContext[FarmerContext],
    query: str,
    top_k: int = 12,
) -> str:
    """
    Semantic retrieval over veterinary/agri documents.

    Args:
        ctx: Tool context
        query: English keyword query for retrieval (required). Keep compact and intent-aligned.
        top_k: Requested number of final results (contract-clamped, default: 12)
    """
    try:
        # The caller's own words and herd used to be discarded here, which is
        # what let equine documents answer a cow question (issue #271).
        deps = getattr(ctx, "deps", None)
        caller_utterance = str(getattr(deps, "query", "") or "")
        farmer_info = str(getattr(deps, "farmer_info", "") or "")
        query = _validate_search_query(query)
        endpoint_url = os.getenv('MARQO_ENDPOINT_URL')
        if not endpoint_url:
            raise ValueError("Marqo endpoint URL is required")
        index_name = os.getenv('MARQO_INDEX_NAME', 'amul-veterinary-index')
        if not index_name:
            raise ValueError("Marqo index name is required")

        capabilities = await asyncio.to_thread(_get_index_capabilities_sync, endpoint_url, index_name)
        if capabilities.get("exists"):
            logger.info(
                "Index capabilities: tensor_fields=%s, text_tensor=%s, text_for_embedding_tensor=%s, has_is_reference=%s",
                capabilities.get("tensor_fields", []),
                capabilities.get("has_text_tensor"),
                capabilities.get("has_text_for_embedding_tensor"),
                capabilities.get("has_is_reference_filter"),
            )
        else:
            logger.warning("Could not inspect index '%s': %s", index_name, capabilities.get("error"))

        logger.info("Searching for '%s' in index '%s'", query, index_name)

        use_e5_query_prefix = _env_bool("MARQO_USE_E5_QUERY_PREFIX", True)
        exclude_reference_chunks = _env_bool("MARQO_EXCLUDE_REFERENCE", True)
        query_expansion_profile = os.getenv("MARQO_QUERY_EXPANSION_PROFILE", "gu-v1")
        final_top_k = _resolve_final_top_k(top_k)
        max_per_doc = int(os.getenv("MARQO_MAX_CHUNKS_PER_DOC", "2"))
        candidate_multiplier = int(os.getenv("MARQO_CANDIDATE_MULTIPLIER", "10"))
        candidate_cap = int(os.getenv("MARQO_CANDIDATE_CAP", "120"))
        hybrid_alpha = _parse_float_env("MARQO_HYBRID_ALPHA", 0.6)
        hybrid_rrfk = _parse_int_env("MARQO_HYBRID_RRFK", 60)
        search_limit = min(
            max(final_top_k * max(candidate_multiplier, 1), final_top_k),
            max(candidate_cap, final_top_k),
        )
        search_mode = (os.getenv("MARQO_SEARCH_MODE", "hybrid") or "hybrid").strip().lower()
        if search_mode not in {"hybrid", "tensor", "lexical"}:
            raise ValueError(f"Unsupported MARQO_SEARCH_MODE={search_mode}")

        async def _execute(raw_query: str, attempt: str) -> List[Dict[str, Any]]:
            expanded = _expand_query_by_profile(raw_query, query_expansion_profile)
            effective = _prepare_query_for_e5(expanded) if use_e5_query_prefix else expanded
            params: Dict[str, Any] = {"q": effective, "limit": search_limit}
            if search_mode == "hybrid":
                params["search_method"] = "hybrid"
                params["hybrid_parameters"] = {
                    "retrievalMethod": "disjunction",
                    "rankingMethod": "rrf",
                    "alpha": hybrid_alpha,
                    "rrfK": hybrid_rrfk,
                }
            else:
                params["search_method"] = search_mode
            if exclude_reference_chunks and capabilities.get("has_is_reference_filter", False):
                params["filter_string"] = "is_reference:false"

            base_metadata = {
                "endpoint_url": endpoint_url,
                "index_name": index_name,
                "search_mode": search_mode,
                "query_expansion_profile": query_expansion_profile,
                "attempt": attempt,
                "tool": "search_documents",
            }
            with start_observation(
                "marqo_search",
                input={"query": raw_query, "search_params": params},
                metadata=base_metadata,
            ) as observation:
                try:
                    hits = await asyncio.to_thread(_marqo_search_sync, endpoint_url, index_name, params)
                except Exception as e:
                    if search_mode != "hybrid":
                        if observation is not None:
                            observation.update(output={"error": str(e)}, metadata=base_metadata)
                        raise
                    logger.warning("Hybrid search failed, retrying with tensor search for query '%s'", raw_query)
                    fallback_params = {"q": effective, "limit": search_limit, "search_method": "tensor"}
                    if exclude_reference_chunks and capabilities.get("has_is_reference_filter", False):
                        fallback_params["filter_string"] = "is_reference:false"
                    if observation is not None:
                        observation.update(
                            metadata={**base_metadata, "fallback_mode": "tensor", "initial_error": str(e)}
                        )
                    hits = await asyncio.to_thread(
                        _marqo_search_sync, endpoint_url, index_name, fallback_params
                    )

                if observation is not None:
                    observation.update(output={"hit_count": len(hits)}, metadata=base_metadata)
            return hits

        # Who the answer is for. Decided once, from the caller's utterance, the
        # model's query and the herd on file — in that order of authority.
        # VOICE_SPECIES_FILTER=false restores the pre-#271 behaviour outright:
        # no hit is dropped and no hit is demoted.
        species_filter_on = _env_bool("VOICE_SPECIES_FILTER", True)
        allowed_species = (
            resolve_caller_species(caller_utterance, query, farmer_info) if species_filter_on else None
        )

        def _partition(hits: List[Dict[str, Any]]):
            if not species_filter_on:
                return hits, []
            return partition_hits_by_species(hits, allowed_species, _metadata_blob)

        results = await _execute(query, "primary")
        kept, dropped = _partition(results)
        gap_reason = _GAP_REASON_SPECIES if (dropped and not kept) else _GAP_REASON_EMPTY
        if dropped:
            logger.info(
                "Species filter dropped %s/%s hits: query=%s allowed=%s",
                len(dropped), len(results), query, sorted(allowed_species) if allowed_species else "any",
            )

        # Nothing survived. Widen the vocabulary once before conceding a gap —
        # the corpus says "anestrus" where the caller says "not coming into heat".
        retry_query = _expand_query_for_retry(query) if not kept else None
        if retry_query:
            logger.info("Empty result set; retrying with expanded query=%s", retry_query)
            retry_results = await _execute(retry_query, "expanded")
            retry_kept, retry_dropped = _partition(retry_results)
            if retry_kept:
                kept = retry_kept
            elif retry_dropped:
                gap_reason = _GAP_REASON_SPECIES

        results = kept
        rerank_mode = (os.getenv("MARQO_RERANK_MODE", "bm25lite") or "bm25lite").strip().lower()
        if rerank_mode not in {"off", "none", "disabled"}:
            results = _rerank_hits(query, results, allowed_species)
        results = _apply_doc_diversity(results, top_k=final_top_k, max_per_doc=max_per_doc)

        logger.info(
            "Search completed: query=%s mode=%s top_k=%s hits=%s profile=%s allowed_species=%s",
            query,
            search_mode,
            final_top_k,
            len(results),
            query_expansion_profile,
            sorted(allowed_species) if allowed_species else "any",
        )

        if len(results) == 0:
            return _RETRIEVAL_GAP.format(query=query, reason=gap_reason)

        search_hits = []
        for hit in results:
            processed_hit = {
                "name": hit.get("name") or hit.get("name_en") or hit.get("name_gu") or hit.get("filename", ""),
                "text": hit.get("text", ""),
                "doc_id": hit.get("doc_id", hit.get("_id", "")),
                "type": hit.get("type", "document"),
                "source": hit.get("source", ""),
                "score": hit.get("_rerank_score", hit.get("_score", hit.get("score", 0.0))),
                "id": hit.get("_id", hit.get("id", "")),
            }
            search_hits.append(SearchHit(**processed_hit))

        document_string = '\n\n----\n\n'.join([str(document) for document in search_hits])
        return "> Search Results for `" + query + "`\n\n" + document_string
    except Exception as e:
        logger.error("Error searching documents: %s for query: %s", e, query)
        raise ModelRetry("Error searching documents, please try again")
