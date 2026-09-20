"""Which animal a retrieved document is about, and which one this caller keeps.

This is a dairy helpline. A cracked-hoof question from a cow farmer was answered
with farriery advice for a horse, because retrieval ranks on similarity and
"cracked hoof" literature is dominated by equine sources — nothing downstream of
Marqo ever asked which animal the caller was holding. See issue #271.

The prompt already carries a species-defaulting rule, but a single instruction
does not outvote twelve retrieved chunks that all talk about horses. So the
decision is made here, on the hits, before the model ever sees them.

Species are grouped the way the helpline thinks about them, not the way a
taxonomist would: cow, buffalo, calf, heifer and bull are one group (`cattle`),
because guidance for either dairy animal is acceptable to either caller. The
groups that matter are the ones we need to keep *out*.

Matching is deliberately crude — term presence, not classification. A document
that never names an animal is treated as general husbandry and always kept,
which is most of the corpus; only a document that positively identifies itself
as being about another animal is dropped.
"""
from __future__ import annotations

import re
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

CATTLE = "cattle"
EQUINE = "equine"

# Latin terms match on word boundaries; Gujarati entries are stems matched as
# substrings, because Gujarati inflects the noun (ઘોડો / ઘોડા / ઘોડી) and a
# boundary would miss every form but the one written here.
_SPECIES_TERMS: Dict[str, Tuple[Sequence[str], Sequence[str]]] = {
    CATTLE: (
        (
            "cow", "cows", "cattle", "bovine", "buffalo", "buffaloes", "buffalos",
            "heifer", "heifers", "bullock", "bullocks", "calf", "calves",
            "cowshed", "dairy animal", "dairy animals", "milch",
        ),
        ("ગાય", "ભેંસ", "ભેસ", "વાછરડ", "પાડી", "પાડો", "બળદ", "વાછરુ"),
    ),
    EQUINE: (
        ("horse", "horses", "equine", "stallion", "foal", "foals", "pony",
         "ponies", "donkey", "donkeys", "mule", "mules", "farrier"),
        ("ઘોડ", "ગધેડ", "ખચ્ચર"),
    ),
    # "kid" and "ram" are deliberately absent: both are ordinary words in this
    # corpus's surroundings (scheme text about children, the name Ram), and a
    # false positive here removes a document the caller needed.
    "goat": (("goat", "goats", "caprine", "doeling", "buckling"), ("બકર", "બોકડ")),
    "sheep": (("sheep", "lamb", "lambs", "ewe", "ewes", "ovine"), ("ઘેટ", "ઘેંટ")),
    "poultry": (
        ("poultry", "chicken", "chickens", "hen", "hens", "broiler", "broilers",
         "layer bird", "chick", "chicks", "duck", "ducks"),
        ("મરઘ", "કૂકડ", "બતક"),
    ),
    "camel": (("camel", "camels"), ("ઊંટ", "ઉંટ")),
    "pig": (("pig", "pigs", "swine", "piglet", "piglets", "porcine"), ("ભૂંડ", "ડુક્કર")),
    "dog": (("dog", "dogs", "puppy", "puppies", "canine"), ("કૂતર", "કુતર", "શ્વાન")),
}


def _compile(latin: Sequence[str], gujarati: Sequence[str]) -> re.Pattern:
    parts: List[str] = []
    if latin:
        parts.append(r"\b(?:" + "|".join(re.escape(t) for t in latin) + r")\b")
    if gujarati:
        parts.append("(?:" + "|".join(re.escape(t) for t in gujarati) + ")")
    return re.compile("|".join(parts), re.IGNORECASE | re.UNICODE)


_SPECIES_PATTERNS: Dict[str, re.Pattern] = {
    name: _compile(latin, gujarati) for name, (latin, gujarati) in _SPECIES_TERMS.items()
}


def detect_species(text: str) -> FrozenSet[str]:
    """Species named anywhere in `text`. Empty when it names none."""
    if not text:
        return frozenset()
    found = {name for name, pattern in _SPECIES_PATTERNS.items() if pattern.search(text)}
    return frozenset(found)


def resolve_caller_species(
    caller_utterance: str = "",
    tool_query: str = "",
    farmer_info: str = "",
) -> Optional[FrozenSet[str]]:
    """What this caller is asking about, best evidence first.

    The caller's own words win: a cow farmer who explicitly asks about goats
    gets goat documents. The model's keyword query is only consulted when the
    utterance named nothing, because the query is the model's paraphrase and can
    drift. The herd profile is the last resort — it is what makes "my animal's
    hoof is cracked" a cattle question without the caller saying so.

    Returns None when nothing identifies the animal, which leaves every species
    admissible except the one this helpline should never give advice about.
    """
    for source in (caller_utterance, tool_query):
        named = detect_species(source)
        if named:
            return named
    if CATTLE in detect_species(farmer_info):
        return frozenset({CATTLE})
    return None


def _is_off_line_equine(doc_species: FrozenSet[str]) -> bool:
    """Equine content with no cattle content beside it.

    A cattle document that mentions horses in passing ("unlike in horses…") is
    still a cattle document and survives; a pure farriery chunk does not.
    """
    return EQUINE in doc_species and CATTLE not in doc_species


def hit_species(hit: Dict[str, object], extra_text: str = "") -> FrozenSet[str]:
    """Species a Marqo hit is about, read from its chunk text plus its metadata.

    Metadata matters on its own: a chunk lifted from the middle of a horse
    document may never repeat the word, but the document's name does.
    """
    return detect_species(f"{hit.get('text') or ''}\n{extra_text}")


def partition_hits_by_species(
    hits: Iterable[Dict[str, object]],
    allowed: Optional[FrozenSet[str]],
    metadata_of,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Split hits into (kept, dropped) for this caller.

    `metadata_of` yields the searchable metadata blob for a hit; it is passed in
    so this module stays free of the Marqo hit shape.
    """
    kept: List[Dict[str, object]] = []
    dropped: List[Dict[str, object]] = []
    for hit in hits:
        doc_species = hit_species(hit, metadata_of(hit))
        if not doc_species:
            kept.append(hit)
            continue
        if allowed is None:
            # Nothing told us what the caller keeps. Everything stays except
            # equine guidance, which is wrong on a dairy line no matter who is
            # calling — the rest is left to the prompt's cattle-default rule.
            (dropped if _is_off_line_equine(doc_species) else kept).append(hit)
            continue
        (kept if doc_species & allowed else dropped).append(hit)
    return kept, dropped


def off_species_penalty(
    doc_species: FrozenSet[str],
    allowed: Optional[FrozenSet[str]],
    weight: float,
) -> float:
    """Demote a hit that is partly about an animal this caller does not keep.

    Only mixed documents reach this: a purely off-species hit was already
    dropped. Demoting rather than dropping keeps a genuine cattle document that
    happens to draw a comparison, while pushing it below the documents that are
    only about the caller's animal.
    """
    if not doc_species:
        return 0.0
    foreign = doc_species - (allowed or frozenset({CATTLE}))
    return -weight if foreign else 0.0
