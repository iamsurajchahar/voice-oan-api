#!/usr/bin/env python3
"""Ask the live index whether it can answer the questions this helpline gets.

Issue #271 reported "no information available" for bringing a buffalo into
heat — a core reproductive-health topic for a dairy line. That report is either
a corpus gap (the documents are not indexed) or a retrieval failure (they are,
and the query does not reach them), and the two need opposite fixes: one is an
ingestion ticket, the other is a code change. Nothing in the repo could tell
them apart, because the corpus lives in Marqo and only production ever queried
it. This script is that missing instrument.

It runs a fixed probe set — the reproduction topics and the Gujarat cattle-shed
subsidy schemes the issue named, plus controls that are known to work — and
reports, per probe, what came back and whether the widened retry rescued it.
Read the output as:

  HIT      documents exist and the caller's own phrasing finds them.
  RETRY    documents exist but only the widened vocabulary finds them —
           a retrieval problem, fixable here.
  SPECIES  everything that matched was about another animal.
  MISS     nothing matched either query — a corpus gap, fixable only by
           indexing the material.

Usage:
    MARQO_ENDPOINT_URL=... MARQO_INDEX_NAME=... python scripts/audit_corpus_coverage.py
    python scripts/audit_corpus_coverage.py --json > coverage.json
    python scripts/audit_corpus_coverage.py --topic reproduction

Read-only: it issues searches and writes nothing.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.services.species import partition_hits_by_species, resolve_caller_species  # noqa: E402
from agents.tools.search import (  # noqa: E402
    _apply_doc_diversity,
    _expand_query_for_retry,
    _get_index_capabilities_sync,
    _marqo_search_sync,
    _metadata_blob,
)

# Each probe is what a caller actually says, paired with the keyword query the
# agent would build from it. Both matter: the second is what reaches Marqo, and
# the first is what decides which animal the answer has to be about.
PROBES: Dict[str, List[tuple]] = {
    "reproduction": [
        ("my buffalo is not coming into heat", "buffalo not coming into heat"),
        ("how do I bring my cow into heat", "bring cow into heat"),
        ("my cow is not showing heat signs", "cow heat signs not showing"),
        ("how do I know when my buffalo is in heat", "buffalo heat detection timing"),
        ("my cow has been inseminated four times and is not conceiving", "cow repeat breeder not conceiving"),
        ("when should I inseminate after the animal comes into heat", "insemination timing after heat"),
        ("my buffalo has not calved for two years", "buffalo long calving interval infertility"),
        ("what feed brings an animal into heat", "mineral mixture anestrus feeding"),
    ],
    "shed_subsidy": [
        ("is there a subsidy for building a cattle shed", "cattle shed subsidy scheme"),
        ("what help is there for a cowshed in Gujarat", "Gujarat cowshed construction assistance"),
        ("how do I apply for the shed scheme", "animal husbandry shed scheme application"),
        ("how much money do I get for a shed", "cattle shed subsidy amount eligibility"),
    ],
    "control": [
        ("my cow has mastitis", "cow mastitis symptoms treatment"),
        ("how much green fodder should I give", "green fodder quantity dairy cow"),
        ("my cow's hoof is cracked", "cracked hoof treatment cattle"),
    ],
}

VERDICTS = ("HIT", "RETRY", "SPECIES", "MISS")


def _search(endpoint: str, index: str, query: str, limit: int, has_reference: bool) -> List[dict]:
    params = {
        "q": f"query: {query}",
        "limit": limit,
        "search_method": "hybrid",
        "hybrid_parameters": {
            "retrievalMethod": "disjunction",
            "rankingMethod": "rrf",
            "alpha": 0.6,
            "rrfK": 60,
        },
    }
    if has_reference:
        params["filter_string"] = "is_reference:false"
    try:
        return _marqo_search_sync(endpoint, index, params)
    except Exception:
        params.pop("hybrid_parameters", None)
        params["search_method"] = "tensor"
        return _marqo_search_sync(endpoint, index, params)


def _names(hits: List[dict], limit: int = 3) -> List[str]:
    seen: List[str] = []
    for hit in hits:
        name = str(hit.get("name") or hit.get("name_en") or hit.get("filename") or hit.get("_id") or "")
        if name and name not in seen:
            seen.append(name)
        if len(seen) >= limit:
            break
    return seen


async def _probe(endpoint: str, index: str, utterance: str, query: str, has_reference: bool) -> dict:
    allowed = resolve_caller_species(utterance, query, "")
    raw = await asyncio.to_thread(_search, endpoint, index, query, 60, has_reference)
    kept, dropped = partition_hits_by_species(raw, allowed, _metadata_blob)
    kept = _apply_doc_diversity(kept, top_k=12, max_per_doc=2)

    result = {
        "utterance": utterance,
        "query": query,
        "species": sorted(allowed) if allowed else None,
        "raw_hits": len(raw),
        "kept": len(kept),
        "off_species": len(dropped),
        "retry_query": None,
        "retry_kept": None,
        "documents": _names(kept),
    }

    if kept:
        result["verdict"] = "HIT"
        return result

    retry_query = _expand_query_for_retry(query)
    if retry_query:
        result["retry_query"] = retry_query
        retry_raw = await asyncio.to_thread(_search, endpoint, index, retry_query, 60, has_reference)
        retry_kept, retry_dropped = partition_hits_by_species(retry_raw, allowed, _metadata_blob)
        result["retry_kept"] = len(retry_kept)
        if retry_kept:
            result["verdict"] = "RETRY"
            result["documents"] = _names(_apply_doc_diversity(retry_kept, top_k=12, max_per_doc=2))
            return result
        dropped = dropped or retry_dropped

    result["verdict"] = "SPECIES" if dropped else "MISS"
    return result


async def _run(topics: List[str], as_json: bool) -> int:
    endpoint = os.getenv("MARQO_ENDPOINT_URL")
    index = os.getenv("MARQO_INDEX_NAME", "amul-veterinary-index")
    if not endpoint:
        print("MARQO_ENDPOINT_URL is not set — nothing to audit.", file=sys.stderr)
        return 2

    capabilities = await asyncio.to_thread(_get_index_capabilities_sync, endpoint, index)
    if not capabilities.get("exists"):
        print(f"Cannot read index {index!r}: {capabilities.get('error')}", file=sys.stderr)
        return 2
    has_reference = bool(capabilities.get("has_is_reference_filter"))

    report: Dict[str, List[dict]] = {}
    for topic in topics:
        report[topic] = [
            await _probe(endpoint, index, utterance, query, has_reference)
            for utterance, query in PROBES[topic]
        ]

    if as_json:
        print(json.dumps({"index": index, "topics": report}, ensure_ascii=False, indent=2))
        return 0

    totals = {v: 0 for v in VERDICTS}
    for topic, rows in report.items():
        print(f"\n=== {topic} ===")
        for row in rows:
            totals[row["verdict"]] += 1
            detail = f"{row['kept']} kept / {row['raw_hits']} hits"
            if row["off_species"]:
                detail += f", {row['off_species']} off-species"
            if row["retry_query"]:
                detail += f", retry -> {row['retry_kept']}"
            print(f"  {row['verdict']:<8} {row['utterance']}")
            print(f"           {detail}")
            if row["documents"]:
                print(f"           {'; '.join(row['documents'])}")

    print("\n=== summary ===")
    for verdict in VERDICTS:
        print(f"  {verdict:<8} {totals[verdict]}")
    if totals["MISS"]:
        print("\n  MISS means the corpus has nothing on that topic — an ingestion ticket,")
        print("  not a code change. RETRY means retrieval, not coverage, was the problem.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic", action="append", choices=sorted(PROBES), help="Audit one topic (repeatable).")
    parser.add_argument("--json", action="store_true", help="Emit the raw report instead of the summary.")
    args = parser.parse_args()
    return asyncio.run(_run(args.topic or sorted(PROBES), args.json))


if __name__ == "__main__":
    raise SystemExit(main())
