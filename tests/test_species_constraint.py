"""Issue #271: a dairy caller must never be answered with another animal's care.

A cracked-hoof question from a cow farmer came back as farriery advice for a
horse, because retrieval ranked on similarity alone and nothing downstream of
Marqo knew which animal was on the line. These tests pin the three parts of the
fix: what counts as another animal, that those hits never reach the model, and
that an empty result set concedes the gap instead of offering a substitute.
"""
import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.deps import FarmerContext
from agents.services.species import (
    detect_species,
    partition_hits_by_species,
    resolve_caller_species,
)
from agents.tools import search as search_mod


HORSE_HIT = {
    "_id": "h1",
    "name": "Hoof care",
    "text": "A crack in the horse's hoof should be rasped by a farrier and kept dry.",
    "_score": 0.91,
}
COW_HIT = {
    "_id": "c1",
    "name": "Cattle foot health",
    "text": "Cracked hooves in cows respond to a zinc sulphate footbath and dry bedding.",
    "_score": 0.80,
}
GENERAL_HIT = {
    "_id": "g1",
    "name": "Footbath preparation",
    "text": "Prepare the footbath at five percent and renew it every fifty animals.",
    "_score": 0.70,
}
GOAT_HIT = {
    "_id": "gt1",
    "name": "Goat management",
    "text": "Goats need hoof trimming every eight weeks to avoid overgrowth.",
    "_score": 0.75,
}


class TestDetection:
    def test_reads_the_animal_out_of_english_and_gujarati(self):
        assert detect_species(HORSE_HIT["text"]) == frozenset({"equine"})
        assert detect_species(COW_HIT["text"]) == frozenset({"cattle"})
        assert detect_species("ઘોડાની ખરી ફાટી ગઈ છે") == frozenset({"equine"})
        assert detect_species("ભેંસ ગરમીમાં આવતી નથી") == frozenset({"cattle"})

    def test_general_husbandry_text_names_no_animal(self):
        assert detect_species(GENERAL_HIT["text"]) == frozenset()

    def test_words_that_only_look_like_animals_are_left_alone(self):
        # "kid" and "ram" are ordinary words here; treating them as goat and
        # sheep would drop documents the caller actually needed.
        assert detect_species("assistance for the farmer's kids under the scheme") == frozenset()
        assert detect_species("submitted by Ram Patel, extension officer") == frozenset()
        assert detect_species("cowpea is a useful green fodder") == frozenset()


class TestCallerSpecies:
    def test_the_callers_own_words_decide(self):
        assert resolve_caller_species("my cow's hoof is cracked", "hoof crack", "") == frozenset({"cattle"})

    def test_an_explicit_other_species_is_honoured(self):
        # A cattle farmer who asks about goats gets goat documents.
        allowed = resolve_caller_species("what diseases do goats get", "goat diseases", "- **Cows:** 4")
        assert allowed == frozenset({"goat"})

    def test_the_herd_answers_when_the_caller_does_not(self):
        allowed = resolve_caller_species("my animal's hoof is cracked", "hoof crack treatment", "- **Buffalo:** 3")
        assert allowed == frozenset({"cattle"})

    def test_nothing_identifies_the_animal(self):
        assert resolve_caller_species("hoof crack", "hoof crack treatment", "") is None


class TestPartition:
    def _split(self, hits, allowed):
        return partition_hits_by_species(hits, allowed, search_mod._metadata_blob)

    def test_equine_is_dropped_for_a_cow_caller(self):
        kept, dropped = self._split([HORSE_HIT, COW_HIT, GENERAL_HIT], frozenset({"cattle"}))
        assert [h["_id"] for h in kept] == ["c1", "g1"]
        assert [h["_id"] for h in dropped] == ["h1"]

    def test_equine_is_dropped_even_when_the_caller_is_unknown(self):
        # The line is a dairy helpline; equine advice is wrong on it regardless.
        kept, dropped = self._split([HORSE_HIT, GENERAL_HIT], None)
        assert [h["_id"] for h in kept] == ["g1"]
        assert [h["_id"] for h in dropped] == ["h1"]

    def test_other_species_survive_when_the_caller_is_unknown(self):
        kept, _ = self._split([GOAT_HIT, GENERAL_HIT], None)
        assert [h["_id"] for h in kept] == ["gt1", "g1"]

    def test_goat_documents_are_kept_for_a_goat_question(self):
        kept, dropped = self._split([GOAT_HIT, COW_HIT], frozenset({"goat"}))
        assert [h["_id"] for h in kept] == ["gt1"]
        assert [h["_id"] for h in dropped] == ["c1"]

    def test_a_cattle_document_that_merely_mentions_horses_survives(self):
        mixed = {
            "_id": "m1",
            "name": "Hoof anatomy",
            "text": "Unlike the horse, the cow bears weight on two claws.",
            "_score": 0.6,
        }
        kept, dropped = self._split([mixed], frozenset({"cattle"}))
        assert [h["_id"] for h in kept] == ["m1"]
        assert dropped == []

    def test_the_documents_name_counts_when_the_chunk_stays_silent(self):
        chunk = {
            "_id": "h2",
            "name_en": "Shoeing the draught horse",
            "text": "Rasp the wall back to a level bearing surface before fitting.",
            "_score": 0.9,
        }
        kept, dropped = self._split([chunk], frozenset({"cattle"}))
        assert kept == []
        assert [h["_id"] for h in dropped] == ["h2"]


class TestRerank:
    def test_a_mixed_document_ranks_below_a_pure_one(self):
        mixed = dict(COW_HIT, _id="m1", text="Cows and horses both crack hooves in dry weather.")
        ranked = search_mod._rerank_hits("cracked hoof cow", [mixed, COW_HIT], frozenset({"cattle"}))
        assert [h["_id"] for h in ranked] == ["c1", "m1"]


def _ctx(query, farmer_info=""):
    return SimpleNamespace(deps=FarmerContext(query=query, farmer_info=farmer_info))


@pytest.fixture
def marqo(monkeypatch):
    """Drive search_documents against a scripted Marqo."""
    monkeypatch.setenv("MARQO_ENDPOINT_URL", "http://marqo.test")
    monkeypatch.setenv("MARQO_INDEX_NAME", "test-index")
    monkeypatch.setattr(
        search_mod,
        "_get_index_capabilities_sync",
        lambda url, name: {"exists": True, "has_is_reference_filter": False, "tensor_fields": ["text"]},
    )
    calls = []

    def _install(responses):
        """`responses` maps a substring of the query to the hits it returns."""
        def _search(url, index, params):
            q = params["q"]
            calls.append(q)
            for needle, hits in responses:
                if needle in q:
                    return list(hits)
            return []
        monkeypatch.setattr(search_mod, "_marqo_search_sync", _search)
        return calls

    return _install


class TestSearchDocuments:
    def test_horse_chunks_never_reach_a_cow_caller(self, marqo):
        marqo([("", [HORSE_HIT, COW_HIT, GENERAL_HIT])])
        out = asyncio.run(
            search_mod.search_documents(_ctx("my cow's hoof is cracked"), "cracked hoof treatment")
        )
        assert "farrier" not in out
        assert "horse" not in out.lower()
        assert "footbath" in out

    def test_an_all_equine_result_set_becomes_an_honest_gap(self, marqo):
        marqo([("", [HORSE_HIT])])
        out = asyncio.run(
            search_mod.search_documents(_ctx("my cow's hoof is cracked"), "cracked hoof treatment")
        )
        assert "RETRIEVAL_GAP" in out
        assert "different animal" in out
        assert "veterinary" in out
        assert "farrier" not in out

    def test_an_empty_index_concedes_rather_than_substitutes(self, marqo):
        marqo([])
        out = asyncio.run(search_mod.search_documents(_ctx("cattle shed subsidy"), "cattle shed subsidy"))
        assert "RETRIEVAL_GAP" in out
        assert "nothing on this topic" in out
        assert "dairy society" in out

    def test_heat_questions_get_a_second_chance_at_the_corpus_vocabulary(self, marqo):
        # The caller says "not coming into heat"; the corpus says "anestrus".
        anestrus = {
            "_id": "a1",
            "name": "Anestrus in buffalo",
            "text": "Anestrus in buffalo responds to mineral mixture and a veterinary hormone protocol.",
            "_score": 0.8,
        }
        calls = marqo([("anestrus", [anestrus])])
        out = asyncio.run(
            search_mod.search_documents(
                _ctx("my buffalo is not coming into heat"), "buffalo not coming into heat"
            )
        )
        assert len(calls) == 2, calls
        assert "RETRIEVAL_GAP" not in out
        assert "Anestrus in buffalo" in out

    def test_the_widened_query_still_respects_the_callers_animal(self, marqo):
        marqo([("anestrus", [HORSE_HIT])])
        out = asyncio.run(
            search_mod.search_documents(
                _ctx("my buffalo is not coming into heat"), "buffalo not coming into heat"
            )
        )
        assert "RETRIEVAL_GAP" in out
        assert "different animal" in out

    def test_a_working_search_is_not_retried(self, marqo):
        calls = marqo([("", [COW_HIT])])
        asyncio.run(search_mod.search_documents(_ctx("cow in heat"), "cow heat detection"))
        assert len(calls) == 1

    def test_the_filter_can_be_switched_off(self, marqo, monkeypatch):
        monkeypatch.setenv("VOICE_SPECIES_FILTER", "false")
        marqo([("", [HORSE_HIT])])
        out = asyncio.run(search_mod.search_documents(_ctx("my cow's hoof is cracked"), "cracked hoof"))
        assert "farrier" in out


class TestRetryExpansion:
    def test_it_widens_the_topics_the_issue_named(self):
        assert "anestrus" in search_mod._expand_query_for_retry("buffalo not coming into heat")
        assert "repeat breeder" in (search_mod._expand_query_for_retry("cow not conceiving") or "")
        assert "subsidy" in (search_mod._expand_query_for_retry("cattle shed scheme") or "")

    def test_it_stays_out_of_the_way_elsewhere(self):
        assert search_mod._expand_query_for_retry("cow mastitis treatment") is None
