"""Issue #271 prompt contract, pinned across every prompt variant.

The retrieval fix keeps off-species documents away from the model, but the model
also has to know what to do with what is left: stay on the caller's animal,
never speak as a farrier, and treat a retrieval gap as a gap rather than an
invitation to reach for a neighbouring topic. A variant that silently loses one
of those lines is the regression these tests exist to catch.
"""
from pathlib import Path

import pytest

PROMPTS = Path(__file__).resolve().parents[1] / "assets" / "prompts"
PROMPT_FILES = [
    "voice_system_translation_pipeline_en.md",
    "voice_system_translation_pipeline_gpt5_1_en.md",
    "voice_system_translation_pipeline_gemma4_en.md",
]


@pytest.fixture(params=PROMPT_FILES)
def prompt(request):
    return (PROMPTS / request.param).read_text(encoding="utf-8").lower()


def test_equine_guidance_is_banned_by_name(prompt):
    assert "never give equine guidance" in prompt
    for animal in ("horse", "donkey", "mule", "pony", "foal"):
        assert animal in prompt, f"{animal} is not named in the ban"
    assert "farrier" in prompt


def test_the_caller_outranks_the_documents(prompt):
    assert "wins over the documents" in prompt


def test_a_retrieval_gap_is_final_and_carries_a_next_step(prompt):
    assert "retrieval_gap" in prompt
    assert "retrieved earlier in this call" in prompt
    assert "nearest veterinary dispensary" in prompt


def test_a_topic_shift_forces_fresh_retrieval(prompt):
    assert "topic shift" in prompt
    assert "shed subsidy question is never answered with treatment advice" in prompt
