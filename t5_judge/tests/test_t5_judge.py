# tests/test_t5_judge.py
import os
import re
import pytest

# Adjust import if your package path is different
from t5_judge.judge import T5Judge, DEFAULT_LABELS

YES, NO, UNKNOWN = DEFAULT_LABELS  # ("yes", "no", "unknown")


# -----------------------------
# Helpers for monkeypatching
# -----------------------------
def _fake_probs_yes(*args, **kwargs):
    return {YES: 0.8, NO: 0.1, UNKNOWN: 0.1}

def _fake_probs_no(*args, **kwargs):
    return {YES: 0.1, NO: 0.8, UNKNOWN: 0.1}

def _fake_probs_unknown(*args, **kwargs):
    return {YES: 0.2, NO: 0.2, UNKNOWN: 0.6}

def _fake_probs_from_prompt(self, prompt: str, labels):
    """
    Deterministic stub that inspects the FACT in the prompt:
      - if FACT contains 'still working' => YES
      - if FACT contains 'broke down'    => NO
      - else                              => UNKNOWN
    """
    m = re.search(r"FACT:\s*(.+?)\s*\\n?Answer:", prompt, re.DOTALL)
    fact = m.group(1).lower() if m else ""
    if "still working" in fact:
        return _fake_probs_yes()
    if "broke down" in fact:
        return _fake_probs_no()
    return _fake_probs_unknown()


# -----------------------------
# Unit tests (fast)
# -----------------------------
def test_judge_claim_yes(monkeypatch):
    monkeypatch.setattr(T5Judge, "_label_probs", _fake_probs_yes, raising=True)
    j = T5Judge(model_name="google/flan-t5-large", device="cpu")
    res = j.judge_claim("EVIDENCE", "FACT")
    assert res.probs[YES] == pytest.approx(0.8)
    assert res.truth_score > 0  # p(yes) - p(no)


def test_judge_claim_no(monkeypatch):
    monkeypatch.setattr(T5Judge, "_label_probs", _fake_probs_no, raising=True)
    j = T5Judge(device="cpu")
    res = j.judge_claim("EVIDENCE", "FACT")
    assert res.probs[NO] == pytest.approx(0.8)
    assert res.truth_score < 0  # p(yes) - p(no)


def test_judge_claim_unknown(monkeypatch):
    monkeypatch.setattr(T5Judge, "_label_probs", _fake_probs_unknown, raising=True)
    j = T5Judge(device="cpu")
    res = j.judge_claim("EVIDENCE", "FACT")
    assert res.probs[UNKNOWN] == pytest.approx(0.6)
    assert res.truth_score == pytest.approx(0.0)  # yes==no => 0


def test_batch_judge_claims_respects_batch_size(monkeypatch):
    monkeypatch.setattr(T5Judge, "_label_probs", _fake_probs_from_prompt, raising=True)
    j = T5Judge(device="cpu")

    pairs = [
        ("The car broke down, and John had to drive it to a garage.",
         "The car was still working after it had a problem."),   # YES (contains 'still working')
        ("The car broke down, and John had to drive it to a garage.",
         "The car broke down and stopped."),                      # NO (contains 'broke down')
        ("No clear evidence provided.", "It might rain later."),  # UNKNOWN
        ("Laptop failed to boot.", "The laptop broke down."),     # NO
        ("Engine repaired successfully.", "The engine was still working."),  # YES
    ]
    out = j.batch_judge_claims(pairs, batch_size=2)
    assert len(out) == len(pairs)

    # Check the first, second, and last classification directions via truth_score sign
    assert out[0].truth_score > 0     # YES
    assert out[1].truth_score < 0     # NO
    assert abs(out[2].truth_score) < 1e-6  # UNKNOWN approx
    assert out[3].truth_score < 0     # NO
    assert out[4].truth_score > 0     # YES


# -----------------------------
# Optional integration test (slow)
# -----------------------------
@pytest.mark.skipif(os.getenv("RUN_INTEGRATION_JUDGE", "0") != "1",
                    reason="Set RUN_INTEGRATION_JUDGE=1 to run model-backed test.")
def test_integration_judge_claim_real_model_cpu():
    """
    Runs the real model on CPU to ensure end-to-end plumbing works.
    This will download weights the first time; keep it skipped in CI by default.
    """
    j = T5Judge(model_name="google/flan-t5-large", device="cpu")
    evidence = "The car broke down, and John had to drive it to a garage."
    fact = "The car was still working after it had a problem."
    res = j.judge_claim(evidence, fact)

    # We don't assert exact probabilities; just sanity-check the shape and score bounds
    assert set(res.probs.keys()) == set(DEFAULT_LABELS)
    assert -1.0 <= res.truth_score <= 1.0
