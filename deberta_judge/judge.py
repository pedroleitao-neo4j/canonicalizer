#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import os, time, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_MODEL = os.getenv("MODEL", "microsoft/deberta-v2-xxlarge-mnli")
DEFAULT_LABELS = ("entailed", "contradicted", "unknown")

@dataclass(frozen=True)
class JudgeResult:
    labels: Tuple[str, str, str]
    probs: Dict[str, float]
    truth_score: float


class DebertaJudge:
    """Deterministic label scorer using DeBERTa-v3 (MNLI-style)."""

    def __init__(
        self,
        model_name=DEFAULT_MODEL,
        device=None,
        torch_dtype=None,
        max_length: int = 512,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.max_length = max_length

        if torch_dtype is None and device == "cuda":
            torch_dtype = torch.float16

        # safer tokenizer call (some MNLI checkpoints ship only slow tokenizers)
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch_dtype
        )
        self.model.to(device).eval()

        # label mapping
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        self.idx_entail = label2id.get("entailment")
        self.idx_contra = label2id.get("contradiction")
        self.idx_neutral = label2id.get("neutral")
        if None in (self.idx_entail, self.idx_contra, self.idx_neutral):
            raise ValueError(f"Unexpected label set: {id2label}")

        torch.set_grad_enabled(False)
        if device == "cuda":
            torch.backends.cudnn.benchmark = False

    # internal scoring helper
    @torch.inference_mode()
    def _score_batch(self, premises: List[str], hypotheses: List[str]) -> List[JudgeResult]:
        enc = self.tok(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits  # [B, 3]
        probs = torch.softmax(logits, dim=-1)

        results: List[JudgeResult] = []
        for p in probs:
            p_ent = float(p[self.idx_entail])
            p_con = float(p[self.idx_contra])
            p_neu = float(p[self.idx_neutral])
            results.append(
                JudgeResult(
                    labels=DEFAULT_LABELS,
                    probs={
                        "entailed": p_ent,
                        "contradicted": p_con,
                        "unknown": p_neu,
                    },
                    truth_score=p_ent - p_con,
                )
            )
        return results

    # public API
    def judge_claim(self, evidence: str, fact: str, labels=DEFAULT_LABELS) -> JudgeResult:
        return self.batch_judge_claims([(evidence, fact)], labels, batch_size=1)[0]

    def batch_judge_claims(
        self, pairs: Sequence[Tuple[str, str]], labels=DEFAULT_LABELS, batch_size: int = 32
    ) -> List[JudgeResult]:
        results: List[JudgeResult] = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            premises, hypotheses = zip(*batch)
            results.extend(self._score_batch(list(premises), list(hypotheses)))
        return results


# Demo / benchmark

def run_bench(judge, pairs, batch_size, repeat=3):
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        _ = judge.batch_judge_claims(pairs, batch_size=batch_size)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.time()
        times.append(t1 - t0)
    return sum(times) / len(times)


def main():
    use_dtype = None
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability()[0]
        use_dtype = torch.bfloat16 if major >= 8 else torch.float16

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Model:  {DEFAULT_MODEL}")
    judge = DebertaJudge(model_name=DEFAULT_MODEL, torch_dtype=use_dtype)

    pairs = [
        ("OpenAI was founded in 2015 by a group including Elon Musk and Sam Altman.", "OpenAI was founded in 2015."),
        ("Apple introduced the iPhone in 2007 during a keynote by Steve Jobs.", "The iPhone launched in 2007."),
        ("The Nile is the longest river in the world, longer than the Amazon.", "The Amazon is the longest river."),
        ("Paris is the capital of France.", "Paris is the capital of Germany."),
        ("Microsoft acquired GitHub in 2018.", "GitHub was acquired by Microsoft."),
        ("The square root of 16 is 4.", "The square root of 16 is 5."),
        ("Mount Everest is the highest mountain on Earth.", "Mount Everest is in Asia."),
    ] * 8

    print("\nWarmup...")
    _ = judge.judge_claim(*pairs[0])

    print("\nBenchmarking (3 runs each)...")
    t_micro = run_bench(judge, pairs, batch_size=1, repeat=3)
    t_batch = run_bench(judge, pairs, batch_size=32, repeat=3)

    print(f"\nAverage over 3 runs:")
    print(f"  micro-batch (bs=1):  {t_micro:.3f}s  →  {(len(pairs)/t_micro):.2f} samples/s")
    print(f"  batched     (bs=32): {t_batch:.3f}s  →  {(len(pairs)/t_batch):.2f} samples/s")

    # Print results for each pair
    print("\nSample results:")
    results = judge.batch_judge_claims(pairs, batch_size=4)
    for (premise, hypothesis), result in zip(pairs, results):
        print(f"Premise:   {premise}")
        print(f"Hypothesis:{hypothesis}")
        print(f"Result:    {result.probs}, truth_score={result.truth_score:.4f}\n")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
