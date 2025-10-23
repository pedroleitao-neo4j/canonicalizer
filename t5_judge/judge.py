#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import os, time
import logging
import torch
from textwrap import dedent
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

DEFAULT_MODEL = os.getenv("KG_JUDGE_MODEL", "google/flan-t5-xxl")
DEFAULT_LABELS = ("entailment", "contradiction", "neutral")

@dataclass(frozen=True)
class JudgeResult:
    labels: Tuple[str, str, str]
    probs: Dict[str, float]
    truth_score: float

@dataclass(frozen=True)
class EntityTypeResult:
    best_type: str
    confidence: float
    scores: Dict[str, float]

class T5Judge:
    """Deterministic label scorer using FLAN-T5, batched for GPU throughput."""

    def __init__(self,
                 model_name="google/flan-t5-xxl",
                 device: Optional[str] = None,
                 torch_dtype=None,
                 max_enc_len: Optional[int] = None,
                 prompt_overhead: int = 96,
                 evidence_ratio: float = 0.8,
                 logger: Optional[logging.Logger] = None,
                 quantize: bool = False,
                 qbits: int = 8):   # 8 or 4
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.logger = logger or logging.getLogger("t5_judge")

        # tokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            # T5 uses </s> as both EOS & PAD in many checkpoints
            self.tok.pad_token = self.tok.eos_token

        # model (quantized if requested & on CUDA)
        use_bnb = bool(quantize and device == "cuda")
        device_map = "auto" if use_bnb else None  # let HF place quantized weights

        if use_bnb:
            if qbits == 8:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            elif qbits == 4:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,  # good for RTX 3090
                )
            else:
                raise ValueError("qbits must be 4 or 8")

            # torch_dtype is optional for 8-bit; for 4-bit compute dtype set above
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch_dtype
            )
        else:
            # non-quantized path (CPU, MPS, or CUDA fp16/bf16)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype
            )
            self.model.to(self.device)

        self.model.eval()

        # Ensure generation config has sane IDs
        gc = self.model.generation_config
        if gc.pad_token_id is None:
            gc.pad_token_id = self.tok.pad_token_id
        if gc.eos_token_id is None:
            gc.eos_token_id = self.tok.eos_token_id

        # caches / knobs
        self._label_cache: Dict[Tuple[str, ...], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._amp_dtype = torch_dtype if (self.device == "cuda" and torch_dtype in (torch.float16, torch.bfloat16)) else None
        self._enc_max_len_cfg = max_enc_len
        self._prompt_overhead = int(max(0, prompt_overhead))
        self._evidence_ratio = float(min(max(evidence_ratio, 0.0), 1.0))

    # guardrail helpers
    def _enc_max_len(self) -> int:
        # Try model config first, then tokenizer, then fall back to 512.
        for k in ("n_positions", "max_position_embeddings"):
            v = getattr(self.model.config, k, None)
            if isinstance(v, int) and 0 < v < 1_000_000:
                return v
        ml = getattr(self.tok, "model_max_length", None)
        if isinstance(ml, int) and 0 < ml < 1_000_000:
            return ml
        return 512

    def _max_enc_len(self) -> int:
        return int(self._enc_max_len_cfg) if self._enc_max_len_cfg else self._enc_max_len()

    def _tok_len(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def _truncate_by_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return self.tok.decode(ids[:max_tokens], skip_special_tokens=True).strip()

    # internals
    def _build_prompt(self, evidence: str, fact: str, labels: Sequence[str]) -> str:
        # labels expected like ("entailed", "contradicted", "unknown")
        if len(labels) != 3:
            raise ValueError(f"Expected 3 labels, got {len(labels)}")

        # Token-budgeting: reserve overhead for instructions/labels, split remainder between evidence/fact.
        max_len = self._max_enc_len()
        budget = max(0, max_len - self._prompt_overhead)
        ev_budget = int(budget * self._evidence_ratio)
        fa_budget = budget - ev_budget

        ev_trunc = self._truncate_by_tokens(evidence, ev_budget)
        fa_trunc = self._truncate_by_tokens(fact, fa_budget)

        few_shot = dedent("""\
        premise: Paris is the capital of France.
        hypothesis: Paris is a city.
        Does the premise entail the hypothesis? Answer with one of: entailment, contradiction, neutral.
        answer: entailment
        
        premise: The Nile is the longest river in the world.
        hypothesis: The Nile is an organization.
        Does the premise entail the hypothesis? Answer with one of: entailment, contradiction, neutral.
        answer: contradiction
        
        premise: The shop was doing a promotion on Apple.
        hypothesis: Apple is a product.
        Does the premise entail the hypothesis? Answer with one of: entailment, contradiction, neutral.
        answer: neutral
        """)
        
        prompt = (
            few_shot.strip() +"\n\n"
            f"premise: {ev_trunc}\n"
            f"hypothesis: {fa_trunc}\n"
            f"Does the premise entail the hypothesis? Answer with one of: {labels[0]}, {labels[1]}, {labels[2]}.\n"
            "answer:"
        )

        self.logger.debug("Built prompt (len=%d tokens): %s", self._tok_len(prompt), prompt)

        return prompt

    def _prepare_label_tensors(self, labels: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        key = tuple(labels)
        if key in self._label_cache:
            dec_inputs, targets = self._label_cache[key]
            return dec_inputs.to(self.device), targets.to(self.device)

        label_tok_ids: List[List[int]] = [self.tok(l, add_special_tokens=False).input_ids for l in labels]
        start_id = self.model.config.decoder_start_token_id
        pad_id = self.tok.pad_token_id

        max_len = 0
        dec_seqs, tgt_seqs = [], []
        for y in label_tok_ids:
            dec = [start_id] + y[:-1]
            tgt = y[:]
            max_len = max(max_len, len(dec))
            dec_seqs.append(dec)
            tgt_seqs.append(tgt)

        dec_inputs, targets = [], []
        for dec, tgt in zip(dec_seqs, tgt_seqs):
            dec_padded = dec + [pad_id] * (max_len - len(dec))
            tgt_padded = tgt + [-100]   * (max_len - len(tgt))
            dec_inputs.append(dec_padded)
            targets.append(tgt_padded)

        di = torch.tensor(dec_inputs, dtype=torch.long)
        tg = torch.tensor(targets, dtype=torch.long)
        self._label_cache[key] = (di, tg)
        return di.to(self.device), tg.to(self.device)

    @torch.no_grad()
    def _label_probs_batched(self, prompts: Sequence[str], labels: Sequence[str]) -> torch.Tensor:
        B, L = len(prompts), len(labels)
        enc = self.tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_enc_len(),  # hard cap (2nd line of defense)
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        dec_inputs_L, targets_L = self._prepare_label_tensors(labels)
        Tlen = dec_inputs_L.size(1)

        def _rep(x): return x.repeat_interleave(L, dim=0)
        enc_rep = {k: _rep(v) for k, v in enc.items()}
        dec_inputs_rep = dec_inputs_L.unsqueeze(0).repeat(B, 1, 1).reshape(B * L, Tlen)
        targets_rep = targets_L.unsqueeze(0).repeat(B, 1, 1).reshape(B * L, Tlen)

        if self.device == "cuda" and self._amp_dtype is not None:
            ctx = torch.cuda.amp.autocast(dtype=self._amp_dtype)
        else:
            class _NoOp:
                def __enter__(self): pass
                def __exit__(self, *a): return False
            ctx = _NoOp()

        with ctx:
            logits = self.model(**enc_rep, decoder_input_ids=dec_inputs_rep).logits
            logprobs = torch.log_softmax(logits, dim=-1)

        gather_tgt = targets_rep.clone()
        gather_tgt[gather_tgt == -100] = 0
        tok_ll = logprobs.gather(-1, gather_tgt.unsqueeze(-1)).squeeze(-1)
        mask = targets_rep != -100
        seq_ll = (tok_ll * mask).sum(dim=-1)
        ll_matrix = seq_ll.view(B, L)
        return torch.softmax(ll_matrix, dim=1)

    # Public API
    def judge_claim(self, evidence, fact, labels=DEFAULT_LABELS):
        probs_mat = self._label_probs_batched([self._build_prompt(evidence=evidence, fact=fact, labels=labels)], labels)
        row = probs_mat[0].tolist()
        probs = {lbl: float(p) for lbl, p in zip(labels, row)}
        return JudgeResult(tuple(labels), probs, probs.get(labels[0], 0) - probs.get(labels[1], 0))

    def batch_judge_claims(self, pairs, labels=DEFAULT_LABELS, batch_size=32):
        results, batch = [], []
        def _flush(cur):
            if not cur: return
            prompts = [self._build_prompt(evidence=ev, fact=fa, labels=labels) for ev, fa in cur]
            probs_mat = self._label_probs_batched(prompts, labels)
            for row in probs_mat.detach().cpu().tolist():
                probs = {lbl: float(p) for lbl, p in zip(labels, row)}
                results.append(JudgeResult(tuple(labels), probs, probs.get(labels[0], 0) - probs.get(labels[1], 0)))
        for ev, fa in pairs:
            batch.append((ev, fa))
            if len(batch) >= batch_size:
                _flush(batch)
                batch = []
        _flush(batch)
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

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {DEFAULT_MODEL}")
    judge = T5Judge(model_name=DEFAULT_MODEL, torch_dtype=use_dtype)

    pairs = [
        ("OpenAI was founded in 2015 by a group including Elon Musk and Sam Altman.", "OpenAI was founded in 2015."),
        ("Apple introduced the iPhone in 2007 during a keynote by Steve Jobs.", "The iPhone launched in 2007."),
        ("The Nile is the longest river in the world, longer than the Amazon.", "The Amazon is the longest river."),
        ("Paris is the capital of France.", "Paris is the capital of Germany."),
        ("Microsoft acquired GitHub in 2018.", "GitHub was acquired by Microsoft."),
        ("The square root of 16 is 4.", "The square root of 16 is 5."),
        ("Mount Everest is the highest mountain on Earth.", "Mount Everest is in Asia."),
    ] * 8  # repeat to simulate ~50 examples

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