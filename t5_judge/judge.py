from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEFAULT_MODEL = "google/flan-t5-large"
DEFAULT_LABELS = ("entailed", "contradicted", "unknown")

@dataclass(frozen=True)
class JudgeResult:
    labels: Tuple[str, str, str]           # (entailed, contradicted, unknown) in the order you passed
    probs: Dict[str, float]                # normalized probs over labels
    truth_score: float                     # p(entailed) - p(contradicted)

@dataclass(frozen=True)
class EntityTypeResult:
    best_type: str
    confidence: float                      # p(entailed) for best_type
    scores: Dict[str, float]               # {type: p(entailed)}

class T5Judge:
    """
    A lightweight, deterministic 'statement judge' using FLAN-T5 (encoder-decoder).
    It scores short label strings by token log-likelihood, avoiding free-form generation.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model_name = model_name
        # Pick device between cpu, cuda or mps
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.model.to(self.device).eval()

        # cache tokenized labels across calls
        self._label_cache: Dict[Tuple[str, ...], List[List[int]]] = {}

    # Core helpers
    def _build_prompt(self, evidence: str, fact: str, labels: Sequence[str]) -> str:
        return (
            "Determine whether the FACT is entailed by the EVIDENCE. "
            f"Respond with one word: {labels[0]}, {labels[1]}, or {labels[2]}.\n\n"
            f"FACT: {fact}\n"
            f"EVIDENCE: {evidence}\n"
            "Answer:"
        )

    def _tokenize_labels(self, labels: Sequence[str]) -> List[List[int]]:
        key = tuple(labels)
        if key not in self._label_cache:
            self._label_cache[key] = [
                self.tok(l, add_special_tokens=False).input_ids for l in labels
            ]
        return self._label_cache[key]

    @torch.no_grad()
    def _label_probs(self, prompt: str, labels: Sequence[str]) -> Dict[str, float]:
        """Compute normalized P(label | prompt) by teacher-forced token log-likelihood."""
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=self.tok.model_max_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        label_ids_list = self._tokenize_labels(labels)

        start_id = self.model.config.decoder_start_token_id
        if start_id is None:
            raise RuntimeError("decoder_start_token_id is not set for this model.")

        logps = []
        for label_ids in label_ids_list:
            # decoder prefix starts with start token
            dec_inp = torch.tensor([[start_id]], device=self.device)
            total_logprob = 0.0
            for tok_id in label_ids:
                out = self.model(**enc, decoder_input_ids=dec_inp)
                next_logits = out.logits[:, -1, :]  # [1, vocab]
                logprob = torch.log_softmax(next_logits, dim=-1)[0, tok_id]
                total_logprob += float(logprob)
                dec_inp = torch.cat([dec_inp, torch.tensor([[tok_id]], device=self.device)], dim=1)
            logps.append(total_logprob)

        probs = torch.softmax(torch.tensor(logps), dim=0).tolist()
        return {lbl: float(p) for lbl, p in zip(labels, probs)}

    # Public APIs
    def judge_claim(
        self,
        evidence: str,
        fact: str,
        labels: Sequence[str] = DEFAULT_LABELS,
    ) -> JudgeResult:
        """
        Score a (evidence, fact) pair into entailed / contradicted / unknown.
        Returns probabilities and truth_score = p(entailed) - p(contradicted).
        """
        probs = self._label_probs(self._build_prompt(evidence, fact, labels), labels)
        p_ent = probs.get("entailed", 0.0)
        p_con = probs.get("contradicted", 0.0)
        return JudgeResult(labels=tuple(labels), probs=probs, truth_score=p_ent - p_con)

    def judge_entity_type(
        self,
        sentence: str,
        mention: str,
        candidate_types: Sequence[str],
        type_templates: Optional[Dict[str, str]] = None,
    ) -> EntityTypeResult:
        """
        For each candidate type, computes P(entailed) that 'mention is a <type>' is supported by `sentence`.
        Returns best type + per-type scores.
        """
        templates = type_templates or {
            "ORG": "is an organization.",
            "PERSON": "is a person.",
            "PRODUCT": "is a product.",
            "LOC": "is a location.",
            "GPE": "is a country or a city.",
            "EVENT": "is an event.",
            "WORK_OF_ART": "is a work of art.",
            "LAW": "is a law.",
        }
        scores: Dict[str, float] = {}
        for t in candidate_types:
            hyp = f"{mention} {templates.get(t, f'is a {t.lower()}.')}"
            res = self.judge_claim(sentence, hyp)
            scores[t] = res.probs.get("entailed", 0.0)

        best_type = max(scores, key=scores.get)
        return EntityTypeResult(best_type=best_type, confidence=scores[best_type], scores=scores)

    def batch_judge_claims(
        self,
        pairs: Iterable[Tuple[str, str]],
        labels: Sequence[str] = DEFAULT_LABELS,
        batch_size: int = 8,
    ) -> List[JudgeResult]:
        """
        Convenience: looped version for small batches. (Simple and readable;
        if you need real throughput, vectorize prompts and micro-batch.)
        """
        out: List[JudgeResult] = []
        batch: List[Tuple[str, str]] = []
        for ev, fa in pairs:
            batch.append((ev, fa))
            if len(batch) >= batch_size:
                out.extend(self._judge_claims_batch(batch, labels))
                batch.clear()
        if batch:
            out.extend(self._judge_claims_batch(batch, labels))
        return out

    def _judge_claims_batch(
        self, batch: List[Tuple[str, str]], labels: Sequence[str]
    ) -> List[JudgeResult]:
        return [self.judge_claim(e, f, labels) for (e, f) in batch]