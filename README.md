# Clean Knowledge Graphs with a Judge Model

This repository accompanies the presentation [“Clean Knowledge Graphs with a Judge Model”](https://docs.google.com/presentation/d/18vC1CbmhKj3WVNa4z-AZe633xf_VV1sZatxJ0nROEAc/edit?usp=sharing) and provides code and examples for entity–relationship (E-R) extraction, factual validation, and graph cleaning using LLMs and Natural Language Inference (NLI) models.

Producing high-quality knowledge graphs from unstructured text is challenging due to noisy LLM extractions. This repo presents an overall approach and implementation towards addressing this by integrating deterministic judge models for validation within the extraction pipeline.

## Background: The Precision–Recall Trade-off

Achieving **high recall** in Entity–Relationship (E-R) extraction is essential to capture as many entities and relationships as possible.  
However, this goal is constrained by the **precision–recall trade-off** — pursuing higher recall typically reduces precision, introducing incorrect or spurious facts.

A key challenge in information extraction is to balance **completeness** (recall) and **accuracy** (precision) so that the resulting knowledge graph is both **trustworthy** and **useful**.

| Recall | Precision | Outcome | Description |
|:-------|:-----------|:---------|:-------------|
| **High** | Low | *Noisy & Risky Graph* | Captures many facts but polluted with false entities or relationships — misleading for analytics or AI reasoning. |
| **Low** | High | *Accurate but Incomplete* | Trusted but sparse — safe for compliance or regulated domains, yet limited for discovery and insight generation. |
| **Balanced** | Balanced | *Trusted & Complete Graph* | Captures most relationships while maintaining accuracy; enables reliable downstream analytics and AI agents. |

**Goal:** Build graphs that are both *complete enough* for insight and *clean enough* for trust.

## 3️⃣ LLMs vs NLI Models

Large Language Models (LLMs) and Natural Language Inference (NLI) models serve very different purposes in knowledge graph construction.

LLMs are excellent for *discovery* — they can extract a wide range of entities and relationships from text.  
However, they are **generative** and **probabilistic**, meaning their outputs may include hallucinations or inconsistencies.  

NLI models, on the other hand, act as **deterministic validators**, evaluating whether a given extracted claim is actually supported by the source text.

| | **Large Language Models** | **Natural Language Inference (NLI) Models** |
|:-|:-|:-|
| **Type** | Generative and associative | Deterministic entailment classifier |
| **Mechanism** | Predicts the next most likely token to generate entities or relations | Evaluates if a *hypothesis* logically follows from a *premise* |
| **Strengths** | High recall, flexible, context-aware | High precision, factual grounding, reproducible |
| **Weaknesses** | Hallucination, memorization, non-determinism | Lower recall, less generative or exploratory |
| **Ideal Use Case** | Initial E–R extraction and discovery | Validation, scoring, and cleaning extracted facts |

> **Together**, they form a complementary system:  
> the LLM proposes *what might be true*, and the NLI model verifies *what is actually supported by the evidence*.

## Why a Judge Model

Decoder-only LLMs (e.g., GPT, LLaMA) are **not designed to score their own outputs**.  
They generate text token-by-token, but they don’t produce a single probability representing the truth of a statement.  
While some APIs expose token-level log probabilities (`logprobs`), these are often unavailable or impractical for hosted models.

To verify whether an extracted entity–relationship (E-R) claim is actually supported by evidence, we introduce a **Judge Model** —  
a Natural Language Inference (NLI) model that deterministically evaluates *entailment*.

**Example:**

> **Premise:** “Direct Edge spokesman Jim Gorman and BATS spokesman Randy Williams said in an email…”  
> **Hypothesis:** “Direct Edge is an organization.”  
> **Does the premise entail the hypothesis?** → **Neutral**

The NLI model doesn’t hallucinate or rely on prior training associations — it bases its answer **entirely on the given text**.

This approach enables:
- **Deterministic scoring** — each fact gets a reproducible entailment score (logit).  
- **Evidence grounding** — ensures facts are supported by the source text.  
- **Quality control** — filters out unsupported or ambiguous E-R pairs before graph ingestion.  
- **Balance** — complements high-recall LLM extraction with high-precision validation.

> *Think of the Judge Model as the factual referee for your Knowledge*

## 5️⃣ Models for NLI

Natural Language Inference (NLI) models are designed to evaluate whether a **hypothesis** logically follows from a **premise**.  
Unlike decoder-only LLMs, these models provide **deterministic, interpretable scores** (logits) representing *entailment*, *contradiction*, or *neutral* outcomes.

They typically use **encoder–decoder** or **cross-encoder** architectures that process both the premise and hypothesis jointly, building a bi-directional representation before producing a prediction.

### Key Characteristics

- **Deterministic entailment model** — evaluates support, contradiction, or uncertainty.  
- **True scoring** — returns logits (not generated probabilities).  
- **Grounded in evidence** — decisions are based only on provided text.  
- **Lightweight and efficient** — smaller models, quantizable, GPU-friendly.  
- **Easily fine-tuned** — often improved with just a few hundred domain examples.  
- **Complements LLMs** — adds factual verification and consistency checking.

### Common Pretrained NLI Models

| Model | Description |
|:--|:--|
| `google/flan-t5-small`, `flan-t5-base`, `flan-t5-large` | Encoder–decoder models fine-tuned for entailment tasks. |
| `microsoft/deberta-v3-large-mnli` | Cross-encoder with strong contextual understanding and precision. |
| `facebook/bart-large-mnli` | Sequence-to-sequence model with robust entailment performance. |
| `google/flan-ul2` | Larger model capable of mixed task generalization. |
| `google/long-t5-tglobal-*` | Extended context models handling >2048 tokens, ideal for long documents. |

All of these are available on [Hugging Face](https://huggingface.co/models) and can be fine-tuned for domain-specific ER validation tasks.

> ⚡ **Tip:** Pretrained NLI models can reach strong performance with minimal additional data — a few hundred labeled examples are often enough to adapt to your domain.

## Architecture Overview

The system integrates **Large Language Models (LLMs)** and **Natural Language Inference (NLI)** models into a multi-stage pipeline for building clean, grounded Knowledge Graphs.

### Pipeline Stages

```mermaid
flowchart LR
    A[Documents] --> B[Raw Graph]
    B --> C[Judged Graph]
    C --> D[Grounded Graph]
    C -->|Judges & Scores| E[ER Extraction Validation]
    D --> F[Wikipedia Grounding]
    E --> C
```

1. **Documents → Raw Graph**  
   Entities and relationships are first extracted from text using an LLM.  
   This stage maximizes recall, but may introduce noise, duplicates, or hallucinated facts.

2. **Raw Graph → Judged Graph**  
   Each extracted fact is evaluated by a **Judge Model** (an NLI model such as FLAN-T5 or DeBERTa-MNLI).  
   The model determines whether the fact is *entailed*, *neutral*, or *contradicted* by the evidence and assigns a **truth score**.  
   This creates a scored version of the graph where every node or relationship is annotated with factual confidence.

3. **Judged Graph → Grounded Graph**  
   Entities and relationships that pass validation are **disambiguated and linked** to external sources (e.g., Wikipedia).  
   This produces a **grounded, verifiable Knowledge Graph** ready for analytics, reasoning, or agent use.

### Key Advantages

- **Combines discovery and validation** — LLMs for recall, NLI models for precision.  
- **Evidence-based scoring** — every fact is traceable to its source text.  
- **Lightweight and modular** — works efficiently with quantized models on a single GPU.  
- **Extensible** — can support additional grounding sources beyond Wikipedia.  
- **Trusted output** — produces graphs suitable for downstream AI reasoning and compliance contexts.

## How to use this codebase

The code in this repository is intended to demonstrate the concepts presented in the "Clean Knowledge Graphs with a Judge Model" talk, providing a starting point for building your own knowledge graph extraction and validation pipelines. The main re-usable components are the `t5_judge` and `deberta_judge` modules, which implement the Judge Model functionality using FLAN-T5 and DeBERTa-MNLI, respectively. You can adapt these modules to fit your specific use case, fine-tune them on your domain data, and integrate them into your knowledge graph construction workflows.

Note that `t5_judge` is much more capable even without much fine-tuning, while `deberta_judge` is lighter and faster but will likely require more domain-specific training data to achieve similar performance.

The fastest way to understand and experiment with the pipeline, is just to run it end-to-end on the provided sample documents:

```bash
conda env create -f environment.yml
conda activate canonicalizer
python canonicalizer.py
```

Make sure you fill in all necessary variables in `.env` before running the code.

Note this code was developed and tested on Linux with an NVIDIA GPU. It should work as is on a Mac with an M1/M2 chip, but performance may vary.

Once you have run the pipeline, look at the `judge_stats.ipynb` notebook to see relevant stats and visualizations of the judged entities and relationships.