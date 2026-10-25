# Hallucination strategies for agent builders

**Objective:** Survey how the field detects and corrects LLM hallucinations, then organize methods into a **strategy menu**—combinable options chosen by **use case** and **resources**, not a single silver bullet.

**Status:** Living research note (2026-03). Toward a shared “master strategy set” for qlab and other agent builders.

---

## 1. What problem are we solving?

| Term | Meaning | Typical trigger |
|------|---------|-----------------|
| **Factuality** | Output vs **world** truth | Open QA, biographies, medical/legal claims |
| **Faithfulness** | Output vs **given context** | RAG, summarization, grounded chat |
| **Hallucination (loose)** | Unsupported, misleading, or invented content | Either of the above |

**Detection** — score or flag bad spans/answers before or after user sees them.  
**Correction** — revise text, re-decode, re-retrieve, abstain, or train the model to err less.

Many failures are **confident and fluent**; the menu below separates methods that need **external truth** from those that only check **internal consistency**.

**Survey anchors (background reading):**

- [ACM TOIS survey (2024)](https://dl.acm.org/doi/10.1145/3703155) — factuality vs faithfulness; fact-checking vs uncertainty.
- [Siren’s Song survey](https://arxiv.org/html/2309.01219v3) — benchmarks and task formats.
- [KG + LLM hallucination survey (NAACL 2024)](https://aclanthology.org/2024.naacl-long.219.pdf).
- [Comprehensive survey taxonomy (2025)](https://arxiv.org/html/2510.06265v2) — retrieval, uncertainty, embedding, learning, self-consistency.

---

## 2. Resource and constraint dimensions (how to navigate the menu)

Before picking strategies, fix these constraints. They determine what is **available** vs **theoretical**.

| Dimension | Low resource | High resource |
|-----------|--------------|---------------|
| **Model access** | Closed API (text only, no logprobs) | Open weights; logprobs; hidden states |
| **Knowledge** | No curated KB; maybe web search API | Corpus + vector DB; Wikidata/KG; enterprise docs |
| **Training** | Zero fine-tuning | NLI/detector FT; DPO/RLHF; Self-RAG-style tokens |
| **Latency / cost** | Single pass or +1 call | Many samples; retrieve per claim; full verify pipeline |
| **Risk** | Best-effort UX | Abstain or human review on uncertainty |

**Use-case tags** used in the menu:

- **OD** — open-domain factuality  
- **RAG** — must stick to retrieved passages  
- **AGENT** — tools, multi-step plans, memory  
- **LONG** — long-form generation (many atomic claims)  
- **HIGH** — high stakes (health, law, money)

---

## 3. Strategy menu (catalog)

Each row is one **option** in the master set. **D** = detect, **C** = correct.  
**FT** = fine-tuning (or train a separate model). **KB** = external knowledge. **Logits** = token/layer probabilities or activations used in the method.

### 3.1 Prompting and self-check (no FT, no KB)

| ID | Strategy | D/C | FT | KB | Logits | Best for | Weak when |
|----|----------|-----|----|----|--------|----------|-----------|
| **S1** | [Chain-of-Verification (CoVe)](https://arxiv.org/abs/2309.11495) | Both | No | No* | No | OD, LONG; reduce self-confirmation bias | Model wrong on isolated Q too; no ground truth |
| **S2** | [SelfCheckGPT](https://arxiv.org/abs/2303.08896) | D | No | No | No | OD; black-box API | Same lie in every sample; extra samples cost |
| **S3** | [Self-consistency / voting](https://arxiv.org/abs/2203.11171) | D/C | No | No | No | Reasoning-style QA | Systematic bias across samples |
| **S4** | [Self-Refine / critique loops](https://arxiv.org/abs/2303.17651) | C | No | No | No | Formatting, logic, draft quality | Critic shares generator blind spots |
| **S5** | [LLM-as-judge](https://arxiv.org/abs/2306.05685) | D | No | No | No | Fast filtering, eval pipelines | Judge hallucinates; correlated errors |

\*Extends to **S1+KB** when verification answers use retrieval (recommended for HIGH).

#### S1 — Chain-of-Verification (CoVe)

Large language models often produce plausible but incorrect factual statements, especially in long-form answers where errors compound across sentences. Chain-of-Verification treats the model as its own fact-checker: it drafts a response, plans targeted verification questions, answers those questions in a separate pass, and then revises the original answer from the checks. The key design choice is **isolated** verification—follow-up answers must not condition on the draft, so the model is less likely to repeat the same hallucination. Joint one-shot variants help, but factored step-by-step execution gives larger gains because attention to a bad draft otherwise reinforces errors. CoVe needs no fine-tuning or external corpus in its base form; it reduces hallucinations on list QA, multi-span QA, and long-form generation, though it still relies on parametric knowledge when verification is not retrieval-grounded.

#### S2 — SelfCheckGPT

SelfCheckGPT targets **black-box** generators that expose only text, not token probabilities or weights. The method samples multiple stochastic answers to the same prompt and treats inconsistency across samples as evidence of hallucination: when the model truly “knows” a fact, samples tend to agree, whereas fabricated details diverge. Sentence-level scores can be built with NLI, QA-over-samples, n-gram overlap, BERTScore, or a lightweight LLM “supported by context?” prompt over the sample bundle. It is zero-resource in the sense of requiring no external knowledge base, but it is not zero-cost—detection quality trades off against the number of extra generations. SelfCheckGPT works well as a filter or ranker; it fails when every sample repeats the same confident falsehood.

#### S3 — Self-consistency

Self-consistency improves chain-of-thought and other reasoning prompts by sampling **diverse** reasoning paths instead of taking a single greedy decode. The model generates multiple full responses (typically with temperature sampling); the final answer is chosen by majority vote over extracted conclusions, or the most consistent outcome across paths. The idea is that independent reasoning trajectories may err in different ways, while the correct answer often emerges as the mode. No fine-tuning, retrieval, or logit access is required—only multiple forward passes at inference time. It helps on multi-step reasoning benchmarks but does not guarantee factual truth when all paths share the same systematic bias or missing knowledge.

#### S4 — Self-Refine (critique loops)

Self-Refine is an iterative **generate–feedback–refine** loop in which the same LLM first produces a draft, then acts as critic to give natural-language feedback, then revises the output conditioned on that feedback. The process can repeat for several rounds until the critic signals satisfaction or a step limit is reached. It improves quality on tasks where errors are visible from the draft itself—code, constrained formats, logic puzzles, and some dialogue—without weight updates or external tools. The critic and generator share one model and one parametric worldview, so shared blind spots (including factual hallucinations) often survive refinement. It is best viewed as a general output polisher; pairing it with verification or retrieval is needed for ground-truth factuality.

#### S5 — LLM-as-judge

LLM-as-judge uses a strong language model to **score or rank** another model’s outputs against rubrics: helpfulness, harmlessness, correctness, or task-specific criteria. The approach scales evaluation and filtering pipelines where human labeling is expensive, and it powers benchmark suites that compare chat models via model-written critiques. Judges can run on the same API stack as the agent under test and need no fine-tuning if the evaluator model is already capable. Reliability is limited by **correlated errors**—the judge may hallucinate reasons, favor verbose or confident answers, or agree with incorrect but fluent text. For hallucination **detection**, judges work best as a fast screen combined with evidence-based checks (retrieval, atomic verification), not as the sole source of truth.

### 3.2 Uncertainty and decoding (no FT; KB optional)

| ID | Strategy | D/C | FT | KB | Logits | Best for | Weak when |
|----|----------|-----|----|----|--------|----------|-----------|
| **S6** | Token entropy / perplexity | D | No | No | Yes | Cheap triage on open models | Confident errors; creative truth |
| **S7** | [Semantic entropy](https://www.nature.com/articles/s41586-024-07421-0) | D | No | No | Often | Confabulation without KB | Multiple samples; clustering cost |
| **S8** | [Calibrated Entropy Score (CES)](https://arxiv.org/abs/2605.28264) | D | No | No | Yes | Single-pass detect with logit API | Needs logit access + calibration data |
| **S9** | [p(True) / boolean probing](https://arxiv.org/abs/2207.05221) | D | No | No | Yes | Short claim check | Long answers; miscalibration |
| **S10** | [DoLa (layer-contrast decode)](https://arxiv.org/abs/2309.03883) | C | No | No | Yes | OD on open weights | Cannot fix wrong pretraining “facts” |
| **S11** | [Contrastive decoding](https://arxiv.org/abs/2210.15097) | C | No | No | Yes | Reduce generic/absurd continuations | Setup varies by expert model |
| **S12** | [ITI (activation steering)](https://arxiv.org/abs/2306.03341) | C | Probe† | No | Activations | TruthfulQA-style on open weights | Tradeoff vs helpfulness; needs hooks |

†Small probe set on true/false; no full LM fine-tune.

#### S6 — Token entropy / perplexity

Uncertainty-based hallucination detection treats low model confidence as a warning sign: at each generated token, entropy or perplexity summarizes how peaked the next-token distribution is. When the model spreads probability mass over many continuations, the surface text is more likely to be under-specified or fabricated, so scalar aggregates over the sequence provide a cheap pre-screen. The method needs access to token-level probabilities (open weights or a logprob API) and adds negligible logic beyond forward passes already used for generation. It is widely used as a baseline because it requires no labels, retrieval, or fine-tuning. High-confidence hallucinations—flat wrong answers with a sharp distribution—are the main failure mode, as are fluent creative passages that are true but look “uncertain” by entropy alone.

#### S7 — Semantic entropy

Semantic entropy extends uncertainty estimation from **tokens** to **meanings** so that paraphrases of the same correct answer do not inflate apparent disagreement. The model samples multiple free-form answers; responses are clustered into equivalence classes (via bi-directional NLI or similar), and entropy is computed over semantic classes rather than raw strings. High semantic entropy flags **confabulations**—arbitrary incorrect generations—without requiring task-specific labels or a knowledge base at detection time. The approach generalizes across QA-style tasks when the model can be sampled repeatedly at inference. Cost grows with sample count and clustering, and factual errors that appear identically across samples still evade the signal.

#### S8 — Calibrated Entropy Score (CES)

Calibrated Entropy Score formalizes hallucination detection as a statistical test on the **distribution of per-token entropies**, not just their mean (as in length-normalized perplexity). A single generation pass yields an entropy trace; CES combines mean and tail behavior through a reference CDF so scores are comparable across models and sequence lengths. The method targets settings with black-box **logit** access but without multiple sampled answers, closing part of the gap to heavier semantic-entropy pipelines. Finite-sample guarantees motivate thresholding for deployable false-positive control. Performance still depends on calibration data and on logit availability; it does not by itself verify claims against external truth.

#### S9 — p(True) / boolean probing

This family asks the language model direct **yes/no** or **True/False** questions about a claim and reads confidence from the probability mass on affirmative versus negative tokens. Kadavath et al. show that larger models can be surprisingly well-calibrated on such probes when elicited in short form, enabling reference-free factuality checks from parametric knowledge alone. FActScore and related evaluators use the same idea to score atomic statements after decomposition. The signal is strongest for isolated factual probes and weak for long, entangled generations where claims interact. Miscalibration and sycophancy still produce confident “True” on false statements.

#### S10 — DoLa (decoding by contrasting layers)

DoLa reduces factual hallucinations at **decode time** by contrasting logits from later transformer layers against earlier ones, amplifying knowledge thought to reside in deeper layers. No retrieval, tool use, or weight update is required—only the ability to read intermediate layer projections during autoregressive generation. The method improves truthfulness on multiple-choice and open-ended benchmarks for open-weight LMs (e.g., TruthfulQA gains for LLaMA-family models). It cannot correct facts the model never learned or learned wrongly during pretraining, since it reweights rather than retrieves. Layer-selection heuristics and model architecture affect gains, so transfer requires per-family tuning.

#### S11 — Contrastive decoding

Contrastive decoding modifies the next-token distribution by contrasting a **strong expert** LM against a weaker or partially masked **amateur** model, down-weighting tokens the amateur favors but the expert does not. The intent is to suppress generic, repetitive, or implausible continuations that survive standard sampling. Like DoLa, it is a generation-time intervention that avoids fine-tuning and external KB when the contrast pair is available. Expert–amateur pairing and masking schemes vary by implementation, which complicates a one-size-fits-all agent recipe. Factual hallucinations that both models endorse are largely untouched.

#### S12 — Inference-Time Intervention (ITI)

Inference-Time Intervention identifies attention heads whose activations linearly separate true from false statements on a small probe set, then **shifts** those activations along a “truthful” direction during every decoding step. The LM weights stay frozen; only a handful of labeled examples are needed to find heads and directions, unlike RLHF-scale preference training. ITI substantially raises TruthfulQA accuracy on instruction-tuned LLaMA variants but exposes a tradeoff between truthfulness and helpfulness controlled by intervention strength. It requires **open-weight** access to internal states and is tied to benchmarks used to find directions. Out-of-distribution tasks may see smaller gains than in-distribution truthfulness tests.

**Menu rule:** If you only have **closed chat API (S5-era)**, prefer **S1, S2, S5** and **abstention**; skip **S6–S12** unless the provider exposes logprobs or you run a local verifier model.

### 3.3 Retrieval and evidence (KB; FT optional)

| ID | Strategy | D/C | FT | KB | Logits | Best for | Weak when |
|----|----------|-----|----|----|--------|----------|-----------|
| **S13** | [RAG](https://arxiv.org/abs/2005.11401) | C | No | Yes | No | OD, RAG, AGENT | Ignores or contradicts context |
| **S14** | [FActScore pipeline](https://arxiv.org/abs/2305.14251) | D | No | Yes‡ | Optional | LONG, HIGH auditing | Decomposition + verify cost |
| **S15** | [RARR](https://arxiv.org/abs/2210.08726) | Both | No | Yes | Optional | Attribution + post-edit | Agreement module errors |
| **S16** | [CRAG](https://arxiv.org/abs/2401.15884) | C | Eval§ | Yes | No | RAG when corpus is noisy | Bad evaluator routing |
| **S17** | [Self-RAG](https://arxiv.org/abs/2310.11511) | Both | Yes | Yes | Tokens | AGENT, RAG, citation-aware | Training cost; retrieval infra |
| **S18** | KG-augmented RAG / verify | D/C | Varies | Yes (KG) | No | Structured facts, multi-hop | KG coverage gaps |
| **S19** | Tool / search loops ([ReAct](https://arxiv.org/abs/2210.03629), [Toolformer-style](https://arxiv.org/abs/2302.04761)) | Both | No | Yes | No | AGENT, verifiable steps | Tool misuse; stale search |

‡No-context and self-check variants exist without KB.  
§Lightweight retrieval evaluator in CRAG; can swap models.

**KG survey:** [Can KGs reduce hallucinations?](https://aclanthology.org/2024.naacl-long.219.pdf)

#### S13 — Retrieval-Augmented Generation (RAG)

Retrieval-augmented generation conditions the LM on passages retrieved from a corpus so answers are grounded in external text rather than parametric memory alone. A retriever ranks chunks for the query; the generator reads them in context and produces the response, which cuts many open-domain factual gaps that pure LMs cannot fill. RAG is the default **correction-at-source** strategy for enterprise agents with document stores and is composable with almost every other menu item. It does not enforce faithfulness: models may ignore, misquote, or contradict retrieved passages, especially under long contexts or weak retrievals. Quality hinges on corpus coverage, chunking, and retrieval precision as much as on the generator.

#### S14 — FActScore pipeline

FActScore evaluates and can drive correction of **long-form** text by decomposing generations into atomic factual statements and scoring each against trusted sources (e.g., Wikipedia). The metric reports the fraction of atoms supported by evidence, enabling fine-grained comparison of models and prompts on biography-style tasks. Automated variants prompt an evaluator LM with retrieved passages or use self-consistency checks when KB access is limited. The pipeline is ideal for auditing and for HIGH-stakes review workflows where sentence-level truth matters. Cost scales with the number of atoms and retrieval calls per atom, and verifier LM errors propagate into scores.

#### S15 — RARR (research and revise)

RARR retrofits **attribution** onto any generator post hoc: it extracts check-worthy spans, issues search queries, retrieves web evidence, tests agreement, and edits the text to align with sources while preserving style. The design decouples attribution from the base LM so strong proprietary models can be used unchanged. It improves supportedness and citation quality on diverse generation tasks with minimal task-specific training. The agreement and edit stages can themselves err—especially when evidence is ambiguous or retrieval is off-topic—so end-to-end factuality is not guaranteed. RARR is best understood as research-then-revise correction with explicit external evidence, not pure detection.

#### S16 — CRAG (corrective RAG)

Corrective Retrieval-Augmented Generation assumes naïve RAG fails when retrieved documents are irrelevant or incomplete. A lightweight **retrieval evaluator** scores document sets for each query and triggers one of three actions: use refined internal knowledge strips when retrieval is good, invoke **web search** when it is bad, or blend both when confidence is ambiguous. A decompose-then-recompose step filters noise from retrieved text before generation. CRAG plugs into existing RAG stacks without retraining the main LM and improves robustness across short- and long-form tasks in the paper benchmarks. Routing mistakes—trusting bad corpora or bad web hits—still produce confident hallucinations, so the evaluator quality is the critical dependency.

#### S17 — Self-RAG

Self-RAG trains a single language model to emit **reflection tokens** that control retrieval (whether to retrieve), critique passage relevance, and judge whether generations are supported and useful. At inference the model adaptively retrieves, generates candidate continuations, and selects outputs that pass self-critique rather than always prepending a fixed number of documents. End-to-end training aligns generation with retrieval and reduces ungrounded text relative to retrieve-then-generate baselines on QA, reasoning, and long-form tasks. It requires curated training data for reflection tokens, retrieval infrastructure, and fine-tuning compute. The framework is powerful for citation-aware agents but heavier to ship than prompt-only RAG or CoVe-style post-hoc verify.

#### S18 — KG-augmented RAG and verification

Knowledge-graph augmentation connects LLM pipelines to structured triples and entity relations so verification can use **graph match**, path finding, and schema constraints—not only vector similarity over chunks. Surveys categorize methods by when graphs are built (index time vs. query time), how LLMs extract or align to ontology, and how conflicts are resolved against global memory. The approach shines on multi-hop factual questions and domains with curated KGs (biomedical, enterprise master data). Coverage gaps in the KG become coverage gaps in answers; extraction errors poison the graph. qlab’s graph-memory harnesses (HippoRAG, MemGraphRAG) sit in this design space as retrieval architectures rather than as standalone hallucination classifiers.

#### S19 — Tool and search loops (ReAct, Toolformer-style)

Tool-augmented agents interleave **reasoning traces** with actions—search, APIs, calculators, code execution—so claims can be checked against live or structured external state. ReAct prompts the model to emit thought–action–observation steps until a final answer; Toolformer-style training (where used) teaches when to call tools from self-supervised traces. This is the natural strategy for AGENT builders who can expose verifiable tools and accept multi-step latency. Failures include wrong tool choice, misread observations, stale search snippets, and hallucinated tool arguments that never run. Tools supply evidence but do not remove the need for faithfulness checks on how observations are summarized into natural language.

### 3.4 Learned detectors and training (FT)

| ID | Strategy | D/C | FT | KB | Logits | Best for | Weak when |
|----|----------|-----|----|----|--------|----------|-----------|
| **S20** | [HaluEval-style NLI detector](https://arxiv.org/abs/2305.11747) | D | Yes | Context | No | RAG faithfulness | Domain shift; needs `(context, answer)` |
| **S21** | [RAGTruth / span detectors](https://arxiv.org/abs/2310.01558) | D | Yes | Context | No | Pinpoint unsupported spans | Annotation / domain match |
| **S22** | [SentHalu (lightweight sentence screen)](https://doi.org/10.5281/zenodo.20555079) | D | Yes | Context | No | RAG pre-flight filter | False positives |
| **S23** | [FactTune / DPO for factuality](https://arxiv.org/abs/2311.08401) | C | Yes | Optional | Optional | Reduce error rate at source | Preference signal quality |
| **S24** | [RLHF / instruction tuning](https://arxiv.org/abs/2203.02155) | C | Yes | No | No | General alignment | Not truth-aligned by default |
| **S25** | Truthfulness-oriented training ([TruthfulQA line](https://arxiv.org/abs/2109.07958)) | C | Yes | No | No | Adversarial factuality | Narrow benchmark vs production |

#### S20 — HaluEval-style NLI detector

HaluEval provides a benchmark and methodology for detecting hallucinations by casting the task as **natural language inference**: given knowledge (or context), a query, and a model answer, a classifier predicts whether the answer is faithful or hallucinated. GPT-4–generated adversarial examples cover QA, dialogue, and summarization settings with both generic and task-specific detectors. Fine-tuned encoders (e.g., DeBERTa-family) reach strong in-domain accuracy when context is present at inference time. The approach is a workhorse for **RAG faithfulness** gates before user delivery. Performance drops under domain shift and when “context” is missing or thin, because the model may learn surface cues rather than deep entailment.

#### S21 — RAGTruth / span-level detectors

RAGTruth focuses on **retrieval-augmented** settings where hallucinations are often subtle insertions not supported by provided passages. The corpus labels unsupported spans in model outputs conditioned on retrieved context, enabling training of detectors that localize errors finer than response-level scores. Token- or span-level models help agents redact, regenerate, or cite only vetted segments instead of rejecting whole answers. Annotations and training target the RAG failure mode where the answer looks fluent and on-topic but adds unstated facts. Generalization requires similar retrieval noise and domain at deploy time; open-domain factuality without context is out of scope.

#### S22 — SentHalu

SentHalu is a **lightweight sentence-level** screener for RAG pipelines: each generated sentence is scored against retrieved context with a compact NLI-initialized cross-encoder fine-tuned on RAGTruth and HaluEval-style data. The goal is high **recall** on likely hallucinations under modest GPU budgets so production systems can filter before display or trigger rewrite. Cross-domain evaluation (medical, dialogue, TruthfulQA) is reported without target-domain fine-tuning to stress portability. False positives remain a practical burden—safe sentences flagged as risky cause unnecessary abstention or rework. It fits as a fast middleware layer between generator and user, not as a sole ground-truth oracle.

#### S23 — FactTune / DPO for factuality

FactTune reduces factual errors by **direct preference optimization** on pairs of candidate answers ranked by automatic truthfulness scores—either retrieval-backed (FactTune-FS) or model-confidence-based (FactTune-MC) without a KB at scoring time. Multiple samples per prompt are compared; the more factual completion becomes the preferred target in DPO, avoiding hand-labeled human preferences at scale. Experiments on biographies and medical QA show large reductions in incorrect atomic claims versus RLHF chat models and decoding-only baselines on held-out topics. Quality ceilings depend on the scorer: bad retrieval or miscalibrated confidence poisons preferences. Best paired with retrieval at inference so the tuned model is not asked to invent missing evidence.

#### S24 — RLHF / instruction tuning

Reinforcement learning from human feedback and instruction fine-tuning align LMs to human preferences—helpfulness, harmlessness, and follow instructions—rather than to verifiable truth per se. RLHF is the default path for commercial chat models and improves usability, refusals, and format compliance across tasks. It can reduce some blatant errors but surveys and factuality work show it does **not** reliably eliminate hallucination and may increase fluent confabulation when rewards favor engagement. For agent builders, RLHF-base models are the starting generator, not a complete hallucination strategy. Truth-specific objectives (FactTune, Self-RAG, retrieval) remain necessary when factual precision is contractual.

#### S25 — Truthfulness-oriented training (TruthfulQA line)

TruthfulQA introduces adversarial questions designed to elicit **imitative falsehoods**—answers that match common misconceptions or popular text rather than fact. The benchmark and follow-on work study training and decoding interventions that increase the rate of truthful, informative answers on such probes. Methods include supervised finetuning on truthful responses, benchmark-driven evaluation, and combinations with ITI or other inference tricks. Gains on TruthfulQA do not automatically transfer to all production domains or long-form generation. The line of work is most relevant when models must resist myth-like prompts (HIGH-stakes consumer advice, science communication) and when you can measure success with similar adversarial QA suites.

**Menu rule:** **S20–S22** shine when you **always have context** (RAG). **S23–S25** change the generator; pair with **S13–S19** so training does not fight missing evidence.

---

## 4. Quick index (all strategies)

| ID | Name | Link |
|----|------|------|
| S1 | CoVe | [2309.11495](https://arxiv.org/abs/2309.11495) |
| S2 | SelfCheckGPT | [2303.08896](https://arxiv.org/abs/2303.08896) |
| S3 | Self-consistency | [2203.11171](https://arxiv.org/abs/2203.11171) |
| S4 | Self-Refine | [2303.17651](https://arxiv.org/abs/2303.17651) |
| S5 | LLM-as-judge | [2306.05685](https://arxiv.org/abs/2306.05685) |
| S6 | Token entropy | See [Huang et al. 2025 survey](https://arxiv.org/html/2601.09929v1) |
| S7 | Semantic entropy | [Nature 2024](https://www.nature.com/articles/s41586-024-07421-0) |
| S8 | CES | [2605.28264](https://arxiv.org/abs/2605.28264) |
| S9 | p(True) | [2207.05221](https://arxiv.org/abs/2207.05221) |
| S10 | DoLa | [2309.03883](https://arxiv.org/abs/2309.03883) |
| S11 | Contrastive decoding | [2210.15097](https://arxiv.org/abs/2210.15097) |
| S12 | ITI | [2306.03341](https://arxiv.org/abs/2306.03341) |
| S13 | RAG | [2005.11401](https://arxiv.org/abs/2005.11401) |
| S14 | FActScore | [2305.14251](https://arxiv.org/abs/2305.14251) |
| S15 | RARR | [2210.08726](https://arxiv.org/abs/2210.08726) |
| S16 | CRAG | [2401.15884](https://arxiv.org/abs/2401.15884) |
| S17 | Self-RAG | [2310.11511](https://arxiv.org/abs/2310.11511) |
| S18 | KG augmentation | [NAACL 2024 survey](https://aclanthology.org/2024.naacl-long.219.pdf) |
| S19 | Tool / search agents | [ReAct](https://arxiv.org/abs/2210.03629) |
| S20 | HaluEval detector | [2305.11747](https://arxiv.org/abs/2305.11747) |
| S21 | RAGTruth | [2310.01558](https://arxiv.org/abs/2310.01558) |
| S22 | SentHalu | [Zenodo](https://doi.org/10.5281/zenodo.20555079) |
| S23 | FactTune | [2311.08401](https://arxiv.org/abs/2311.08401) |
| S24 | RLHF | [2203.02155](https://arxiv.org/abs/2203.02155) |
| S25 | TruthfulQA / truthfulness training | [2109.07958](https://arxiv.org/abs/2109.07958) |

*Last updated: 2026-03-22.*
