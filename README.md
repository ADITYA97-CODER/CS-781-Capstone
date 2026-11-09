# CS781 Capstone: In-House Legal Intelligence System

**Aditya Ojha, roll number 252330601, IIT Kanpur** · Codabench username `a_252330601`

A self-hosted legal-NLP system for Indian court judgments. A compact decoder-only language model (144.6M parameters) is
pretrained from scratch on the provided legal corpus and then adapted to six downstream tasks. The phase folders hold the
code of the best-scoring submission of each of the seven phases, with one training script and one inference script per phase.
`experiments/` holds the code of the other submissions, and every submission is listed under [All submissions by phase](#all-submissions-by-phase).

## Best submission per phase

| Phase | Task | Best submission | Metric | Official score |
|---|---|---|---|---|
| 1 | Legal LM pretraining | Pooled MBR ×16 (Arch E ×8 + masked-SFT Arch E ×8), steering, penalty-aware rerank | BERTScore F1 | 0.8474 |
| 2 | Legal NER | 6 LoRMA+ BIO taggers + 2 LoRMA+ span heads, combined per label | macro-F1 | 0.6938 |
| 3 | Rhetorical roles | 3-configuration wide-context ensemble + Dissent rule | macro-F1 | 0.5930 |
| 4 | Court judgment prediction | 7-model weighted ensemble (Arch E and F) | macro-F1 | 0.8424 |
| 5 | Statute identification | 5-model weighted ensemble with per-label thresholds | macro-F1 | 0.4635 |
| 6 | Abstractive summarization | Cross-system MBR over 6 summarizers, beam-4 decoding | BERTScore F1 | 0.8550 |
| 7 | Prior case retrieval | Arch F LoRMA+ bi-encoder fused with lexical and metadata features | Micro-F1@7 | 0.5162 |

## Repository layout

```
artifacts/tokenizer/legal_bpe_32k/   custom 32k BPE tokenizer
phase1/                              model, pretraining, corpus cleaning, masked SFT, decoding
phase2/ ... phase7/                  one folder per task
  models.sh                          the exact members and flags of the best submission (Phases 2-7)
  train.sh                           trains every member
  inference.sh                       builds the submission from trained weights
experiments/                         code of the other experiments and submissions
```

Phases share code through their folders: `phase1/models` is the backbone used everywhere, `phase2/peft.py` holds the
fine-tuning adapters, and `phase4/cjpe_data.py` and `cjpe_model.py` hold the document packing reused by Phases 5 and 7.
Keep the folders together. Run names in `models.sh` are the names used during the competition.

## Setup

Tested on Linux, Python 3.12, CUDA 12.4, NVIDIA RTX A6000 (48 GB).

```bash
conda create -n cs781 python=3.12 -y && conda activate cs781
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Download the official dataset into `data/` (the code expects `data/Phase_1_Pretraining`, `data/Phase_2_LNER`, and so on):

```bash
huggingface-cli download Exploration-Lab/CS-781-Capstone --repo-type dataset --local-dir data
```

Trained weights are not stored in this repository (each backbone checkpoint is 552 MB). The training scripts write them to
the paths the inference scripts read.

## Running

Every script can be started from any directory. `GPU` selects the device and `PYTHON` the interpreter (defaults: `0`, `python`).

```bash
GPU=0 bash phase1/train.sh            # first: produces the backbones used by every later phase
GPU=0 bash phase1/inference.sh

GPU=0 bash phase2/train.sh
GPU=0 bash phase2/inference.sh        # same pattern for phase3 ... phase7
```

Run Phase 1 first; Phases 2–7 only need its checkpoints and can then run in any order. Training scripts skip a member whose
weights already exist. Neither the dataset nor any submission file is stored in this repository. Each inference script creates a
`submission.zip` holding `prediction.json` at its root:

| Phase | Submission written to | Approximate training time (one RTX A6000) |
|---|---|---|
| 1 | `outputs/phase1/pooled16/` | Arch E 36 h, Arch F 18 h, masked SFT about 1.5 h, plus corpus preparation |
| 2 | `outputs/phase2/combo_bio6_span2/` | 8 members, 30–70 min each |
| 3 | `outputs/phase3/final/` | 3 members, 50–80 min each |
| 4 | `outputs/phase4/ens7_with8_w3/` | 7 members, 40–140 min each |
| 5 | `outputs/phase5/ens5_lwan2/` | 5 members, 2–5 h each |
| 6 | `outputs/phase6/mbr6_b4_test/` | 6 members, about 2 h each, plus about 70 min of beam-4 generation each |
| 7 | `outputs/phase7/f_lorma/` | 1 bi-encoder, about 100 min |

---

## Phase 1: legal language model

**Task.** Continue the start of an Indian legal document by exactly 128 words.

**Architecture (Arch E, `phase1/models/arch_e_diff.py`).** Decoder-only transformer with 20 layers, hidden size 768, SwiGLU
feed-forward (1,920), RMSNorm, rotary position embeddings, grouped-query attention (12 query heads, 4 key/value heads),
QK-normalization, and tied input/output embeddings: 144.6M parameters. Layers alternate between a 512-token sliding window and
global attention. All 20 layers use differential attention: two softmax maps are computed from split query/key groups, the
second is scaled by a learned per-layer λ and subtracted, and the result is renormalized per head. Attention is restricted to the
query's own document (FlexAttention block masks), position ids restart at each document, and boundary labels are masked.
Arch F (`arch_f_nodiff.py`) is the same model without the subtraction.

**Data and preprocessing (`phase1/corpus`, `phase1/tokenizer`).** The four sources are normalized, filtered and split; a 990-prompt
local evaluation set is held out. All text uses a custom 32k byte-pair-encoding tokenizer, where id 0 marks a document boundary.
Sources are mixed at Supreme Court 0.3, High Courts 0.4, Tribunals 0.2, Acts 0.1.

**Training (`phase1/train.py`, `configs/archE.json`).** 4B tokens in 8,138 steps of 491,520 tokens (16 micro-batches of 20 rows of
1,536 tokens), AdamW, peak learning rate 2e-3, weight decay 0.1 on linear weights, gradient clipping at 1.0, bf16.

**Masked task SFT (`phase1/sft`).** `clean_document.py` removes glued multi-record documents, typist initials and mis-encoded
text. Arch E is then fine-tuned for one epoch on 150k prompt→128-word pairs cut from the cleaned corpus, with 30% replay of plain
text. Target tokens of specifics (dates, citations, amounts, names, section numbers) that the prompt does not support are removed
from the loss, and supported specifics get weight 2. The result, `runs/sft_archE_v2_upweight`, is the backbone of later Arch E runs.

**Inference (`phase1/pooled_submission.py`).** Eight samples each from Arch E and the masked-SFT model (temperature 0.7, top-p 0.9,
repetition penalty 1.1, negative logit steering against spurious court headers and citation reporters). `mbr/mbr_rerank.py` picks the
candidate with the highest mean BERTScore against the other candidates, minus small penalties for signature blocks, closing
formulas and specifics absent from the prompt.

---

## Phase 2: legal named-entity recognition

**Task.** Character-exact spans of 12 entity types, scored by strict span macro-F1.

**Model (`ner_model.py`, `span_head.py`).** The backbone with a bidirectional mask reads windows of a 128-token header, a boundary
token and 1,024 content tokens. LoRMA+ (`peft.py`, rank 8, α 16) adapts 140 linear layers with the backbone frozen.

**Members (`models.sh`).** Four BIO taggers on document windows and two on FLERT-style units (a 256-token unit with 64 tokens of
context each side), plus two span-classification heads scoring every span up to 48 tokens, with negative spans re-weighted to 20×
the positive weight. All are trained for 20 epochs on folds 1 and 2.

**Decoding (`ner_data.py`, `make_combo.py`).** BIO probabilities are averaged and decoded with a constrained Viterbi; predicted
token spans are mapped to character offsets and snapped across letter–letter and digit–digit sub-word joins. A string predicted at
least twice with one label in a document is propagated to its other occurrences. The span head supplies RESP, STAT, PREC and
CASENO; the BIO ensemble supplies the other labels.

---

## Phase 3: rhetorical role prediction

**Task.** One of 13 roles per sentence, scored by macro-F1, on Corporate Law and Income Tax judgments trained jointly.

**Model (`rr_model.py`, `train_rr.py`).** Sentences are packed into windows of consecutive sentences. Each sentence's tokens are
mean-pooled and combined with a document-position bucket embedding before classification.

**Members (`models.sh`).** Arch E trained on all 90 labelled documents: LoRMA+ at 3,072-token windows, LoRMA+ at 2,048-token
windows, and full fine-tuning with an MLP head and logit-adjusted loss at 2,048-token windows.

**Inference.** `make_final.py` averages the probabilities and takes the argmax. `apply_dissent_rule.py` then finds a dissenting opinion
with regular expressions and sets its whole span to Dissent, matching how the gold data annotates dissents. Its first-person cue was
added after reading the unlabelled test text; it changes 91 sentences in one test document.

---

## Phase 4: court judgment prediction

**Task.** Binary outcome per case, scored by macro-F1.

**Model (`cjpe_model.py`, `cjpe_data.py`).** Documents are read through views: the last 4 or 8 chunks of 1,024 tokens, or the last
2,048 tokens. Chunks are packed into rows with FlexAttention segment masks, encoded bidirectionally, mean-pooled, and combined across
chunks by attention or mean pooling.

**Validation.** The provided dev split contains every test case, so all decisions use the 994 dev cases that do not overlap the test set.

**Members (`models.sh`).** Arch E LoRMA+ on 8 chunks (weight 3); Arch E LoRMA+, full fine-tuning and LoRA on 4 chunks; Arch F LoRMA+ on
4 chunks; Arch E and F LoRMA+ on the last 2,048 tokens (weight 1 each). `make_ensemble.py` averages class log-probabilities and tunes
the decision threshold on the clean dev cases, since the training labels are 59/41 and the test set is balanced.

---

## Phase 5: legal statute identification

**Task.** Multi-label prediction over 100 statutes, scored by macro-F1.

**Model (`lsi_model.py`, `train_lsi.py`).** The Phase-4 document encoder reads 8 chunks of 1,024 tokens spread evenly over the document.
Heads are attention pooling with 100 sigmoids, or a label-wise attention network (LWAN) with one attention query per statute.

**Members (`models.sh`).** Arch E LoRMA+ LWAN with binary cross-entropy (weight 2), Arch E LoRMA+ LWAN with asymmetric loss (weight 2), and
Arch E full fine-tuning, Arch F LoRMA+ and Arch E LoRMA+ with attention pooling (weight 1 each). `make_ensemble.py` averages sigmoid
probabilities and tunes one threshold per label on the dev split, reporting a cross-fitted estimate.

---

## Phase 6: abstractive summarization

**Task.** Write a headnote-style summary, scored by BERTScore F1.

**Model (`summ_model.py`, `summ_data.py`).** The decoder-only backbone is used as a continuation model:
`[document view] "Summary (about N words):" [summary]`, with the loss on summary tokens only. The view is the first 3,072 and last 1,024
tokens. At inference N comes from a log-linear fit of summary length on document length.

**Members (`models.sh`).** Arch E LoRMA+, Arch F LoRMA+, Arch E Houlsby adapters, Arch E full fine-tuning, Arch E LoRA, and a 10-layer
Arch E student distilled from the LoRMA+ model.

**Inference.** `gen_summ.py` generates with beam 4 and no-repeat 4-grams. `mbr_select.py` picks, per document, the candidate with the
highest mean BERTScore against the other five.

---

## Phase 7: prior case retrieval

**Task.** Return exactly 7 candidate ids per query, scored by Micro-F1@7. A query appears in its own pool and is always excluded.

**Dense model (`train_pcr.py`).** A shared-tower bi-encoder from the Arch F checkpoint with LoRMA+ encodes 8 chunks of 512 tokens spread
over each document. It is trained with InfoNCE (temperature 0.05) against 3 BM25-mined hard negatives per query and in-batch negatives,
using a two-pass cached-gradient step to fit long documents in memory.

**Lexical and metadata features (`lexical.py`, `lex_sweep.py`, `features.py`).** BM25 and TF-IDF over unigrams, bigrams and trigrams, using
the full query and its citation-context sentences; date consistency; and citation signals left in document headers and bodies.

**Fusion (`ltr.py`).** A first logistic regression is fit on the training queries over lexical and metadata features. A second, fit with
5-fold cross-validation over the dev queries, combines its logit with the dense score features, and the top 7 are written.
`inference.sh` runs this fitting step because it needs the dense embeddings.

---

## All submissions by phase

Scores are the official Codabench test scores. Names are the folders under `outputs/` or the run names under `runs_pN/`; the recorded
arguments of each run are in `experiments/run_commands/`. The best submission of each phase is in bold, and "–" means no score was recorded.

### Phase 1 (BERTScore F1)

| Submission | Method | Score |
|---|---|---|
| GPT-2 starter (three uploads) | Fine-tuned course starter checkpoint | 0.8003, 0.8006, 0.8015 |
| Llama, single sample | Fine-tuned pretrained Llama, early baseline | 0.8348 |
| Llama, MBR ×4 | Same model, MBR over 4 samples | 0.8394 |
| Arch D 2B, single sample | 72M sliding-window decoder, 2B tokens | 0.8303 |
| `mbr_docmask` | Arch D with document masking, MBR ×8 | 0.8420 |
| `fact_decoding/official/cad_a0.5` | Arch D + context-aware decoding, MBR ×8 | 0.8428 |
| `mbr_archE/official` | Arch E, MBR ×8 | 0.8450 |
| `archE_steer_official` | Arch E + header and citation steering + penalty-aware rerank, MBR ×8 | 0.8447 |
| `cad_official_EF/archE`, `archF` | Arch E and Arch F + context-aware decoding + loop filter + rerank, MBR ×8 | 0.8473 and 0.8472 (order not recorded) |
| **`masked_sft/official/pooled16`** | **Pooled MBR ×16: Arch E + masked-SFT Arch E, steering, rerank** | **0.8474** |

### Phase 2 (strict span macro-F1)

| Submission | Method | Score |
|---|---|---|
| `e_sftup_full_ens2_prop` | Full fine-tuning, 2 seeds, BIO, document-level propagation | 0.6249 |
| `e_sftup_lorma_s0` | LoRMA+, 1 seed, BIO | 0.6154 |
| `e_sftup_lorma_ens2` | LoRMA+, 2 seeds, BIO | 0.6546 |
| `e_sftup_lorma_ens4` | LoRMA+, 4 seeds, BIO | 0.6607 |
| `e_sftup_lorma_flert_ens2` | LoRMA+ with FLERT document-context windows, 2 seeds | 0.6534 |
| `bio_ens4_doc_flert` | LoRMA+ document windows ×2 + FLERT ×2 | 0.6671 |
| `e_sftup_lorma_kd_s0` | KD from the 4-model ensemble into one LoRMA+ student | 0.6391 |
| `e_sftup_lorma_kd_ens2` | KD students, 2 seeds | 0.6375 |
| `combo_lorma_bio_span` | Per-label combination: BIO ×2 + span head ×2 | 0.6730 |
| `combo_lorma_flert_span` | Per-label combination: FLERT BIO ×2 + span head ×2 | 0.6672 |
| `combo_bio4_span` | Per-label combination: BIO ×4 + span head ×2 | 0.6801 |
| **`combo_bio6_span2`** | **Per-label combination: BIO ×6 + span head ×2** | **0.6938** |

### Phase 3 (macro-F1 over 13 roles)

| Submission | Method | Score |
|---|---|---|
| `e_full_ctx` | Full fine-tuning, 1,024-token context, pooled linear head | 0.5380 |
| `e_full_laya` | Full fine-tuning, per-sentence marker tokens, MLP head, logit adjustment | 0.5172 |
| `e_full_laya_nomarker` | Full fine-tuning, pooled sentences, MLP head, logit adjustment | 0.5750 |
| `e_lorma_ctx` | LoRMA+, pooled linear head | 0.5555 |
| `e_lorma_mlp_la` | LoRMA+, MLP head, logit adjustment | 0.5286 |
| `laya_nm_all_ens2` | Marker-free configuration on all 90 documents, 2 seeds | 0.5264 |
| `laya_nm_all_ens4` | Same, 4 seeds | 0.5205 |
| `probe_ens4_plus_best_PO` | 4-seed ensemble with PrecedentOverruled labels from the 0.5750 file | 0.5301 |
| `cvhalves_wide3` | Wide-context 3-configuration ensemble, cross-validation halves | 0.5587 |
| `all_wide3` | Wide-context 3-configuration ensemble, all 90 documents | 0.5689 |
| `all_wide3_5m` | Same, 5 models | 0.5648 |
| `all_wide3_9m` | Same, 8 models | 0.5655 |
| **`lastday/cand_cvbest_dissentv2`** | **`all_wide3` + Dissent rule** | **0.5930** |

### Phase 4 (macro-F1)

| Submission | Method | Score |
|---|---|---|
| `e_lorma_hier4x1k` | Arch E LoRMA+, last 4 × 1,024 chunks | 0.8088 |
| `e_lorma_last2k` | Arch E LoRMA+, last 2,048 tokens | 0.8119 |
| `f_lorma_hier4x1k` | Arch F LoRMA+, 4 chunks | 0.8132 |
| `e_full_hier4x1k` | Arch E full fine-tuning, 4 chunks | 0.8199 |
| `f_full_hier4x1k` | Arch F full fine-tuning, 4 chunks | 0.8021 |
| `e_lorma_hier8x1k` | Arch E LoRMA+, 8 chunks | 0.8326 |
| `ens6_all` | 6-model equal-weight ensemble | 0.8398 |
| `ens7_with8` | 7-model equal-weight ensemble | 0.8404 |
| **`ens7_with8_w3`** | **7-model ensemble, 8-chunk model at weight 3** | **0.8424** |

### Phase 5 (macro-F1 over 100 statutes)

| Submission | Method | Score |
|---|---|---|
| `e_lorma_spread8` | Arch E LoRMA+, attention pooling | 0.4283 |
| `f_lorma_spread8` | Arch F LoRMA+, attention pooling | 0.4292 |
| `e_lwan_asl` | Arch E LoRMA+, label-wise attention, asymmetric loss | 0.4436 |
| `e_lorma_lwan` | Arch E LoRMA+, label-wise attention, binary cross-entropy | 0.4520 |
| **`ens5_lwan2`** | **5-model weighted ensemble with per-label thresholds** | **0.4635** |

### Phase 6 (BERTScore F1)

| Submission | Method | Score |
|---|---|---|
| `e_full_ht4k` | Arch E full fine-tuning, beam 4 | 0.8494 |
| `f_lorma_ht4k` | Arch F LoRMA+, beam 4 | 0.8515 |
| `e_lora_ht4k` | Arch E LoRA, beam 4 | 0.8525 |
| `e_lorma_ht4k` | Arch E LoRMA+, beam 4 | 0.8530 |
| `e_houlsby_ht4k` | Arch E Houlsby adapters, beam 4 | – |
| `kd_e10_from_lorma` | 10-layer KD student from Arch E LoRMA+, beam 4 | – |
| **`mbr6_b4_test`** | **Cross-system MBR over the 6 models above** | **0.8550** |

### Phase 7 (Micro-F1@7)

| Submission | Method | Score |
|---|---|---|
| `8_ArchE_KD_10layerStudent` | KD student bi-encoder + LTR | 0.5109 |
| `5_ArchE_FullFT` | Arch E full fine-tuning bi-encoder + LTR | 0.5115 |
| `7_ArchE_LoRMA` | Arch E LoRMA+ bi-encoder + LTR | 0.5127 |
| `0_Ensemble_EF_ctx_CE` | Arch E + F context-query bi-encoders + cross-encoder + LTR | 0.5127 |
| `3_ArchE_LoRMA_CrossEncoder` | Arch E LoRMA+ cross-encoder reranker + LTR | 0.5145 |
| `2_ArchE_LoRMA_ctxQuery` | Arch E LoRMA+, citation-context queries + LTR | 0.5162 |
| **`1_ArchF_LoRMA`** | **Arch F LoRMA+ bi-encoder + LTR** | **0.5162** |

---

## References

Method papers are cited in the project report. Dataset and task definitions: Joshi et al., *IL-TUR: Benchmark for Indian Legal Text
Understanding and Reasoning*, ACL 2024.
