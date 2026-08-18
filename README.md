
MANAL FATIMA
BIA DATA SCIENCE 



# Banking Intent Router — knowing what you don't know

Fine-grained customer-support intent classification with an explicit **abstain**
mechanism, comparing three text representations on the same task.

The question this project asks is not "which model classifies best" but:
**what does an intent router do when the query isn't one of the intents it knows?**

Live demo: a Streamlit app that runs all three models side by side and shows the
abstain decision, confidence, and per-query latency.

---

## Key findings

1. **Fine-tuning a transformer bought nothing.** Frozen MiniLM embeddings with a
   logistic regression (macro-F1 **0.9212**) matched a fine-tuned DistilBERT
   (**0.9209**) — a 0.03-point difference against a 0.75-point standard error.
   What a model was *pretrained for* mattered more than whether it was fine-tuned.
2. **Rejecting near-domain queries is far harder than rejecting off-topic ones.**
   At 95% target coverage, MiniLM rejects 96.9% of non-banking queries but only
   40.7% of held-out banking intents.
3. **Under input corruption, sparse features fail silently and dense features fail
   safe.** At 20% character corruption, TF-IDF's confidently-wrong rate rises to
   0.335 while MiniLM's *falls* to 0.040 — from near-identical raw accuracy.
4. **Regularised linear models were under-confident, not over-confident.**
   Temperature scaling fitted T ≈ 0.57 for both linear tracks (ECE 0.158 → 0.017),
   while DistilBERT was already calibrated (T = 0.95, ECE 0.021) because its
   checkpoint was selected at minimum validation loss.
5. **A cheap-first cascade halves serving latency for half a point of accuracy.**

---

## Data

| Source | Role | Size |
|---|---|---|
| BANKING77 (Casanueva et al., 2020) | in-scope intents | 13,083 queries, 77 intents |
| CLINC150 `plus` OOS split (Larson et al., 2019) | out-of-domain probes | 1,000 queries |

BANKING77 is real online-banking customer-service data released by PolyAI.
Seven intents were removed from training entirely and reused as **in-domain
unknown** queries, following the ID-OOS methodology in the intent-detection
literature. Each held-out intent was chosen to have a semantic neighbour that
remains in training, so the model must reject queries that look like things it knows.

Held out: `card_swallowed`, `contactless_not_working`, `virtual_card_not_working`,
`card_acceptance`, `top_up_limits`, `receiving_money`, `lost_or_stolen_card`.

### Partitions

| Split | Rows | Purpose |
|---|---|---|
| `train` | 8,103 | fit models (70 intents) |
| `val` | 1,430 | model selection + abstain threshold |
| `test_in` | 2,800 | in-scope test (40 per intent, balanced) |
| `test_idoos` | 280 | in-domain unknown (held-out intents) |
| `test_oodoos` | 1,000 | out-of-domain unknown (CLINC) |

### EDA findings that drove design decisions

- Training imbalance ratio **5.34** (35–187 per intent) → macro-F1 as headline metric.
- Token length p99 = 53 → `max_length = 64` (99.70% of queries uncut).
- Train/test text overlap: **6 of 3,079** unique test queries (0.19%) — measured,
  reported, not removed (removing them would break comparability with published results).
- CLINC OOS queries are systematically shorter (mean 8.7 words vs 11.9) — a length
  confound that the in-domain unknown set is not subject to.

---

## Method

Three representations, one classifier family, one evaluation protocol.

| Track | Representation | Dimensionality |
|---|---|---|
| A | TF-IDF, 1–2 grams, `min_df=2`, `sublinear_tf` | 9,055 sparse |
| B | Frozen `all-MiniLM-L6-v2` sentence embeddings | 384 dense |
| C | Fine-tuned `distilbert-base-uncased` | — |

**Model selection used the one-standard-error rule**, not `argmax` on validation.
With 1,430 validation rows the standard error is ~0.0074, and six Track B candidates
were statistically indistinguishable. Naive argmax selected a model that scored
0.85 points *worse* on test than the one-SE choice — the selection rule was fixed
before test data was touched.

**Abstain mechanism.** Rejection uses the energy score, −(−logsumexp(logits)), fixed
a priori on the basis of published OOD work rather than chosen after seeing results.
The threshold is the 5th percentile of validation energy, i.e. a business decision
to accept rejecting 5% of legitimate queries. The out-of-scope sets are used exactly
once, to measure what that choice bought.

**Training config (Track C):** 10 epochs, lr 3e-5, batch 32, AdamW, linear schedule
with 10% warmup, gradient clipping at 1.0, mixed precision, Colab T4.
Checkpoint selected at **minimum validation loss** (epoch 6), not maximum macro-F1
(epoch 11) — validation loss is negative log-likelihood, a proper scoring rule,
so it selects the calibration-optimal checkpoint.

---

## Results

### In-scope classification (test, 2,800 queries, 70 intents)

| Track | Macro-F1 | Accuracy | Fit time |
|---|---|---|---|
| A — TF-IDF | 0.8870 | 0.8871 | 4.5 s |
| B — Frozen MiniLM | **0.9212** | 0.9214 | 1.2 s |
| C — Fine-tuned DistilBERT | 0.9209 | 0.9207 | ~3 min (T4) |

Macro-F1 and accuracy coincide because the test split is balanced at 40 per intent;
the imbalance is in the training distribution, not the measurement.

### Out-of-scope rejection (energy score, 95% target coverage)

| Track | Coverage | Rejected ID-OOS | Rejected OOD | AUROC ID-OOS | AUROC OOD | Acc. on accepted |
|---|---|---|---|---|---|---|
| A | 0.957 | 0.271 | 0.831 | 0.828 | 0.968 | 0.903 |
| B | 0.960 | 0.407 | **0.969** | 0.901 | **0.992** | 0.935 |
| C | 0.964 | **0.471** | 0.944 | **0.913** | 0.989 | 0.937 |

Accuracy on accepted queries exceeds accuracy on all queries for every track —
the confidence score carries real information about correctness.

No single score wins both regimes: energy dominates far-OOD, negative entropy is
better on near-OOD (0.924 vs 0.901 for Track B). Energy was fixed in advance and
was not switched after seeing this.

### Robustness to character corruption (600 test queries)

Fraction of queries **answered and wrong**:

| Corruption | A — TF-IDF | B — MiniLM | C — DistilBERT |
|---|---|---|---|
| 0% | 0.063 | 0.055 | 0.053 |
| 5% | 0.152 | 0.078 | 0.080 |
| 10% | 0.190 | 0.052 | 0.083 |
| 20% | **0.335** | **0.040** | 0.060 |

At 20% corruption both models lose comparable raw accuracy (0.427 vs 0.472). The
difference is that MiniLM sheds coverage (0.96 → 0.19) as it loses accuracy, while
TF-IDF keeps answering (0.96 → 0.67). Character corruption is the setting most
hostile to a lexical model by construction — an out-of-vocabulary token is invisible
to TF-IDF, while subword tokenisation degrades gradually.

### Calibration

| Track | T | ECE before | ECE after | Accuracy (unchanged) |
|---|---|---|---|---|
| A | 0.567 | 0.1577 | 0.0171 | 0.8871 |
| B | 0.575 | 0.1450 | 0.0117 | 0.9214 |
| C | 0.950 | 0.0212 | 0.0166 | 0.9207 |

T < 1 means the linear tracks were **under**-confident — a direct consequence of the
L2 regularisation selected in Phase 2, which shrinks logit magnitudes and flattens
the softmax. Temperature scaling left accuracy identical to four decimals but slightly
*reduced* max-softmax OOD AUROC (0.977 → 0.957 for Track B): calibration and
separability are different objectives.

### Cascade router

TF-IDF answers when its energy clears a validation-percentile threshold; otherwise
the query escalates to MiniLM. Single-query latency measured on an Intel i7-7820HQ.

| Policy | Accuracy | Mean latency | Rejected OOD |
|---|---|---|---|
| TF-IDF only | 0.8871 | 2.0 ms | 0.127 |
| **Cascade, escalate 50%** | **0.9164** | **24.8 ms** | **0.962** |
| MiniLM only | 0.9214 | 50.0 ms | 0.969 |

Escalating only the least-confident 22% of queries already recovers 69% of the
accuracy gap — nearly everything TF-IDF gets wrong, it is already unconfident about.

---

## Repository layout

```
src/
  config.py        paths, seed, split fractions
  data.py          BANKING77 + CLINC loaders (requests-based, cached to data/raw)
  splits.py        held-out intent selection, stratified split, label remapping
  selection.py     one-standard-error model selection rule
  track_a.py       TF-IDF pipeline + grid
  track_b.py       sentence embeddings (cached .npy) + classifier grid
  oos.py           five confidence scores, thresholds, AUROC
  calibration.py   ECE, reliability bins, temperature scaling
  robustness.py    character corruption experiment
  cascade.py       two-stage router + latency model
notebooks/         01_eda … 07_cascade, one per phase
app/
  streamlit_app.py three-track live demo
reports/figures/   all figures used in the report
```

Track C is trained in Colab (see `notebooks/` for the exported notebook); the
fine-tuned weights and per-split logits are downloaded into `models/track_c/`.

`data/` and `models/` are git-ignored — everything in them is regenerable from code.

---

## Reproducing

Requires Python 3.11. Pinned to the torch 2.2.2 generation because the development
machine is an Intel Mac, for which PyTorch publishes no wheels beyond 2.2.2.

```bash
git clone https://github.com/MANAL-inv/banking-intent-router.git
cd banking-intent-router

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the notebooks in order, or:

```python
from src.splits import build_splits
from src import track_a, track_b, oos, calibration, robustness, cascade

build_splits()
track_a.run()
track_b.run()      # downloads all-MiniLM-L6-v2 on first run
oos.run()          # requires models/track_c/logits_*.npy from Colab
calibration.run()
robustness.run()
cascade.run()
```

Launch the app:

```bash
streamlit run app/streamlit_app.py
```

---

## Limitations

- **70 intents, not 77.** Seven were held out by design, so results are not directly
  comparable to published BANKING77 numbers.
- **Validation is small (1,430 rows, 13–28 per class).** Differences under ~0.75
  points are within one standard error and are not interpreted.
- **The out-of-domain probes are systematically shorter than in-domain text**, so
  length is a partial confound for OOD rejection. The in-domain unknown set is not
  subject to this and is the more trustworthy measure.
- **Robustness was tested with synthetic character corruption only** — not real
  user typos, code-switching, or transliteration.
- **A single Roman Urdu query was tested informally** (both neural tracks abstained,
  TF-IDF answered confidently). This is an observation, not an evaluation.
- **MiniLM was never fine-tuned**, so the comparison isn't a clean 2×2 of
  representation × fine-tuning.
- Environment pinned to an older library generation for hardware reasons —
  reproducibility gained, compatibility with current releases lost.

## Future work

- Fine-tune MiniLM itself to complete the 2×2.
- Select the rejection score conditional on expected out-of-scope type.
- Export to ONNX to decouple serving from the `transformers` version.
- A Roman Urdu / code-switched evaluation set for Pakistani deployment.

## References

- Casanueva, Temčinas, Gerz, Henderson, Vulić (2020). *Efficient Intent Detection
  with Dual Sentence Encoders.* NLP4ConvAI @ ACL. arXiv:2003.04807
- Larson et al. (2019). *An Evaluation Dataset for Intent Classification and
  Out-of-Scope Prediction.* EMNLP.
- Liu, Wang, Owens, Li (2020). *Energy-based Out-of-distribution Detection.* NeurIPS.
- Guo, Pleiss, Sun, Weinberger (2017). *On Calibration of Modern Neural Networks.* ICML.