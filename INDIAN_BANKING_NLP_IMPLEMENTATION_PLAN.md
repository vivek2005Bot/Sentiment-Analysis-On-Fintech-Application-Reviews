# Indian Banking Review Intelligence --- Master Implementation Plan

## 0. Mission

Build an industry-level NLP/FinTech review-intelligence system from the
existing Google Play Store data. The repository already contains data
under:

``` text
GOVT_Banks/
PVT_banks/
```

Files may be CSV, XLSX, or XLS. **Do not assume filenames, formats, or
column names. Discover them programmatically. Never modify raw files.**

The current corpus is expected to contain approximately 8
Government/Public-sector banks, 8 Private banks, and roughly 1.36M
reviews, but the implementation must calculate and report the actual
numbers from the files rather than hard-code them.

The final system must include: data validation, multilingual/Hinglish
handling, transformer/BERT benchmarking, sentiment, failure detection,
hierarchical risk classification, emotion, aspect sentiment, PageRank,
Government-vs-Private comparison, temporal analysis, response analysis
when metadata exists, explainability, statistical testing, and a
production-style Streamlit dashboard.

------------------------------------------------------------------------

# 1. Non-Negotiable Rules

1.  **Never modify `GOVT_Banks/` or `PVT_banks/`.** Treat them as
    immutable raw input.
2.  Discover `.csv`, `.CSV`, `.xlsx`, `.XLSX`, `.xls`, `.XLS`
    recursively.
3.  Never hard-code source filenames or source column names.
4.  Preserve provenance for every normalized review: `source_file`,
    `source_folder`, `source_row`, `bank`, `bank_type`.
5.  Never silently drop rows. Every removal must be logged with
    before/removed/after counts and reason.
6.  Never invent missing metadata. Missing fields remain null.
7.  Do not aggressively remove emojis, Hinglish, Indian abbreviations,
    transliteration, punctuation, or code-mixed content.
8.  Do not use TF-IDF/Naive Bayes as the main final classifier.
    Classical models may be optional baselines only.
9.  Do not claim a model is superior without a held-out evaluation.
10. Do not claim causality from observational review data. Use
    "associated with", "observed difference", or "correlated with"
    unless a causal design is actually implemented.
11. Mask PII before displaying reviews in Streamlit.
12. Heavy model training must happen offline; Streamlit must load
    precomputed artifacts.
13. Use configuration files rather than scattering constants through
    code.
14. Inspect the existing repository before creating or replacing code.
15. Work incrementally: demo sample first, then annotation sample, then
    full-corpus inference.

------------------------------------------------------------------------

# 2. Research Objective

The project should answer:

-   What are users saying about Indian banking apps?
-   What is the sentiment?
-   Which emotions are present?
-   Which aspects/services are being discussed?
-   Is the complaint ordinary operational friction or a critical
    failure?
-   If it is a failure, is the primary risk Technical Performance or
    Financial Security?
-   How does this differ between Government and Private banks?
-   Which aspects are structurally central according to an interaction
    graph/PageRank?
-   How does performance change across English, Hinglish, transliterated
    and Indian-language reviews?
-   Which transformer model works best for this corpus and task?
-   Does hierarchical routing improve separation of friction from
    critical risk compared with a flat classifier?
-   Are developer response latency and sentiment/rating changes
    statistically associated, when reply metadata exists?

------------------------------------------------------------------------

# 3. Target Repository

Create this structure if it does not already exist:

``` text
PROJECT_ROOT/
├── GOVT_Banks/                    # existing raw data; NEVER MODIFY
├── PVT_banks/                     # existing raw data; NEVER MODIFY
├── data/
│   ├── inventory/
│   ├── interim/
│   ├── processed/
│   ├── annotations/
│   ├── predictions/
│   └── samples/
├── models/
│   ├── sentiment/
│   ├── failure/
│   ├── risk/
│   ├── emotion/
│   └── aspect/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── data_validator.py
│   ├── preprocessing.py
│   ├── language_detector.py
│   ├── annotation.py
│   ├── model_registry.py
│   ├── train.py
│   ├── evaluate.py
│   ├── sentiment_engine.py
│   ├── failure_detector.py
│   ├── risk_engine.py
│   ├── emotion_engine.py
│   ├── aspect_engine.py
│   ├── pagerank_engine.py
│   ├── response_engine.py
│   ├── explainability.py
│   ├── statistical_analysis.py
│   └── utils.py
├── notebooks/
├── streamlit_app/
│   ├── app.py
│   ├── pages/
│   ├── components/
│   └── utils/
├── configs/
│   ├── config.yaml
│   └── model_config.yaml
├── reports/
│   ├── tables/
│   ├── figures/
│   └── model_results/
├── tests/
├── logs/
├── requirements.txt
├── README.md
└── IMPLEMENTATION_PLAN.md
```

------------------------------------------------------------------------

# 4. Phase 0 --- Inspect Environment and Repository

Before changing anything:

1.  List repository files.
2.  Inspect existing Python modules, notebooks, and requirements.
3.  Inspect `GOVT_Banks/` and `PVT_banks/`.
4.  Check Python, PyTorch, Transformers, pandas/polars, CUDA/GPU, RAM
    and disk.
5.  Preserve existing working code unless there is a documented reason
    to refactor it.

Create a system diagnostic that records:

``` text
Python version
OS
CPU
RAM
Disk
CUDA availability
GPU model
GPU VRAM
PyTorch version
Transformers version
pandas/polars version
Streamlit version
```

------------------------------------------------------------------------

# 5. Phase 1 --- Data Discovery and Inventory

Create `src/data_loader.py`.

Recursively discover supported files under:

``` text
GOVT_Banks/
PVT_banks/
```

First read headers and small samples, not entire large files.

Create:

``` text
data/inventory/data_inventory.csv
data/inventory/data_dictionary.json
```

Inventory fields:

``` text
source_file
source_folder
file_type
bank
bank_type
rows
columns
review_column
rating_column
date_column
review_id_column
app_version_column
reply_column
reply_date_column
```

Possible source column names include `review`, `content`, `review_text`,
`text`, `rating`, `score`, `at`, `date`, `review_date`, `reviewId`,
`review_id`, `replyContent`, `reply_text`, etc. Build a robust mapping
layer rather than hard-coding one schema.

If a required field cannot be identified, report it clearly and stop
rather than silently guessing.

------------------------------------------------------------------------

# 6. Phase 2 --- Canonical Dataset

Normalize every file into a common schema:

``` text
review_id
bank
bank_type
review_text
rating
review_date
source_file
source_folder
source_row
```

Optional fields:

``` text
app_version
reply_text
reply_date
user_name
thumbs_up
```

Use null for genuinely missing information.

Recommended storage after normalization:

``` text
data/processed/master_reviews.parquet
```

Use Parquet/DuckDB/SQLite for scalable analytics rather than repeatedly
loading 1M+ CSV rows into Streamlit.

------------------------------------------------------------------------

# 7. Phase 3 --- Data Quality

Run and save a quality report containing:

-   total rows
-   rows per bank
-   Government/Private counts
-   empty reviews
-   duplicate IDs
-   duplicate normalized text
-   suspicious near-duplicates where practical
-   rating distribution
-   missingness
-   date range
-   text length distribution
-   URL/email/phone counts
-   emoji counts
-   HTML artifacts
-   malformed records

Whenever records are removed, write a removal log:

``` text
stage
reason
rows_before
rows_removed
rows_after
```

Do not over-clean. Similar reviews are not automatically duplicates.

------------------------------------------------------------------------

# 8. Phase 4 --- Preprocessing for Transformer NLP

Maintain at least two text fields:

``` text
review_text_raw
review_text_model
```

`review_text_raw` is the original text.

`review_text_model` may normalize harmful artifacts such as malformed
HTML and excessive whitespace, but should preserve semantic information.

Do NOT blindly:

-   remove stopwords;
-   remove all emojis;
-   stem everything;
-   lemmatize everything;
-   delete banking abbreviations;
-   translate all Hinglish into English.

Preserve terms such as:

``` text
UPI KYC NEFT IMPS OTP ATM SMS
```

and Indian code-mixed/transliterated language.

------------------------------------------------------------------------

# 9. Phase 5 --- Language, Script, Hinglish and Transliteration

Create `src/language_detector.py`.

Add fields:

``` text
language
script
is_code_mixed
is_transliterated
language_confidence
```

Target categories may include:

``` text
English
Hindi
Hinglish
Tamil
Tanglish
Telugu
Kannada
Malayalam
Bengali
Marathi
Gujarati
Punjabi
Other_Indian
Other
```

Do not assume Latin script means English.

Examples:

``` text
"upi payment nahi ho raha"
```

should be eligible for Hinglish/code-mixed detection.

``` text
"ரொம்ப slow ah irukku"
```

should be eligible for Tamil-English code-mixing detection.

Produce a language report by bank and bank type.

------------------------------------------------------------------------

# 10. Phase 6 --- Annotation Dataset

Do not manually label the full 1M+ corpus.

Create a high-quality manually reviewed dataset of approximately
10,000--20,000 samples, adjusted to available annotation resources.

Sampling must be stratified across:

-   all banks;
-   Government/Private;
-   ratings;
-   date periods;
-   languages;
-   likely positive/neutral/negative cases;
-   likely failure cases;
-   code-mixed cases.

Create annotation guidelines before training.

Recommended annotation fields:

``` text
review_id
sentiment
failure_flag
operational_friction
critical_failure
primary_risk
secondary_risk
risk_overlap
emotion
severity
language
is_code_mixed
is_transliterated
aspect
aspect_sentiment
annotation_confidence
annotator_id
annotation_version
```

If multiple annotators are available, measure agreement using Cohen's
Kappa or Krippendorff's Alpha as appropriate.

------------------------------------------------------------------------

# 11. Sentiment Labels

Primary production task:

``` text
POSITIVE
NEUTRAL
NEGATIVE
```

Optionally investigate 5-class sentiment, but do not make the system
unnecessarily complex without evidence that it helps.

------------------------------------------------------------------------

# 12. Failure and Friction Labels

A negative review is not automatically a critical failure.

## Operational Friction

Examples:

-   slow UI;
-   confusing navigation;
-   poor UX;
-   feature requests;
-   too many steps;
-   inconvenient workflow;
-   general dissatisfaction without an incident.

## Critical Failure

Examples:

-   app crash during transaction;
-   server unavailable during transaction;
-   repeated timeout preventing a transaction;
-   unauthorized transaction;
-   unexpected money deduction;
-   money lost/stuck in transit;
-   fraud;
-   suspected account compromise;
-   suspected data breach.

Fields:

``` text
failure_flag = YES/NO
operational_friction = YES/NO
critical_failure = YES/NO
```

------------------------------------------------------------------------

# 13. Core Two-Tier Risk Architecture

This is the central proposed architecture.

## Tier 1 --- Failure Gate

``` text
Review
  |
  +--> Failure = NO
  |       |
  |       +--> Operational Friction / Standard E-Satisfaction analysis
  |
  +--> Failure = YES
          |
          +--> Tier 2
```

## Tier 2 --- Risk Classifier

Allowed primary risk classes:

``` text
TECHNICAL_PERFORMANCE_RISK
FINANCIAL_SECURITY_RISK
```

The primary risk must be mutually exclusive.

Examples:

``` text
"app crashes every time I open it"
=> TECHNICAL_PERFORMANCE_RISK
```

``` text
"money was deducted without authorization"
=> FINANCIAL_SECURITY_RISK
```

------------------------------------------------------------------------

# 14. Mixed-Risk Reviews

Do not discard nuance merely because the official primary-risk output is
single-label.

Example:

``` text
"app crashes when paying rent and my money is stuck"
```

This contains technical and financial signals.

Store:

``` text
primary_risk = FINANCIAL_SECURITY_RISK
secondary_risk = TECHNICAL_PERFORMANCE_RISK
risk_overlap = YES
```

The official two-class benchmark must use only `primary_risk`, so no
overlap occurs in the benchmark target.

Recommended initial deterministic priority rule:

``` text
IF unauthorized/missing/deducted/fraud/security/data-breach signal:
    primary = FINANCIAL_SECURITY_RISK
ELSE IF crash/server/timeout/system-availability signal:
    primary = TECHNICAL_PERFORMANCE_RISK
ELSE:
    failure = NO
```

This is a starting annotation rule. Version any change to it.

------------------------------------------------------------------------

# 15. Emotion Analysis

Suggested labels:

``` text
ANGER
FRUSTRATION
FEAR
SADNESS
DISAPPOINTMENT
JOY
TRUST
CONFUSION
NEUTRAL
OTHER
```

Use multi-label emotion if the annotation budget supports it.

Example:

``` text
"Money deducted and nobody is helping 😡😭"
=> ANGER + FEAR
```

------------------------------------------------------------------------

# 16. Severity

Create an explicit severity policy:

``` text
1 = Minor suggestion
2 = Minor inconvenience
3 = Significant operational problem
4 = Critical technical failure
5 = Financial/security incident
```

Do not present this as an objective universal risk score. It is a
project-defined annotation policy.

------------------------------------------------------------------------

# 17. Aspect Taxonomy

Initial aspect taxonomy:

``` text
APP_PERFORMANCE
UI_UX
LOGIN
OTP
UPI
NEFT
IMPS
ATM
CUSTOMER_SERVICE
TRANSACTION
SECURITY
FRAUD
MONEY_TRANSFER
KYC
ACCOUNT
LOAN
CREDIT_CARD
NOTIFICATION
SERVER
CRASH
FEATURE_REQUEST
CHARGES
OTHER
```

Version-control the taxonomy.

------------------------------------------------------------------------

# 18. Aspect-Based Sentiment

Do not reduce every review to one sentiment.

Example:

``` text
"App is good but UPI is terrible and customer service is useless."
```

Expected conceptual output:

``` text
APP              -> POSITIVE
UPI              -> NEGATIVE
CUSTOMER_SERVICE  -> NEGATIVE
```

Build a manually validated aspect test set before reporting final
aspect-extraction performance.

------------------------------------------------------------------------

# 19. Phase 7 --- Transformer/BERT Benchmark

Do not literally claim "all BERT models." Benchmark a representative
set:

### General

-   BERT
-   RoBERTa
-   DistilBERT

### Multilingual

-   mBERT
-   XLM-R Base
-   XLM-R Large if hardware permits

### Indian-language focused

-   IndicBERT
-   IndicBERTv2
-   MuRIL

### Newer Indic models

-   evaluate compatible IndicBERT-v3 variants when hardware and task
    setup permit.

IndicBERT's official model card describes it as a multilingual ALBERT
model pretrained on 12 major Indian languages. IndicBERTv2 covers
substantially more Indic languages. AI4Bharat currently lists
IndicBERT-v3 270M, 1B and 4B variants. MuRIL is especially relevant to
Indian languages and transliterated text.

Reference model pages:

-   https://huggingface.co/ai4bharat/indic-bert
-   https://huggingface.co/ai4bharat/IndicBERTv2-SS
-   https://huggingface.co/ai4bharat/models
-   https://huggingface.co/google/muril-base-cased

Important: verify each checkpoint's task compatibility. Do not directly
treat a masked-language-model or token-classification checkpoint as a
sequence-classification model without the correct fine-tuning
architecture.

------------------------------------------------------------------------

# 20. Model Registry

Create `configs/model_config.yaml`.

Example:

``` yaml
models:
  - name: bert-base-uncased
    family: BERT
  - name: roberta-base
    family: RoBERTa
  - name: distilbert-base-uncased
    family: DistilBERT
  - name: bert-base-multilingual-cased
    family: mBERT
  - name: xlm-roberta-base
    family: XLM-R
  - name: ai4bharat/indic-bert
    family: IndicBERT
  - name: ai4bharat/IndicBERTv2-MLM-only
    family: IndicBERTv2
  - name: google/muril-base-cased
    family: MuRIL
```

The exact checkpoint should be configurable, validated at startup, and
recorded in experiment metadata.

------------------------------------------------------------------------

# 21. Model Tasks

Benchmark separately:

### Task A

Sentiment: Positive/Neutral/Negative.

### Task B

Failure: Failure/No Failure.

### Task C

Risk: Technical/Financial, only on failure-positive examples.

### Task D

Emotion.

### Task E

Aspect classification/extraction.

Do not force a model into an unsuitable architecture just to produce a
comparison table.

------------------------------------------------------------------------

# 22. Evaluation Metrics

Report:

``` text
Accuracy
Precision
Recall
Macro F1
Weighted F1
ROC-AUC where appropriate
PR-AUC where appropriate
Confusion Matrix
```

For imbalanced risk tasks, prioritize Macro-F1 and per-class
recall/precision rather than accuracy alone.

Also report:

``` text
training_time
inference_time
model_size
parameter_count where available
GPU_memory where available
```

------------------------------------------------------------------------

# 23. Risk-Sensitive Evaluation

A false negative for Financial Security Risk may have greater
operational significance than a false positive. Build an optional
risk-cost matrix, but explicitly document the chosen weights.

Do not invent an "objective" cost. Treat it as a project-defined
sensitivity analysis.

------------------------------------------------------------------------

# 24. Flat vs Hierarchical Experiment

This is a key research experiment.

## Flat baseline

``` text
Review
  -> NORMAL / OPERATIONAL_FRICTION / TECHNICAL_RISK / FINANCIAL_RISK
```

## Proposed hierarchy

``` text
Review
  -> Failure?
      -> NO  -> Friction/Normal
      -> YES -> Technical/Financial
```

Compare:

``` text
Macro F1
Financial Recall
Technical Recall
Friction F1
Critical Failure F1
Confusion Matrix
```

Research question:

> Does hierarchical routing reduce confusion between operational
> friction and acute financial/security risk?

Only report improvement if measured on a held-out test set.

------------------------------------------------------------------------

# 25. Phase 8 --- Language-Specific Evaluation

Create test subsets for:

``` text
English
Hinglish
Hindi
Tamil
Telugu
Other Indian languages
Code-mixed
Non-code-mixed
Romanized/transliterated
Native-script
```

Report model performance per subset:

``` text
Precision
Recall
Macro F1
```

This is essential to the Indian-language contribution.

------------------------------------------------------------------------

# 26. Phase 9 --- Cross-Bank and Time Generalization

Use three evaluation designs:

### Random split

Baseline.

### Time split

Older reviews for training, newer reviews for test.

### Cross-bank split

Hold out one bank as an unseen test bank and rotate held-out banks where
feasible.

Prevent duplicate/near-duplicate leakage across splits.

This tests whether the model learns banking-language patterns rather
than memorizing bank-specific phrases.

------------------------------------------------------------------------

# 27. Phase 10 --- Ablation Studies

At minimum:

1.  Text only.
2.  Text + emojis.
3.  Text + language/code-mixing metadata.
4.  Text + emojis + language + transliteration features.
5.  Flat architecture.
6.  Hierarchical architecture.

Report the change in Macro F1 and relevant risk recalls.

------------------------------------------------------------------------

# 28. Emoji Analysis

Do not delete emojis by default.

Create:

``` text
emoji_present
emoji_count
emoji_types
```

Compare models with emojis retained vs removed on a controlled test set.

------------------------------------------------------------------------

# 29. Rating/Sentiment Mismatch

Compare star rating with model-predicted sentiment.

Investigate:

``` text
5-star + negative text
1-star + positive text
```

Possible causes include sarcasm, rating mistakes, mixed sentiment,
review updates, or model errors. Do not automatically label mismatches
as sarcasm.

------------------------------------------------------------------------

# 30. Optional Sarcasm Module

Create a small manually validated sarcasm dataset before training.

Example:

``` text
"Wow amazing app, crashed five times 😂"
```

Evaluate sarcasm separately from sentiment.

------------------------------------------------------------------------

# 31. Phase 11 --- PageRank / Aspect Interaction Graph

The existing `pagerank_engine.py` should construct an aspect-interaction
graph.

Definition:

``` text
Node = aspect
Edge = two aspects occurring in the same review
Edge weight = co-occurrence frequency
```

Run PageRank with:

``` text
damping_factor = 0.85
```

Record:

``` text
damping_factor
tolerance
max_iterations
iterations_used
converged
```

Do not confuse PageRank centrality with severity. A central topic is not
automatically a severe topic.

Create:

``` text
data/predictions/aspect_edges.parquet
reports/tables/pagerank_global.csv
reports/tables/pagerank_by_bank.csv
reports/tables/pagerank_govt.csv
reports/tables/pagerank_private.csv
```

PageRank output fields:

``` text
aspect
pagerank_score
degree
weighted_degree
bank
bank_type
```

------------------------------------------------------------------------

# 32. Phase 12 --- Government vs Private Comparison

For every bank calculate:

``` text
total_reviews
average_rating
positive_rate
neutral_rate
negative_rate
failure_rate
operational_friction_rate
critical_failure_rate
technical_risk_rate
financial_risk_rate
severity_distribution
emotion_distribution
aspect_distribution
language_distribution
```

Use normalized rates, not raw counts alone, because banks may have
different review volumes.

Always display denominators.

------------------------------------------------------------------------

# 33. Improvement Analysis

For each Government bank, identify:

``` text
Top technical problems
Top financial concerns
Top UX/friction issues
Top negative aspects
Top angry/frustrated emotions
Top critical failures
```

Compare against the Private-bank group. Then perform the reverse
comparison for Private banks.

Create an evidence table:

``` text
bank
issue
bank_group_rate
comparison_group_rate
difference
sample_size
confidence_interval
```

Do not make a generic "better/worse" claim without defining the metric
and reporting the underlying evidence.

------------------------------------------------------------------------

# 34. Statistical Testing

Use methods appropriate to the data:

-   Chi-square/Fisher's exact test for categorical distributions where
    appropriate.
-   Mann-Whitney U/Kruskal-Wallis for suitable non-normal
    numeric/ordinal comparisons.
-   Logistic or ordinal regression for response/rating/sentiment
    relationships where appropriate.

Report:

``` text
sample_size
effect_size
confidence_interval
p_value
```

Do not rely on p-values alone.

Where multiple comparisons are performed, consider a correction such as
Benjamini-Hochberg.

------------------------------------------------------------------------

# 35. Phase 13 --- Developer Response Analysis

Only enable this module if reply content and timestamps exist.

Required:

``` text
review_date
reply_date
reply_text
```

Calculate:

``` text
response_time_hours
```

Buckets:

``` text
<24h
24–48h
48–72h
>72h
No response
```

Test whether response latency is associated with:

``` text
rating improvement
sentiment improvement
negative follow-up
```

Do not claim "48-hour responses recover sentiment" unless the project's
own analysis establishes it. The correct research framing is:

> Is response latency statistically associated with sentiment/rating
> reversal?

------------------------------------------------------------------------

# 36. Phase 14 --- App Version Analysis

If `app_version` exists, calculate by version:

``` text
negative_rate
failure_rate
technical_risk_rate
financial_risk_rate
crash_rate
```

Look for unusual changes around releases. Do not claim that a release
caused an issue without an appropriate causal design.

------------------------------------------------------------------------

# 37. Phase 15 --- Temporal and Early-Warning Analysis

Produce daily/weekly/monthly trends for:

``` text
sentiment
failure
technical risk
financial risk
friction
emotion
major aspects
```

Implement optional anomaly detection comparing current rates to rolling
historical baselines.

Example:

``` text
UPI Technical Risk
Historical baseline = X%
Current period = Y%
Status = Potential anomaly
```

This is an analytical alert, not proof of an outage.

------------------------------------------------------------------------

# 38. Phase 16 --- Explainability

For selected models implement a defensible explanation method such as:

``` text
Integrated Gradients
SHAP where computationally feasible
token attribution
```

For a review display:

``` text
Prediction
Confidence
Important tokens/phrases
```

Do not claim attention weights alone prove model reasoning.

------------------------------------------------------------------------

# 39. Phase 17 --- Confidence and Human-in-the-Loop

Every prediction must store:

``` text
prediction
confidence
model_name
model_version
prediction_timestamp
```

Create a configurable threshold such as:

``` yaml
human_review:
  enabled: true
  confidence_threshold: 0.70
```

Low-confidence, mixed-risk, model-disagreement, financial-risk and
unusual-language cases can enter a human review queue.

The threshold must be validated rather than treated as universally
correct.

------------------------------------------------------------------------

# 40. Phase 18 --- Master Prediction Schema

Create a master prediction dataset with:

``` text
review_id
bank
bank_type
review_text
rating
review_date
app_version
language
script
is_code_mixed
is_transliterated
sentiment
sentiment_confidence
failure_flag
failure_confidence
operational_friction
critical_failure
primary_risk
primary_risk_confidence
secondary_risk
risk_overlap
emotion
emotion_confidence
severity
aspect
aspect_sentiment
response_time_hours
response_bucket
sarcasm
sarcasm_confidence
model_name
model_version
prediction_timestamp
```

Not every field will be populated for every review. Missing information
must remain null.

------------------------------------------------------------------------

# 41. Privacy and PII

Banking reviews may contain phone numbers, account numbers, transaction
IDs, emails and names.

Create a reusable PII-masking function in `src/utils.py`.

Mask before dashboard display:

``` text
phone numbers
account numbers
transaction IDs
email addresses
other obvious personal identifiers
```

Example:

``` text
Account ****1234
Phone ******4321
Email a***@example.com
```

Do not send raw sensitive review data to external APIs unless explicitly
approved.

------------------------------------------------------------------------

# 42. Full Corpus Strategy

Do not fine-tune every large transformer on the complete corpus at the
beginning.

Use:

``` text
Full corpus
    -> stratified annotation sample
    -> benchmark representative models
    -> select strongest candidates
    -> final fine-tuning
    -> batch inference on full corpus
```

For development, use a demo dataset first.

Create:

``` text
data/samples/demo_reviews.csv
```

The complete pipeline must be able to run against this sample.

------------------------------------------------------------------------

# 43. Recommended Initial Training Configuration

Use a configurable starting point, not a guaranteed optimum:

``` yaml
max_length: 256
learning_rate: 2.0e-5
epochs: 3
warmup_ratio: 0.1
weight_decay: 0.01
early_stopping: true
```

Tune batch size, learning rate, max length and epochs using validation
data.

Consider 256/384/512 sequence lengths where justified.

Use mixed precision and gradient accumulation when hardware permits.

------------------------------------------------------------------------

# 44. Class Imbalance

First inspect class distributions.

Potential methods:

``` text
class-weighted loss
weighted sampling
oversampling
undersampling
focal loss
```

Do not apply every method simultaneously without an experiment.

Select the approach based on validation performance and risk-class
recall.

------------------------------------------------------------------------

# 45. Data Leakage Prevention

Mandatory checks:

-   no duplicate review IDs across train/test;
-   no near-identical reviews across train/test;
-   no future information leakage;
-   no developer-response information leaking into a pre-response
    prediction task;
-   no same review represented in multiple derived datasets with
    conflicting labels.

For response-effectiveness analysis, define exactly what information was
available at each timestamp.

------------------------------------------------------------------------

# 46. Experiment Tracking

Every experiment must save:

``` text
experiment_id
model
task
dataset_version
train_size
validation_size
test_size
language_distribution
hyperparameters
epochs
learning_rate
batch_size
max_length
seed
accuracy
macro_f1
weighted_f1
precision
recall
roc_auc
pr_auc
training_time
inference_time
```

Save JSON and CSV results under `reports/model_results/`.

------------------------------------------------------------------------

# 47. Reproducibility

Set and record seeds for:

``` text
Python
NumPy
PyTorch
Transformers
```

Save:

``` text
model checkpoint
tokenizer
configuration
training arguments
dataset version
annotation version
random seed
```

If Git is available, record the commit hash in experiment metadata.

------------------------------------------------------------------------

# 48. Streamlit Dashboard

The dashboard must be a presentation/decision-support layer over
precomputed outputs. It must not train models on page load.

Recommended dependencies:

``` text
streamlit
plotly
pandas or polars
duckdb (recommended)
```

Recommended pages:

``` text
1. Executive Overview
2. Sentiment Intelligence
3. Risk Intelligence
4. Government vs Private
5. Language & Hinglish
6. Model Benchmark
7. Aspect & PageRank
8. Response Intelligence
9. Review Investigator
10. Critical Review Queue
11. Model Explainability
12. Early Warning
13. Data & Methodology
```

------------------------------------------------------------------------

# 49. Streamlit Page Specifications

## Executive Overview

KPIs:

``` text
Total Reviews
Government Reviews
Private Reviews
Banks
Negative Rate
Failure Rate
Technical Risk Rate
Financial Risk Rate
```

Charts:

-   sentiment distribution;
-   risk distribution;
-   Government vs Private;
-   reviews over time.

## Sentiment Intelligence

Filters:

``` text
Bank
Bank Type
Date
Rating
Language
Sentiment
```

Charts:

-   sentiment by bank;
-   sentiment over time;
-   sentiment by rating;
-   sentiment by language.

## Risk Intelligence

Show:

``` text
Technical Performance Risk
Financial Security Risk
Operational Friction
Critical Failure
```

Charts:

-   risk by bank;
-   risk over time;
-   risk by language;
-   severity distribution.

## Government vs Private

Compare normalized rates for:

``` text
Sentiment
Technical Risk
Financial Risk
Friction
Critical Failure
Severity
Emotion
Top Aspects
```

## Language & Hinglish

Show:

``` text
language distribution
code-mixed rate
transliterated rate
sentiment by language
risk by language
model F1 by language
```

## Model Benchmark

Show:

``` text
Model
Task
Accuracy
Macro F1
Weighted F1
Precision
Recall
Inference Time
Model Size
```

Include confusion matrices and language-specific performance.

## Aspect & PageRank

Show:

``` text
Top aspects
Top PageRank aspects
Government PageRank
Private PageRank
Bank-specific PageRank
```

If stable, render an interactive aspect network.

## Response Intelligence

Only show if reply metadata exists. Display:

``` text
Response Rate
Average Response Time
Median Response Time
<24h
24–48h
48–72h
>72h
No Response
```

Then show association between response latency and rating/sentiment
change.

## Review Investigator

Allow search/filter and show:

``` text
Original review
Bank
Bank type
Rating
Date
Language
Code-mixing
Sentiment + confidence
Failure + confidence
Primary risk
Secondary signal
Emotion
Severity
Aspects
Developer response
Response time
```

## Critical Review Queue

Filter for critical failures and/or financial-security risk. Include
confidence, severity and response status.

## Model Explainability

For a selected review display prediction, confidence and supported
attribution/explanation.

## Early Warning

Show potential anomalies in risk/topic rates compared with historical
baselines.

## Data & Methodology

Show:

-   data sources;
-   date range;
-   number of reviews;
-   bank counts;
-   label definitions;
-   model names/versions;
-   evaluation methodology;
-   denominator definitions;
-   known limitations.

------------------------------------------------------------------------

# 50. Issue Explorer

Build a search feature where an analyst can enter:

``` text
UPI
OTP
server
fraud
money deducted
login
```

Return:

``` text
mention count
banks affected
Government vs Private distribution
sentiment
risk
emotion
trend
language distribution
top representative reviews
```

All displayed review text must be PII-masked.

------------------------------------------------------------------------

# 51. Industry-Level Bank Improvement Matrix

Create a table:

``` text
bank
issue
bank_group_rate
comparison_group_rate
difference
volume
severity
suggested_operational_area
```

The "suggested operational area" must be based on observed issue
categories, not invented internal knowledge about the bank.

------------------------------------------------------------------------

# 52. Industry-Level Early Warning

Build an anomaly/monitoring view.

For example:

``` text
UPI Technical Risk
Historical baseline: X%
Current period: Y%
Status: Potential anomaly
```

Allow filtering by bank, bank type, language and aspect.

This is an analytical early-warning signal, not confirmation of an
actual production outage.

------------------------------------------------------------------------

# 53. Model Disagreement

Run multiple selected models on a common evaluation/sample set.

Identify cases where:

``` text
Model A prediction != Model B prediction
```

Prioritize disagreement cases involving:

-   Financial Security Risk;
-   Hinglish;
-   sarcasm candidates;
-   mixed-risk cases;
-   low confidence.

These can feed the human-review queue.

------------------------------------------------------------------------

# 54. Expected Output Files

Create at minimum:

``` text
data/processed/master_reviews.parquet

data/predictions/master_predictions.parquet
data/predictions/sentiment_predictions.parquet
data/predictions/failure_predictions.parquet
data/predictions/risk_predictions.parquet
data/predictions/emotion_predictions.parquet
data/predictions/aspect_predictions.parquet
data/predictions/aspect_edges.parquet

reports/tables/bank_summary.csv
reports/tables/govt_vs_private.csv
reports/tables/language_summary.csv
reports/tables/model_benchmark.csv
reports/tables/pagerank_global.csv
reports/tables/pagerank_by_bank.csv
reports/tables/response_analysis.csv

reports/figures/
reports/model_results/
```

------------------------------------------------------------------------

# 55. Final Research Tables

Automatically generate:

### Table 1 --- Dataset

``` text
Bank | Bank Type | Reviews | Average Rating | Date Range
```

### Table 2 --- Language

``` text
Language | Reviews | Percentage | Negative Rate | Failure Rate
```

### Table 3 --- Model Benchmark

``` text
Model | Task | Accuracy | Macro F1 | Weighted F1 | Precision | Recall | Inference Time
```

### Table 4 --- Risk

``` text
Bank | Technical Risk % | Financial Risk % | Friction % | Critical Failure %
```

### Table 5 --- Government vs Private

``` text
Metric | Government | Private | Difference | Statistical Test | Effect Size
```

### Table 6 --- PageRank

``` text
Aspect | PageRank | Degree | Weighted Degree
```

### Table 7 --- Response

``` text
Response Bucket | Reviews | Mean Rating Change | Median Rating Change | Sentiment Change
```

------------------------------------------------------------------------

# 56. Required Visualizations

At minimum:

1.  Sentiment distribution.
2.  Sentiment over time.
3.  Government vs Private sentiment.
4.  Technical vs Financial risk.
5.  Risk by bank.
6.  Risk by language.
7.  Risk by rating.
8.  Emotion distribution.
9.  Emotion by risk.
10. Aspect frequency.
11. Aspect sentiment.
12. PageRank top aspects.
13. Government vs Private PageRank.
14. Response latency distribution.
15. Response latency vs rating/sentiment change.
16. Model benchmark.
17. Confusion matrices.
18. Language-specific model performance.
19. Flat vs hierarchical comparison.
20. Severity distribution.
21. App-version trend when available.
22. Potential anomaly timeline.

------------------------------------------------------------------------

# 57. Global Dashboard Filters

Where appropriate:

``` text
Bank Type
Bank
Date Range
Rating
Language
Sentiment
Risk
Severity
Emotion
Aspect
```

All metrics must preserve correct denominators after filtering.

------------------------------------------------------------------------

# 58. KPI Definitions

Every dashboard KPI must have a visible methodology definition.

Example:

``` text
Technical Risk Rate
= Technical-risk reviews / all valid reviews
```

If calculating risk only among failure reviews, label it explicitly:

``` text
Technical risk among failure-positive reviews
```

Never mix denominators silently.

------------------------------------------------------------------------

# 59. Performance Requirements

Do not load the entire raw 1M+ corpus into Streamlit on every
interaction.

Recommended:

``` text
Raw CSV/XLSX
      -> normalized Parquet
      -> DuckDB/cached aggregates
      -> Streamlit
```

Precompute expensive aggregations and model predictions.

------------------------------------------------------------------------

# 60. Configuration

Create `configs/config.yaml` similar to:

``` yaml
data:
  govt_dir: "GOVT_Banks"
  private_dir: "PVT_banks"
  processed_dir: "data/processed"

models:
  max_length: 256
  batch_size: 16
  learning_rate: 0.00002
  epochs: 3

pagerank:
  damping: 0.85
  tolerance: 0.000001
  max_iterations: 100

human_review:
  enabled: true
  confidence_threshold: 0.70

dashboard:
  mask_pii: true
```

Actual batch size must be adapted to hardware.

------------------------------------------------------------------------

# 61. Logging

Use Python `logging`.

Every major stage logs:

``` text
timestamp
stage
input_rows
output_rows
duration
warnings
errors
```

------------------------------------------------------------------------

# 62. Tests

Create unit tests for:

``` text
data loader
column mapper
PII masking
language mapping
risk routing
mixed-risk priority
PageRank convergence
metric calculations
response-time calculation
```

Also create an integration test using the demo dataset.

------------------------------------------------------------------------

# 63. CLI Workflow

Provide commands similar to:

``` bash
python -m src.data_loader
python -m src.data_validator
python -m src.preprocessing
python -m src.language_detector
python -m src.train --task sentiment
python -m src.train --task failure
python -m src.train --task risk
python -m src.train --task emotion
python -m src.evaluate
python -m src.pagerank_engine
python -m src.statistical_analysis
streamlit run streamlit_app/app.py
```

If the implementation uses different commands, document them in
`README.md`.

------------------------------------------------------------------------

# 64. Quality Gates

Do not proceed to the next major phase until the previous phase passes
its gate.

## Data gate

Know:

``` text
file count
bank count
review count
bank mapping
columns
missingness
duplicates
```

## Annotation gate

Have:

``` text
annotation guidelines
class distribution
agreement statistics where applicable
```

## Model gate

Have:

``` text
held-out test set
leakage checks
baseline metrics
confusion matrix
```

## Hierarchical gate

Have:

``` text
failure-gate metrics
risk-classifier metrics
end-to-end metrics
```

## Dashboard gate

Every chart must use validated, versioned outputs.

------------------------------------------------------------------------

# 65. Final Research Questions

The final report should answer:

### RQ1

How effectively can transformer-based models classify sentiment in
multilingual and code-mixed Indian banking reviews?

### RQ2

Does hierarchical risk classification reduce confusion between
operational friction and critical financial/technical failures compared
with a flat classifier?

### RQ3

Which transformer architecture performs best across English, Hinglish
and Indian-language subsets?

### RQ4

Does transliteration-aware or Indian-language-focused pretraining
improve Romanized/code-mixed review classification?

### RQ5

What are the dominant technical and financial-security risk patterns
across Government and Private banks?

### RQ6

Which aspects are structurally central in the banking-review interaction
network?

### RQ7

Is response latency associated with rating or sentiment reversal when
reply metadata is available?

### RQ8

Which model provides the most useful trade-off between predictive
performance, multilingual performance, latency and compute cost?

------------------------------------------------------------------------

# 66. Definition of Done

The project is complete only when:

## Data

-   [ ] Government files discovered and inventoried.
-   [ ] Private files discovered and inventoried.
-   [ ] Canonical dataset created.
-   [ ] Duplicate handling documented.
-   [ ] Data-quality report generated.

## NLP

-   [ ] Language detection.
-   [ ] Hinglish/code-mixing detection.
-   [ ] Sentiment model.
-   [ ] Failure detector.
-   [ ] Hierarchical risk model.
-   [ ] Emotion model.
-   [ ] Aspect extraction.
-   [ ] Aspect sentiment.

## Model benchmark

-   [ ] BERT.
-   [ ] RoBERTa.
-   [ ] mBERT.
-   [ ] XLM-R.
-   [ ] IndicBERT.
-   [ ] IndicBERTv2 or compatible variant.
-   [ ] MuRIL.
-   [ ] Language-specific evaluation.
-   [ ] Cross-bank evaluation.
-   [ ] Time-based evaluation.
-   [ ] Flat vs hierarchical comparison.
-   [ ] Ablation studies.

## Analytics

-   [ ] Government vs Private.
-   [ ] Bank-level analysis.
-   [ ] Risk analysis.
-   [ ] Emotion analysis.
-   [ ] Aspect analysis.
-   [ ] PageRank.
-   [ ] Temporal analysis.
-   [ ] App-version analysis if available.
-   [ ] Response analysis if available.
-   [ ] Statistical testing.
-   [ ] Explainability.
-   [ ] Human review queue.

## Dashboard

-   [ ] Executive Overview.
-   [ ] Sentiment.
-   [ ] Risk.
-   [ ] Government vs Private.
-   [ ] Language/Hinglish.
-   [ ] Model Benchmark.
-   [ ] Aspect/PageRank.
-   [ ] Response Intelligence.
-   [ ] Review Investigator.
-   [ ] Critical Review Queue.
-   [ ] Explainability.
-   [ ] Early Warning.
-   [ ] Methodology.

## Engineering

-   [ ] Raw data never modified.
-   [ ] Configuration file.
-   [ ] Logging.
-   [ ] Tests.
-   [ ] Reproducibility.
-   [ ] PII masking.
-   [ ] Model metadata.
-   [ ] README.
-   [ ] Requirements.
-   [ ] Demo pipeline.

------------------------------------------------------------------------

# 67. EXACT FIRST ACTIONS FOR CLINE / ANTIGRAVITY

The coding agent must perform these actions **in this order**:

1.  Inspect the current repository.
2.  Inspect `GOVT_Banks/`.
3.  Inspect `PVT_banks/`.
4.  Identify every CSV/XLS/XLSX file.
5.  Read headers and small samples only.
6.  Determine bank names and bank types from folder/file metadata; do
    not invent mappings.
7.  Detect candidate review/rating/date/reply columns.
8.  Create `data/inventory/data_inventory.csv`.
9.  Create `data/inventory/data_dictionary.json`.
10. Report actual file count, bank count, row count and schema.
11. Build the canonical loader.
12. Build the canonical dataset.
13. Run data-quality checks.
14. Create a stratified annotation sample.
15. Create the annotation guideline/template.
16. Only after labels exist, train the benchmark models.
17. Evaluate on held-out data.
18. Implement the hierarchical classifier.
19. Run flat-vs-hierarchical evaluation.
20. Run language-specific and cross-bank evaluation.
21. Run emotion/aspect/PageRank analysis.
22. Run Government-vs-Private statistical comparison.
23. Run response analysis only if timestamps exist.
24. Build Streamlit using precomputed artifacts.
25. Run tests and an end-to-end demo.
26. Update `README.md` with exact commands and results.

**Do not skip the data inventory and annotation design.**

------------------------------------------------------------------------

# 68. Final Architecture

``` text
                 GOVT_Banks + PVT_banks
                           |
                           v
                   DATA DISCOVERY
                           |
                           v
                  CANONICAL DATASET
                           |
                           v
                    QUALITY CHECK
                           |
                           v
                     DEDUPLICATION
                           |
                           v
              LANGUAGE / SCRIPT / CODE-MIX
                           |
                           v
                    ANNOTATED DATA
                           |
                           v
                  BERT MODEL BENCHMARK
                           |
              +------------+------------+
              |                         |
              v                         v
         SENTIMENT              FAILURE DETECTOR
                                      |
                             +--------+--------+
                             |                 |
                            NO                YES
                             |                 |
                             v                 v
                         FRICTION       RISK CLASSIFIER
                                             |
                                  +----------+----------+
                                  |                     |
                             TECHNICAL              FINANCIAL
                                  |                     |
                                  +----------+----------+
                                             |
                                             v
                                          SEVERITY
                                             |
                                             v
                                          EMOTION
                                             |
                                             v
                                     ASPECT EXTRACTION
                                             |
                                             v
                                      ASPECT SENTIMENT
                                             |
                                             v
                                           PAGERANK
                                             |
                                             v
                                  GOVT vs PRIVATE ANALYSIS
                                             |
                                             v
                                  RESPONSE/TEMPORAL ANALYSIS
                                             |
                                             v
                                       EXPLAINABILITY
                                             |
                                             v
                                       STREAMLIT APP
```

------------------------------------------------------------------------

# 69. Important Interpretation Rules

Keep these concepts separate:

-   **Sentiment** = positive/neutral/negative attitude.
-   **Emotion** = expressed emotional state.
-   **Aspect** = service/topic being discussed.
-   **Operational Friction** = inconvenience, UX issue, feature request
    or non-critical dissatisfaction.
-   **Failure** = significant system/transaction problem.
-   **Technical Performance Risk** = system/app/server/performance
    failure.
-   **Financial Security Risk** = fraud, unauthorized transaction, money
    loss/stuck funds, or security/data-breach concern.
-   **Severity** = seriousness according to the project's annotation
    policy.
-   **PageRank** = structural centrality in the aspect co-occurrence
    network.

Do not collapse these into one unexplained score.

------------------------------------------------------------------------

# 70. Suggested Final Project Titles

Academic:

**Transformer-Based Hierarchical Risk and Sentiment Intelligence for
Indian Banking App Reviews**

Industry:

**Indian Banking Review Intelligence Platform: Multilingual Sentiment,
Failure Detection, Risk Classification and Competitive Analytics**

------------------------------------------------------------------------

# 71. Final Agent Instruction

Treat this file as the **master specification and implementation
contract**.

When an implementation detail is uncertain:

1.  inspect the existing repository and actual data;
2.  prefer configuration over hard-coded assumptions;
3.  document the decision in `docs/DECISIONS.md`;
4.  do not invent data or metadata;
5.  keep raw data immutable;
6.  validate every major stage before continuing.

The objective is not merely to produce a sentiment-analysis notebook.
The objective is to build a reproducible, evaluated, multilingual,
hierarchical banking-risk intelligence system with a Streamlit
decision-support interface.
