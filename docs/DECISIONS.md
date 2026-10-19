# Implementation decisions log (plan section 71.3)

## D1 - Python 3.14 (Homebrew) venv
`/usr/bin/python3` is blocked by an Xcode license prompt; project venv uses
Homebrew Python 3.14.6. pandas 3.0.6, pyarrow 25, torch 2.14, transformers
5.17 all publish cp314 wheels - verified by installation.

## D2 - pandas 3 / pyarrow dtype strictness
Mixed object columns (str + float NaN + Timestamp) crash parquet conversion.
Master loader coerces: string columns -> `string` dtype, dates -> datetime64,
`thumbs_up` -> `Int64`. Engine code avoids `pd.Series(pd.NA, dtype=float64)`
(uses np.nan) and `.fillna(False)` on NA-able boolean masks before boolean
indexing.

## D3 - Row counting: pyarrow exact, not wc -l
CSV line counts overcount multiline quoted reviews; inventory uses
`pyarrow.csv` (newlines_in_values=True) and labels rows_source exactly.

## D4 - Empty Pvt bank files
HDFC/Kotak/IDFC/Federal/Bandhan/City Union CSVs are 4-byte BOM-only
(interrupted scrape). Status `empty_file`, usable=False. Never fabricated.

## D5 - Sentiment baseline weight
rating_prior_weight=0.30 (lexicon 0.70) so 5-star sarcastic text is not
auto-positive; plan s.29 mismatch flag marks rating/text disagreements for
human review rather than assuming sarcasm.

## D6 - Severity collision order
Financial incident (5) overrides critical failure (4); friction (2) only
downgrades non-failure rows; feature-request-only friction = 1.

## D7 - Silver vs human labels
`annotation_sample.csv` ships blank human label columns;
`annotation_sample_silver.csv` carries `label_source=silver_rules_v1`.
Benchmarks on silver labels are provisional by construction.

## D8 - transformers 5.x API
`TrainingArguments(eval_strategy=...)` (not evaluation_strategy); MPS device
auto-selected; fp16 off (unreliable on MPS).

## D9 - Unit tests must never write into real output directories
Root cause found: `tests/test_analytics.py` called the annotation builder (and
the temporal engines) without an output override, so **every `pytest` run**
silently replaced the 12,513-row human sample with 5 fixture rows and rewrote
`reports/tables/temporal_trends.csv` + `bank_improvement_matrix.csv` with
10-row fixture aggregates. Two-part fix:

1. `tests/conftest.py` provides an autouse `isolate_output_dirs` fixture that
   monkeypatches `src.config.get_paths()` so every writable `*_dir` resolves
   into a per-test tmp dir. Raw data and config paths stay read-only.
2. `annotation.build_annotation_sample()` gained `out_dir` / `force` and logs
   an explicit `OVERWRITE GUARD` warning whenever an existing sample is
   replaced by a smaller one. `python -m src.annotation --demo` runs on the
   demo sample and writes only to `data/samples/annotations_demo/`.

Regression tests assert the real sample's `st_mtime_ns` is unchanged.

## D10 - Data reconciliation: pyarrow/pandas counts, never wc -l
Raw CSVs contain embedded newlines inside quoted review text, so `wc -l`
over-reports (e.g. SBI 90,746 shell lines vs 65,000 actual records). Every
inventory row count was re-verified against pandas row counts and reconciles
exactly to `load_summary.json` (449,379 rows, 11 banks). `wc -l` is not a
valid integrity check for these files.

## D11 - Flat 4-class head + benchmark dedup (s.24)
`src/model_registry.py` gained a `flat` task (NORMAL / OPERATIONAL_FRICTION /
TECHNICAL_RISK / FINANCIAL_RISK) and `configs/model_config.yaml` a matching
`flat:` task block with `derived: true` - `train.load_task_data` computes
`flat_label` via `evaluate.flat_label` so the flat head trains on the SAME
annotation rows as the gate+risk heads. `src/train.py` now persists the model
+ `label_map.json` in both smoke and full runs (previously smoke runs saved
nothing, so `evaluate.flat_vs_hierarchical` could never find the flat head).
`label_map.json` is co-located at `<task>/<model>/model/label_map.json` (what
`evaluate.predict_probs` reads) and at the parent dir. `benchmark_log.csv` is
deduplicated on (model, task, split, train/test size, seed), keeping the latest
repeat run - the log had 6 rows from repeated identical smoke runs.

## D12 - Figure 17 label-id fix
`figures.fig_confusion_matrices` passed string label names to
`sklearn.confusion_matrix(..., labels=...)` while the eval dumps store integer
label ids, producing an empty matrix. Now passes `list(range(len(labels)))`.
All 22 s.56 figures now generate (22/22).

