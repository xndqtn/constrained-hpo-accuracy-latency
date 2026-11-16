# Constrained Hyperparameter Optimization: Accuracy–Latency Trade-off

> Bài tập lớn cá nhân môn Nhập môn Tối ưu hóa — Random Forest, Breast Cancer Wisconsin Diagnostic.

## Problem and formulation

Tìm cấu hình RF x = (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features) để cực tiểu hóa 1 − F1(validation), dưới ràng buộc median latency của 1.000 lần dự đoán ≤ 30 ms. Với các nghiệm khả thi, chọn F1(validation) lớn nhất rồi ưu tiên latency thấp; nếu không có nghiệm khả thi, chọn nghiệm vi phạm ít nhất.

| Tham số | Không gian chung của Random Search và TPE |
|---|---|
| n_estimators | integer 10–300 |
| max_depth | integer 2–25 |
| min_samples_split | integer 2–20 |
| min_samples_leaf | integer 1–15 |
| max_features | float 0.2–1.0 |

## Dataset and methodology

- Dataset: Breast Cancer Wisconsin (Diagnostic), 569 records, 30 features; provided by scikit-learn and originally from UCI.
- Stratified split (seed 42): train 397, validation 86, untouched test 86. Random Forest does not require scaling.
- Methods: scikit-learn default RF, 30-trial uniform Random Search, 30-trial actual Optuna TPE, same space and validation objective.
- One shared fixed 1,000-row validation-derived latency batch, median of 3 timings per trial, plus warm-up. Fixed random_state and n_jobs=1.
- Constrained loss for TPE = (1 − F1) when feasible; otherwise 1 + latency violation / budget + (1 − F1).
- Final selection depends ONLY on validation F1/latency, never test labels. Once selected, the three models are evaluated on test once in the final step.

## Results from the included run

Command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/pipeline.py --trials 30 --tau_max 30 --seed 42 --repeats 3`

| Method | Test F1 | Test accuracy | Median latency (ms/1000) | Feasible |
|---|---:|---:|---:|---|
| Default Baseline | 0.9434 | 0.9302 | 9.553 | Yes |
| Random Search | 0.9533 | 0.9419 | 12.866 | Yes |
| Optuna TPE | 0.9423 | 0.9302 | 1.207 | Yes |

**Interpretation:** Random Search achieved the highest held-out test F1 in this run, whereas Optuna TPE produced the lowest measured latency. Under 30 ms, all 60 search trials were feasible; therefore the constraint was not binding on this machine. Do not interpret these single-seed results as evidence that TPE always outperforms Random Search.

### Selected settings (selected on validation only)

| Field | Random Search | Optuna TPE |
|---|---:|---:|
| trial | 21 | 17 |
| n_estimators | 137 | 11 |
| max_depth | 18 | 14 |
| min_samples_split | 14 | 13 |
| min_samples_leaf | 8 | 7 |
| max_features | 0.6521888851849511 | 0.8682198938514627 |

![Search history](figures/convergence_history.png)

![Empirical F1–latency trade-off](figures/pareto_accuracy_latency.png)

## How to run

Python 3.10+ (experiment generated on Python 3.13.5):

```bash
python -m pip install -r requirements.txt
python src/pipeline.py --trials 30 --tau_max 30 --seed 42 --repeats 3
```

Output artifacts are overwritten in `results/` and `figures/`. Logs include every trial and the automatically selected final parameters. Experiment metadata records the runtime platform, versions, seed and selection. Actual timings vary across hardware.

## Repository structure

```text
src/pipeline.py
data/README.md
results/trials_log.csv
results/test_benchmark_summary.csv
results/experiment_metadata.json
figures/convergence_history.png
figures/pareto_accuracy_latency.png
report/bao_cao_do_an_toi_uu_hoa.docx
report/bao_cao_do_an_toi_uu_hoa.pdf
requirements.txt
README.md
.gitignore
```

## Limitations and deployment decision

- Only one seed / one split was run; repeat across several seeds and use mean ± standard deviation before claiming method superiority.
- Timing varies with CPU scheduling; 3 repetitions per search trial are limited and the batch is not a clinical workload.
- All configurations passed the 30 ms constraint in this environment; future work should prospectively specify a tighter, hardware-relevant budget and rerun optimization.
- Breast Cancer Wisconsin is a didactic tabular dataset; the project is **not validated for medical or real-time clinical deployment**.
- For a latency-prioritized scenario on the measured machine, TPE is the faster option; for F1-prioritized selection on the untouched test set, Random Search performed better. A deployment policy must be chosen before looking at test outcomes.

## Data source

Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). Breast Cancer Wisconsin (Diagnostic) dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

## AI assistance

AI was used to help review code, refine experimental logging, and draft documentation. Users must inspect and reproduce the results themselves; numbers in the report were generated from CSV/JSON files in this repository.

## GitHub checklist

Publish the repository as public; add 5 meaningful commits over the actual development process (do not fabricate history); add your URL to the report and LMS, create release v1.0 if appropriate.