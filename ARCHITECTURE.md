# Architecture

Map of the repository after the 2026-10-16 clean-up. Two models live here side by side and share nothing but the
`data/` folder: a **DeepFM rating regressor** written in this repository, and a **vendored S3Rec baseline** used for
sequential movie recommendation.

## Structure

```
Recommend_System/
├── train_deepfm.py            # entry point 1: DeepFM on the movie-rating task (run from the repo root)
├── src/
│   ├── deepfm/
│   │   ├── model.py           #   DeepFM subclass of deepctr-torch BaseModel; forward returns (y_pred, mse)
│   │   ├── dataset.py         #   TrainDataset + get_dataloader: DataFrame -> float32 tensors
│   │   └── trainer.py         #   train / eval loops, early stopping, weight saving, test-set dump
│   └── utils/
│       ├── __init__.py        #   re-exports seed, parser, metrics
│       ├── parser.py          #   argparse defaults = the hyperparameters (see Conventions)
│       ├── metrics.py         #   MAE/MSE/RMSE/R2/Adjusted-R2 + Precision/Recall/F1, on numpy
│       ├── plot.py            #   plot_hist -> results/figures/{title}_plot.png
│       └── seed.py            #   seed_everything
├── notebooks/
│   ├── build_s3rec_inputs.ipynb       # TMDB/MovieLens CSVs -> data/train_items.csv, data/Ml_item2att.json
│   └── build_movie_rating_splits.ipynb # rating splits -> data/train_val_test2.pkl (DeepFM input)
├── third_party/s3rec/         # entry points 2-4: run_pretrain.py, run_train.py, inference.py (run from this folder)
│   ├── README.md              #   upstream note: re-packaged from github.com/aHuiWang/CIKM2020-S3Rec
│   ├── models.py, modules.py  #   S3RecModel, transformer blocks
│   ├── datasets.py, trainers.py, utils.py
│   ├── preprocessing.py, sample_submission.ipynb  # baseline helpers, original competition paths
│   └── output/                #   Pretrain.pt, Finetune_full-Ml.pt, submission.csv
├── data/                      # committed inputs; data/attributes/ (raw downloads) is git-ignored
├── results/                   # prediction dumps + figures
├── weights/                   # DeepFM checkpoints
├── requirements.txt           # curated, from the imports actually used
└── requirements-freeze.txt    # original `pip freeze` of the development machine
```

Early history (January 2025) also carried per-member folders (`BHJ/`, `JJI/`, `KJH/`) and three further public
implementations — BGCN, EVCF and RecVAE — all removed before `HEAD`. What survives is the DeepFM and S3Rec line.

Source files are 18 Python modules (2,341 lines) and 3 notebooks; the largest single file is the vendored
`third_party/s3rec/utils.py` (359 lines).

## Dependencies

External (see `requirements.txt`): `torch`, `deepctr-torch` (DeepFM only), `numpy`, `pandas`, `scipy` (S3Rec sparse
rating matrices), `scikit-learn` (notebooks), `matplotlib`, `tqdm`.

Internal imports form two disjoint graphs, because the two models are run from different working directories.

| Module | Imported by | Fan-in |
|---|---|---|
| `third_party/s3rec/utils.py` | `datasets`, `trainers`, `run_pretrain`, `run_train`, `inference` | 5 |
| `third_party/s3rec/models.py` | `run_pretrain`, `run_train`, `inference` | 3 |
| `third_party/s3rec/datasets.py` | `run_pretrain`, `run_train`, `inference` | 3 |
| `third_party/s3rec/trainers.py` | `run_pretrain`, `run_train`, `inference` | 3 |
| `src/utils/metrics.py` | `src/deepfm/trainer.py`, `src/utils/__init__.py` | 2 |
| `src/utils/parser.py` | `train_deepfm.py`, `src/utils/__init__.py` | 2 |
| `src/utils/plot.py` | `src/deepfm/trainer.py` | 1 |
| `src/deepfm/{model,dataset,trainer}.py` | `train_deepfm.py` | 1 each |

`third_party/s3rec/utils.py` is the baseline's interface for everything that is not a network: sequence loading,
rating matrices, the submission writer, early stopping and the `recall_at_k` / `ndcg_k` metrics. It is also the
module with the highest coupling in the repository, which is the reason the vendored folder is left as it is.

Data dependencies, per entry point:

| Entry point | Reads | Writes |
|---|---|---|
| `train_deepfm.py` | `data/train_val_test2.pkl` (git-ignored; built by `notebooks/build_movie_rating_splits.ipynb`) | `weights/DeepFM_movie.pth`, `results/DeepFM_movie.csv`, `results/figures/{Loss,R2}_plot.png` |
| `third_party/s3rec/run_pretrain.py` | `../../data/train_items.csv`, `../../data/Ml_item2att.json` | `output/Pretrain.pt` |
| `third_party/s3rec/run_train.py` | same, plus `output/Pretrain.pt` | `output/Finetune_full-Ml.pt` (RECALL@k / NDCG@k go to stdout only: `args.log_file` is set but never written) |
| `third_party/s3rec/inference.py` | same, plus `output/Finetune_full-Ml.pt` | `output/submission.csv` |

## Conventions

- **Working directory is part of the interface.** `train_deepfm.py` uses paths relative to the repository root
  (`data/`, `weights/`, `results/`); the S3Rec scripts use flat imports (`from utils import ...`) and `../../data/`,
  so they must be run from `third_party/s3rec/`. Notebooks use `../data/`.
- **Hyperparameters live in defaults, not on the command line.** `src/utils/parser.py` sets `sys.argv = ['']` before
  parsing, so flags passed to `train_deepfm.py` are ignored; change the `add_argument` defaults instead. The S3Rec
  scripts are the opposite: ordinary `argparse`, every knob is a flag.
- **Metrics are numpy-first.** Everything in `src/utils/metrics.py` goes through `ensure_tensor_array`, so tensors,
  lists and scalars are all accepted; the trainer therefore reports `R2` on detached numpy arrays while the loss stays
  a tensor.
- `src/deepfm/trainer.py` imports `src.utils.plot` and `src.utils.metrics` with `*`; the names it relies on are
  `plot_hist`, `R2` and `ensure_tensor_array`.
- Checkpoints, prediction dumps and figures are written, never read back, except `weights/DeepFM_movie.pth`, which
  `Trainer.test` reloads before dumping predictions.
- Comments are in Korean, identifiers and file names in English; source files are `lower_snake_case`, vendored files
  keep their upstream names so that a diff against upstream stays readable.
- **No tests.** There is no test suite and no CI in this repository.

## Impact

- **Changing a data location** touches three places independently: `train_deepfm.py` (`file_path`), the `--data_dir`
  default in the three S3Rec scripts, and the notebook cells. There is no shared config.
- **Adding a model** of the DeepFM kind means a new `src/<model>/` package plus a new root entry point; `src/utils/`
  is already model-agnostic. Nothing in `src/` imports `third_party/`, and nothing in `third_party/s3rec/` imports
  `src/` — keep it that way, so the vendored folder can be re-synced with upstream.
- **Complexity, as a proxy for risk.** There are no tests, so a real CRAP score (which needs coverage) cannot be
  computed here; the ranking below is `cc² + cc`, the CRAP value a function would have at zero coverage, from
  `python -m lizard`. 105 functions were measured; the worst are in the vendored code.

  | Function | CC | NLOC | CRAP at 0 coverage |
  |---|---|---|---|
  | `third_party/s3rec/trainers.py::iteration` | 9 | 59 | 90 |
  | `third_party/s3rec/run_train.py::main` | 8 | 108 | 72 |
  | `third_party/s3rec/datasets.py::__getitem__` (SASRecDataset) | 7 | 85 | 56 |
  | `third_party/s3rec/datasets.py::__getitem__` (PretrainDataset) | 6 | 53 | 42 |
  | `src/deepfm/trainer.py::train_and_eval` | 6 | 49 | 42 |
  | `src/utils/metrics.py::ensure_tensor_array` | 6 | 8 | 42 |
  | `src/deepfm/model.py::forward` | 5 | 19 | 30 |

  The only own-code hot spots are `train_and_eval` (loop, early stopping and the optional backbone-reload branch in
  one function) and `ensure_tensor_array` (six type branches, used by every metric). Both are the first candidates
  if tests are ever added.
- **Stored artifacts are older than the code.** `results/DeepFM_movie.csv` holds continuous predictions (30,498
  distinct values over 32,210 rows), while `src/deepfm/model.py` now rounds its output to half-stars, and
  `results/DeepFM_CTR.csv` has a `Probability` column that no current code path produces. Both files, and the two
  `weights/*.pth`, come from earlier revisions (the CTR notebooks were deleted in February 2025). Do not read
  performance off them.
- **Untouched on purpose:** `third_party/s3rec/preprocessing.py` and `third_party/s3rec/sample_submission.ipynb`
  still point at the upstream competition layout (`../data/train/*.tsv|csv`), which this repository does not contain.
  `notebooks/build_s3rec_inputs.ipynb` is the replacement actually used here.
