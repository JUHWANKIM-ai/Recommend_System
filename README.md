# Recommend_System (recommender-system project, 2025)

[![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)](requirements.txt)
[![deepctr-torch](https://img.shields.io/badge/deepctr--torch-0.2.9-1f6feb)](requirements.txt)
[![Data](https://img.shields.io/badge/data-MovieLens%20%2B%20TMDB-FFD21E)](#data)

Two recommenders over the same movie data, kept in one repository. **DeepFM** predicts the rating a user gives a
movie from two sparse fields (user, movie) and five movie attributes, as a regression on the half-star scale;
**S3Rec** predicts the next items of a user's viewing sequence and writes a top-10 recommendation per user. The
DeepFM side (entry point, training loop, metrics, preprocessing notebooks) is written here; the S3Rec side is a
public baseline, vendored under `third_party/` and left close to upstream — see [Code provenance](#code-provenance).

The repository was worked on between January and February 2025 as a small team project (early history has per-member
folders) and three further public implementations — BGCN, EVCF and RecVAE — were tried and removed before the final
commit. A second, smaller task is visible only in its data and artifacts: click-through prediction on a web-article
log (`data/view_log.csv`, `data/train_val_test.pkl`); the code for it was deleted in February 2025.

## Pipeline

```
DeepFM: rating regression
  raw (not redistributed)                notebooks/                        data/
  ratings.csv, movies_metadata.csv  ─►  build_s3rec_inputs.ipynb      ─►  Ml_item2att.json  (5 attributes / movie)
  links.csv  (Kaggle "The Movies           genre label-encoded,            train_items.csv   (user, item pairs)
  Dataset" + MovieLens links)              release_date as days to
                                           2021-12-16, MinMax scaling
                                      ─►  build_movie_rating_splits   ─►  train_val_test2.pkl
                                           .ipynb                           balanced: 300,000 rows per
                                                                            half-star level (0.5 ... 5.0)
                                                            │
                                                            ▼
  train_deepfm.py  ─►  SparseFeat(userId, movieId, emb 50) + DenseFeat(genres, release_date,
                       popularity, runtime, revenue)
                       │
                       ├─ linear part: dense features only
                       ├─ FM part:     sparse embeddings
                       └─ DNN part:    (256, 128), relu            src/deepfm/model.py
                       │
                       y_pred = round(clamp(out, 0.5, 5) * 2) / 2   (half-star grid)
                       loss   = MSE(y_pred, rating)
                       │
                       ▼
  src/deepfm/trainer.py  Adam(lr 1e-3) + ReduceLROnPlateau, batch 32, up to 500 epochs,
                         early stopping after 10 epochs without a better val loss, R2 reported
                       │
                       ├─► weights/DeepFM_movie.pth        (best val loss)
                       ├─► results/figures/{Loss,R2}_plot.png
                       └─► results/DeepFM_movie.csv        (user_id, movie_id, y_pred, gt)

S3Rec: sequential recommendation                                   third_party/s3rec/
  data/train_items.csv ─► user sequences (max length 50)
  data/Ml_item2att.json ─► item attributes
            │
            ├─ run_pretrain.py   self-supervised: AAP + MIP + MAP + SP   ─►  output/Pretrain.pt
            │                    2-layer transformer, hidden 64, 2 heads
            ├─ run_train.py      fine-tune on next-item prediction,      ─►  output/Finetune_full-Ml.pt
            │                    early stopping on NDCG@10,
            │                    RECALL@{5,10} / NDCG@{5,10} to stdout
            └─ inference.py      top-10 per user                         ─►  output/submission.csv
```

## Stored artifacts

No metric is quoted in this README: the repository keeps no training log, and the committed notebooks hold no
recall@k, NDCG or loss output. The prediction files that are committed were produced by **earlier revisions** of the
code, so numbers read off them would not describe what `HEAD` does — `results/DeepFM_movie.csv` has 30,498 distinct
continuous predictions over 32,210 rows while `src/deepfm/model.py` now rounds to the half-star grid, and
`results/DeepFM_CTR.csv` has a `Probability` column that no current code path writes. What is in the repository:

| Artifact | Contents |
|---|---|
| `results/DeepFM_movie.csv` | 32,210 test rows: `user_id`, `movie_id`, `y_pred`, `gt` |
| `results/DeepFM_CTR.csv` | 6,366 test rows of the article click task: `Probability`, `Predictions`, `Ground Truths` |
| `results/figures/Loss_plot.png`, `F1_plot.png` | train/val curves from those runs |
| `weights/DeepFM_movie.pth`, `DeepFM_CTR.pth` | DeepFM checkpoints (48 MB, 1.8 MB) |
| `third_party/s3rec/output/Pretrain.pt`, `Finetune_full-Ml.pt` | S3Rec checkpoints (44 MB each) |
| `third_party/s3rec/output/submission.csv` | 2,299,110 rows: 10 recommended items for each of 229,911 users |

## Code provenance

| Path | Origin |
|---|---|
| `train_deepfm.py`, `src/deepfm/{dataset,trainer}.py`, `src/utils/`, `notebooks/` | written in this project |
| `src/deepfm/model.py` | the DeepFM of [deepctr-torch](https://github.com/shenweichen/DeepCTR-Torch) (its `BaseModel`, `FM` and `DNN` are imported), with the head changed to a clamped and rounded half-star output that returns the MSE loss |
| `third_party/s3rec/` | movie-recommendation baseline re-packaged from [aHuiWang/CIKM2020-S3Rec](https://github.com/aHuiWang/CIKM2020-S3Rec), per `third_party/s3rec/README.md`; adapted to this repository's data layout (`--data_dir ../../data/`) and with `--using_pretrain` defaulting to `True` (a `store_true` flag, so pretraining can no longer be switched off from the command line — edit the default instead) |
| `third_party/s3rec/preprocessing.py`, `sample_submission.ipynb` | baseline helpers, left untouched: they still read the upstream competition layout (`../data/train/`), which this repository does not contain |

## Data

`data/` holds what the code reads; the raw downloads it is built from are not redistributed and `data/attributes/`
is git-ignored.

| File | Contents |
|---|---|
| `data/train_items.csv` | 2,594,618 `(user, item)` interactions, 229,911 users, 25,364 items — the S3Rec input |
| `data/Ml_item2att.json` | 42,721 movies × 5 attributes (genre code, scaled release_date, popularity, runtime, revenue) |
| `data/train_val_test.pkl` | the article click task: 48,693 / 8,593 / 6,366 rows over 9 fields |
| `data/view_log.csv` | 42,717 article views, 1,415 users, 2,879 articles, with region and country |
| `data/article_info.csv` | 3,008 articles: title, full text, format, language |
| `data/sample_submission.csv` | 1,415 users × 5 articles |
| `data/ncf_data.csv` | 43,019 scraped `(movie, user, rating)` triples |
| `data/users/*.csv` | 6,994 Korean movie reviews scraped for 4 reviewers (title, text, rating, likes) |
| `data/attributes/` | *git-ignored*: `ratings.csv`, `movies_metadata.csv`, `links.csv` from the Kaggle **The Movies Dataset**, and the intermediate pickles the notebooks write |

## Repository layout

| Path | Contents |
|---|---|
| `train_deepfm.py` | entry point of the DeepFM rating model; run from the repository root |
| `src/deepfm/` | `model.py` (network and loss), `dataset.py` (tensors and dataloaders), `trainer.py` (loops, early stopping, dumps) |
| `src/utils/` | `parser.py` (hyperparameter defaults), `metrics.py` (MAE/MSE/RMSE/R2, precision/recall/F1), `plot.py`, `seed.py` |
| `notebooks/` | `build_s3rec_inputs.ipynb`, `build_movie_rating_splits.ipynb` — everything in `data/` is regenerated from these |
| `third_party/s3rec/` | vendored S3Rec baseline: `run_pretrain.py`, `run_train.py`, `inference.py` and its own `utils.py`, run from inside this folder |
| `data/`, `results/`, `weights/` | inputs, prediction dumps and figures, checkpoints |
| `requirements.txt` | the packages actually imported; `requirements-freeze.txt` is the original `pip freeze` |
| `ARCHITECTURE.md` | import graph, fan-in, conventions, complexity ranking |

## Get Started

**1. Install.** Developed on Python 3.11 with CUDA (`device='cuda'` is the default and `train_deepfm.py` builds the
model on `cuda:0`).

```bash
pip install -r requirements.txt
```

The vendored baseline pins older versions (`third_party/s3rec/requirements.txt`, torch 1.10.2); it also runs under
the versions above.

**2. Data.** Download the Kaggle [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
into `data/attributes/` (`ratings.csv`, `movies_metadata.csv`, `links.csv`), then run the notebooks from
`notebooks/`:

```bash
jupyter notebook notebooks/build_s3rec_inputs.ipynb         # -> data/train_items.csv, data/Ml_item2att.json
jupyter notebook notebooks/build_movie_rating_splits.ipynb  # -> data/train_val_test2.pkl
```

`data/train_items.csv` and `data/Ml_item2att.json` are already committed, so the S3Rec scripts run without this step.
`data/train_val_test2.pkl`, which DeepFM needs, is not committed and has to be built.

**3. Train DeepFM** (from the repository root):

```bash
python train_deepfm.py
```

Hyperparameters are the defaults in `src/utils/parser.py` (lr 1e-3, batch 32, 500 epochs, patience 10). Command-line
flags have no effect here: the parser sets `sys.argv = ['']` before parsing. Outputs go to `weights/`, `results/` and
`results/figures/`.

**4. Train S3Rec** (from inside the vendored folder, which is where its flat imports and `output/` resolve):

```bash
cd third_party/s3rec
python run_pretrain.py                     # -> output/Pretrain.pt
python run_train.py                        # fine-tune; loads output/Pretrain.pt by default
python inference.py                        # -> output/submission.csv, top 10 items per user
```

RECALL@{5,10} and NDCG@{5,10} are printed each epoch; nothing writes them to a file.
