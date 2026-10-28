# Steam-to-PlayStation Game Recommender

**Cross-Platform Game Recommendation System**  
*Learning from Steam ownership and playtime data, and evaluating on PlayStation users.*

A machine learning term project (Team 10, 2026) investigating whether game recommendation patterns learned from Steam users can transfer to PlayStation users. We compare Item-Based Collaborative Filtering and Implicit Alternating Least Squares (ALS), each trained with binary game ownership or playtime-weighted interactions, against a popularity-based baseline.

> **Project status:** Environment and experiment design prepared. Dataset preprocessing, training, and evaluations are pending. No experimental results are claimed here.

## Research Questions

1. Do personalized recommenders outperform a popularity-based baseline?
2. Does Item-Based CF or Implicit ALS perform better for PlayStation users?
3. Does incorporating Steam playtime improve performance?
4. Is the effect of playtime different between the algorithms?
5. How does performance compare between Steam validation and PlayStation testing?

## Experimental Design

| ID | Method | Steam training signal |
|---|---|---|
| Baseline | Game popularity | Frequency of Steam game ownership |
| A-1 | Item-Based CF | Binary ownership |
| A-2 | Item-Based CF | Log-transformed playtime weighting |
| B-1 | Implicit ALS | Binary ownership confidence |
| B-2 | Implicit ALS | Playtime-dependent confidence |

**Training and external test are intentionally kept separate.** Steam ownership/playtime records are used for training and tuning. PlayStation records are used **only** for external evaluation, never for recommender training or hyperparameter selection.

For the PlayStation evaluation, each eligible user's matched-game library is separated into *observed games* (model input) and *held-out games* (ground truth). All five configurations share the same user splits, candidates, and held-out labels. PlayStation playtime is not required.

### Data flow

```text
Steam games metadata ──┐                     ┌── PlayStation games metadata
                       └── Title matching ───┘
                             │
                     Shared game ID map
                       /              \
       Steam user-game interactions   PlayStation user libraries
           │                                  │
      Train / Validation                 Observed / Held-out
           │                                  │
  Item CF / ALS / Popularity          External test inputs + ground truth
           └──────────────────────────────────┘
                            │
                Recall@10 / NDCG@10 / Hit Rate@10
```

Game metadata is joined to interaction records **within each platform using that platform's native game IDs**. Shared titles receive a unified ID for common candidate indexing. Steam and PlayStation user IDs are never linked, and their interaction records are never concatenated into a joint training set.

## Datasets

| Dataset | Role | Main files |
|---|---|---|
| [Steam Game Ownership and User Friendships](https://huggingface.co/datasets/steamgamerecommender/data_files_public) | Train/validation | `games.csv`, `users_games.csv` (across available dataset subfolders) |
| [Gaming Profiles 2025 (Steam, PlayStation, Xbox)](https://www.kaggle.com/datasets/artyomkruglov/gaming-profiles-2025-steam-playstation-xbox) | PlayStation external test | PlayStation `games.csv`, `purchased_games.csv` |

The filenames above follow the project proposal; verify actual downloaded paths and column names before processing. The Steam Hugging Face repository contains separate CSV types and subfolders; its combined dataset preview can fail because the CSV types have different schemas. **Download the individual files** instead of interpreting the combined viewer as one dataset.

Do **not** commit raw datasets or large derived interaction matrices to Git. Follow each dataset's license and distribution conditions.

### Expected local data layout

```text
data/
  raw/
    steam/                 # Steam folder(s) containing games.csv, users_games.csv
    playstation/           # PlayStation games.csv, purchased_games.csv
  interim/                 # Temporary mapping / audit outputs (Git-ignored)
  processed/               # Prepared train/val/test data (Git-ignored)
```

Steam dataset files may be organized into subfolders (e.g. numbered partitions). Keep their original paths if convenient; the inspection script searches recursively.

## Setup

Recommended: **Python 3.11**, using a fresh Conda environment. From the project root:

```bash
conda create -n steam-ps-rec python=3.11 -y
conda activate steam-ps-rec
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

For notebooks, select the `steam-ps-rec` Python kernel. CPU-only training is adequate to begin; GPU support is optional, not required.

### Inspect downloaded CSV schemas

Once source CSV files have been placed in `data/raw/`:

```bash
python scripts/inspect_raw_data.py
```

The script lists files, their column names, and their sizes, and samples the first rows **without loading entire datasets**. It does not modify raw data. Preprocessing scripts will be added after actual file schemas and ambiguous game-title matches are audited.

## Planned Preprocessing

1. **Schema audit:** Confirm native game/user ID fields, title fields, playtime units, data types, duplicates, and missing values.
2. **Within-platform mapping:** Associate Steam ownership with Steam game metadata, and PlayStation libraries with PlayStation metadata. Expand nested PlayStation libraries if required by the actual schema.
3. **Cross-platform title matching:** Normalize titles conservatively, create common game IDs, and review collisions/ambiguous titles. Do not blindly merge editions or unrelated games with similar names.
4. **Interaction cleaning:** Validate IDs and duplicated user-game pairs. Preserve *zero-playtime ownership*, which is valid binary feedback. Examine unusually high playtime before deciding on any clipping strategy; fit preprocessing decisions using Steam training data only.
5. **Common catalog:** Keep a reproducible set of matched games and a consistent ranking candidate pool across models.
6. **Steam validation:** Hold out a subset of Steam users for tuning; for each validation user, split owned shared games into observed and hidden sets. Avoid training on validation targets.
7. **PlayStation external test:** For eligible users, split owned shared games into observed and held-out sets using a reproducible random seed. Do not use PS ownership to train or tune the models.
8. **Model-specific matrices:** Prepare binary user-item matrices for A-1/B-1 and playtime-weighted matrices for A-2/B-2. Popularity uses game ownership counts from training only.

The exact minimum number of owned shared games, holdout ratio, random seeds, and playtime weighting formula will be documented before running the experiments. In the proposal's coverage study, a minimum of five matched games was used for feasibility analysis.

## Recommendation and Evaluation

- **Item-Based CF:** Learn Steam item-to-item cosine similarity, then score games using each test user's observed game history.
- **Implicit ALS:** Learn latent factors from Steam implicit interactions; for unseen Steam validation or PlayStation users, perform a **fold-in** step using observed game ownership while freezing trained game factors.
- **Popularity baseline:** Rank games by Steam training-set ownership frequency, excluding observed games.
- **Shared evaluation:** Exclude observed games from Top-K recommendations and evaluate using **Recall@10**, **NDCG@10**, and **Hit Rate@10** against held-out games.

## Repository Structure

```text
steam-to-playstation-recommender/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── check_environment.py
│   └── inspect_raw_data.py
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── src/
│   ├── preprocessing/
│   ├── models/
│   └── evaluation/
└── results/
```

## Notes on Reproducibility

- Use a fixed random seed and save the matched-game catalog plus train/validation/test split identifiers locally.
- Learn transformations, popularity statistics, and hyperparameters **without access to held-out labels or PlayStation test results**.
- Keep all experimental conditions fixed except the algorithm and training signal under investigation.
- Report game matching coverage and the number of eligible Steam/PlayStation users after filtering.

## References

- Steam dataset: https://huggingface.co/datasets/steamgamerecommender/data_files_public
- PlayStation dataset: https://www.kaggle.com/datasets/artyomkruglov/gaming-profiles-2025-steam-playstation-xbox
- `implicit` ALS implementation: https://pypi.org/project/implicit/

## Team

Machine Learning Term Project — Team 10 (2026).
