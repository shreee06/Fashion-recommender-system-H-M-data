# Fashion Recommendation Engine (H&M)

A portfolio project that builds a fashion recommendation engine in Python on the
H&M Personalized Fashion Recommendations dataset: candidate generation, ranking,
offline evaluation and a small Streamlit demo app. The project is under construction;
this README is filled in only from real runs.

## Data

The data comes from the Kaggle competition
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).
It is used for educational and non-commercial purposes under the competition rules.

**The data is NOT included in this repository** (the competition rules forbid
redistribution, so no raw files or samples are committed). To reproduce this project,
accept the competition rules on Kaggle and download the files yourself into `data/raw/`.

- `data/raw/`: raw Kaggle files (git-ignored, user-provided)
- `data/processed/`: derived files produced by the scripts (git-ignored)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Results

Results: to be filled from real runs.
