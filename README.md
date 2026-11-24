# Music recommender on Yambda-50M

A two-stage music recommender (candidate retrieval, then ranking) built on
[Yambda-50M](https://huggingface.co/datasets/yandex/yambda), the public Yandex Music dataset.
The goal is to beat the baselines from the [Yambda paper](https://arxiv.org/abs/2505.22238)
under its official evaluation protocol, and then publish the code, models and a short write-up.

> **Status: work in progress.** The data pipeline, the evaluation and the popularity baselines are done.
> The retrieval and ranking models are next.

## Dataset

Yambda-50M has about 10,000 users, 934,000 tracks and 46.5M listens, plus likes, dislikes,
unlikes and undislikes, over roughly 300 days. Every event has a flag that says whether it
came from the user's own action or from Yandex's recommendations. The dataset is released
under Apache 2.0.

We use one file: `flat/50m/multi_event.parquet`. It is split by time first, and anything
per user is built from the training part only, so the test day can't leak into training.

## Evaluation

We follow the paper's protocol, with the constants taken from the official `benchmarks/` code:

- **Global temporal split:** the last 24 hours of data are the test day. Training ends
  30 minutes before the test day starts, and the cut time is the same for every user.
  Hyperparameters are tuned on a validation day placed just before the test day, with the
  same 30-minute gap.
- **Targets:** **Listen+** (a listen with at least 50% of the track played) and **Like**.
- **Metrics:** NDCG, Recall and Coverage at 10 and 100.

Two things we found while reproducing the paper:

1. **The paper's tables use slightly different formulas from the code that's published now.**
   - NDCG: the paper takes the sum of DCG over all users and divides it by the sum of IDCG.
     The current code averages a per-user NDCG.
   - Recall: the paper divides hits by the number of targets. The current code divides by
     `min(k, number of targets)`.

   With the paper's formulas, our MostPop and DecayPop reproduce the paper exactly.
   `metrics.evaluate(..., paper_compat=True)` is the default, so our numbers can be compared
   directly with the paper. `paper_compat=False` gives the current official formulas.
2. **Most Listen+ test targets are repeats.** A large majority of the tracks users play on the
   test day are tracks they already played during training. The official protocol counts these
   as hits, so it mostly measures how well a model predicts replays. We plan to add a
   **new tracks only** evaluation, which drops repeats from the targets, to measure discovery
   separately.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"

python scripts/download_data.py    # ~384 MB into data/raw/
python scripts/run_baselines.py    # tunes on the validation day, prints the test-day table
pytest                             # metric and model unit tests
```

## Project layout

```
src/yambda_rec/
    data.py        loading, Listen+ / like filters, temporal split
    metrics.py     NDCG, Recall, Coverage (paper-compatible and current-official)
    evaluate.py    test split -> targets, one results row per model
    baselines.py   MostPop, DecayPop
    history.py     per-user history baseline (the user's own tracks, recent plays weighted higher)
scripts/
    download_data.py
    run_baselines.py
notebooks/
    explore.ipynb  first look at the data and a check of the split
tests/
```

## References

- Ploshkin et al., *Yambda-5B — A Large-Scale Multi-modal Dataset for Ranking and Retrieval*, 2025.
  [arXiv:2505.22238](https://arxiv.org/abs/2505.22238)
- Dataset and official benchmark code: [huggingface.co/datasets/yandex/yambda](https://huggingface.co/datasets/yandex/yambda)
