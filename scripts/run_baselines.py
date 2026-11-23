"""Score the baselines on Listen+ and Like and print a results table."""
import polars as pl

from yambda_rec.baselines import decay_pop, most_pop
from yambda_rec.data import TEST_START, VAL_END, VAL_START, likes, listen_plus, load_events, split
from yambda_rec.evaluate import build_targets, num_items, report
from yambda_rec.metrics import evaluate

HOURS_GRID = [0.5, 1, 2, 3, 6, 12, 24]  # official search space


def same_for_all(top: list[int], targets: dict) -> dict:
    return {uid: top for uid in targets}


def tune_decay_pop(df: pl.LazyFrame) -> float:
    """Pick `hours` on the validation day, never on the test day."""
    train, val = split(df, VAL_START, VAL_END)
    targets = build_targets(val)
    n_items = num_items(train)
    best_hours, best_ndcg = None, -1.0
    for hours in HOURS_GRID:
        recs = same_for_all(decay_pop(train, hours), targets)
        ndcg = evaluate(recs, targets, 100, n_items)["ndcg@100"]
        print(f"  hours={hours:>4}: val ndcg@100 = {ndcg:.4f}")
        if ndcg > best_ndcg:
            best_hours, best_ndcg = hours, ndcg
    return best_hours


def run(target_name: str, df: pl.LazyFrame) -> list[dict]:
    print(f"Tuning DecayPop on {target_name}")
    hours = tune_decay_pop(df)

    train, test = split(df, TEST_START)
    targets = build_targets(test)
    n_items = num_items(train)
    return [
        report(f"MostPop / {target_name}", same_for_all(most_pop(train), targets), targets, n_items),
        report(f"DecayPop {hours}h / {target_name}", same_for_all(decay_pop(train, hours), targets),
               targets, n_items),
    ]


def main():
    events = load_events()
    rows = run("Listen+", listen_plus(events)) + run("Like", likes(events))
    with pl.Config(tbl_cols=-1, float_precision=4, tbl_width_chars=200):
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()