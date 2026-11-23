"""Score the baselines on Listen+ and Like and print a results table."""
import polars as pl

from yambda_rec.baselines import most_pop
from yambda_rec.data import TEST_START, likes, listen_plus, load_events, split
from yambda_rec.evaluate import build_targets, num_items, report


def run(target_name: str, df: pl.LazyFrame) -> list[dict]:
    train, test = split(df, TEST_START)
    targets = build_targets(test)
    n_items = num_items(train)

    top = most_pop(train)
    recs = {uid: top for uid in targets}
    return [report(f"MostPop / {target_name}", recs, targets, n_items)]


def main():
    events = load_events()
    rows = run("Listen+", listen_plus(events)) + run("Like", likes(events))
    with pl.Config(tbl_cols=-1, float_precision=4):
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()