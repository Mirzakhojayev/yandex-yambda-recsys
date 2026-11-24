"""Score the baselines on Listen+ and Like and print a results table."""
import polars as pl

from yambda_rec.baselines import decay_pop, most_pop
from yambda_rec.data import TEST_START, VAL_END, VAL_START, likes, listen_plus, load_events, split
from yambda_rec.evaluate import build_targets, num_items, report, new_tracks_only, seen_items, exclude_seen
from yambda_rec.history import fill_with, user_history
from yambda_rec.metrics import evaluate

POP_HOURS_GRID = [0.5, 1, 2, 3, 6, 12, 24]  # official DecayPop search space
HISTORY_HOURS_GRID = [1, 6, 24, 72, 168, 720, 8760]  # 1 hour ... 1 year
LONG_K = 1000


def same_for_all(top: list[int], users) -> dict:
    return {uid: top for uid in users}


def pop_recs(train: pl.LazyFrame, users, hours: float) -> dict:
    return same_for_all(decay_pop(train, hours), users)


def history_recs(train: pl.LazyFrame, users, hours: float, pop_hours: float) -> dict:
    """User's own tracks first, padded to 100 with DecayPop."""
    fallback = decay_pop(train, pop_hours, k=200)
    return fill_with(user_history(train, hours), fallback, users)


def tune(label: str, df: pl.LazyFrame, grid: list, make_recs) -> float:
    """Pick the grid value with the best NDCG@100 on the validation day, never the test day."""
    train, val = split(df, VAL_START, VAL_END)
    targets = build_targets(val)
    n_items = num_items(train)
    print(f"Tuning {label}")
    scores = {}
    for value in grid:
        recs = make_recs(train, targets, value)
        scores[value] = evaluate(recs, targets, 100, n_items)["ndcg@100"]
        print(f"  {value:>6}: val ndcg@100 = {scores[value]:.4f}")
    return max(scores, key=scores.get)


def run(target_name: str, df: pl.LazyFrame) -> list[dict]:
    pop_hours = tune(f"DecayPop / {target_name}", df, POP_HOURS_GRID, pop_recs)
    hist_hours = tune(
        f"History / {target_name}", df, HISTORY_HOURS_GRID,
        lambda train, users, hours: history_recs(train, users, hours, pop_hours),
    )

    train, test = split(df, TEST_START)
    n_items = num_items(train)

    targets = build_targets(test)
    rows = [
        report(f"MostPop / {target_name} / all", same_for_all(most_pop(train), targets),
               targets, n_items),
        report(f"DecayPop {pop_hours}h / {target_name} / all", pop_recs(train, targets, pop_hours),
               targets, n_items),
        report(f"History {hist_hours}h / {target_name} / all",
               history_recs(train, targets, hist_hours, pop_hours), targets, n_items),
    ]

    # New tracks only: repeats removed from the targets AND heard tracks removed from the recs.
    # History is skipped: with the user's own tracks removed it is just DecayPop.
    new_targets = build_targets(new_tracks_only(test, train))
    seen = seen_items(train)
    for name, top in (
        ("MostPop", most_pop(train, k=LONG_K)),
        (f"DecayPop {pop_hours}h", decay_pop(train, pop_hours, k=LONG_K)),
    ):
        recs = exclude_seen(same_for_all(top, new_targets), seen)
        rows.append(report(f"{name} / {target_name} / new only", recs, new_targets, n_items))
    return rows


def main():
    events = load_events()
    rows = run("Listen+", listen_plus(events)) + run("Like", likes(events))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=60, float_precision=4, tbl_width_chars=200):
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
