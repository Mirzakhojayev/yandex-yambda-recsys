"""Build and save the catalog and user sequences for SASRec (validation and test versions)."""
from pathlib import Path

from yambda_rec.catalog import build_catalog, load_mappings
from yambda_rec.data import TEST_START, VAL_END, VAL_START, listen_plus, load_events, split
from yambda_rec.sequences import build_sequences

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def prepare(name, train):
    catalog = build_catalog(train, *load_mappings())
    seqs = build_sequences(train, catalog)
    catalog.write_parquet(PROCESSED / f"catalog_{name}.parquet")
    seqs.write_parquet(PROCESSED / f"sequences_{name}.parquet")
    lengths = seqs["item_idx"].list.len()
    print(f"{name}: {catalog.height} tracks, {seqs.height} users, "
          f"sequence length median {lengths.median():.0f}, max {lengths.max()}")


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    lp = listen_plus(load_events())
    for name, eval_start, eval_end in (("val", VAL_START, VAL_END), ("test", TEST_START, None)):
        train, _ = split(lp, eval_start, eval_end)
        prepare(f"listen_plus_{name}", train)


if __name__ == "__main__":
    main()