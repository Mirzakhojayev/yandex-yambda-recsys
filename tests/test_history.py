import polars as pl

from yambda_rec.history import fill_with, user_history

HOUR = 3600


def make_train(rows):
    return pl.LazyFrame(rows, schema=["uid", "item_id", "timestamp"], orient="row")


def test_recent_play_beats_old_plays():
    # user 1: item 10 played twice long ago, item 20 played once at the very end
    train = make_train([(1, 10, 0), (1, 10, HOUR), (1, 20, 100 * HOUR)])
    # fast decay: the two old plays weigh ~0.9**100 each, the recent one weighs 1
    assert user_history(train, hours=1)[1] == [20, 10]
    # almost no decay: two plays beat one
    assert user_history(train, hours=1e6)[1] == [10, 20]


def test_underflow_ties_broken_by_most_recent_play():
    train = make_train([(1, 10, 0), (1, 30, 50 * HOUR), (1, 20, 100 * HOUR)])
    # tiny `hours`: every play except the last underflows to weight 0.0
    assert user_history(train, hours=0.001)[1] == [20, 30, 10]


def test_k_limits_list_length():
    train = make_train([(1, item, item) for item in range(10)])
    assert len(user_history(train, hours=1, k=3)[1]) == 3


def test_fill_pads_with_fallback_without_duplicates():
    out = fill_with({1: [5, 6]}, fallback=[6, 7, 8, 9], users=[1, 2], k=4)
    assert out[1] == [5, 6, 7, 8]
    assert out[2] == [6, 7, 8, 9]  # user with no history gets the fallback only

def test_full_ties_broken_by_item_id():
    # same timestamp, same score: lower item_id first, every time
    train = make_train([(1, 30, 0), (1, 10, 0), (1, 20, 0)])
    assert user_history(train, hours=1)[1] == [10, 20, 30]