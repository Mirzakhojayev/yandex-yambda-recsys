import math

import pytest

from yambda_rec.metrics import evaluate


def test_hand_example():
    # targets A=1, B=2; model ranks C=3, A=1, D=4 (your NDCG exercise)
    m = evaluate(recs={0: [3, 1, 4]}, targets={0: [1, 2]}, k=3, num_items=10)
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert m["ndcg@3"] == pytest.approx(expected)  # ~0.387
    assert m["recall@3"] == pytest.approx(0.5)
    assert m["coverage@3"] == pytest.approx(0.3)


def test_perfect_ranking():
    m = evaluate(recs={0: [1, 2, 9]}, targets={0: [1, 2]}, k=3, num_items=10)
    assert m["ndcg@3"] == pytest.approx(1.0)
    assert m["recall@3"] == pytest.approx(1.0)


def test_average_over_users():
    # user 0 gets a hit, user 1 doesn't -> both metrics average to 0.5
    m = evaluate(recs={0: [1], 1: [5]}, targets={0: [1], 1: [6]}, k=1, num_items=10)
    assert m["ndcg@1"] == pytest.approx(0.5)
    assert m["recall@1"] == pytest.approx(0.5)