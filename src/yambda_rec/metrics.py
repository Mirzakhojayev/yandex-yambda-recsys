"""Ranking metrics, written to match the official Yambda benchmark
(benchmarks/yambda/evaluation/metrics.py)."""
import math


def _hits(user_recs: list[int], target_set: set[int], k: int) -> list[int]:
    """For each of the top-k positions: 1 if that item is a target, else 0."""
    return [1 if item in target_set else 0 for item in user_recs[:k]]


def _dcg(hits: list[int]) -> float:
    # position 0 (rank 1) -> 1/log2(2) = 1, position 1 (rank 2) -> 1/log2(3), ...
    return sum(h / math.log2(i + 2) for i, h in enumerate(hits))


def evaluate(
    recs: dict[int, list[int]],
    targets: dict[int, list[int]],
    k: int,
    num_items: int,
) -> dict[str, float]:
    """recs: uid -> ranked item ids. targets: uid -> test-day item ids.
    NDCG and Recall are computed per user, then averaged over target users."""
    ndcg_sum = 0.0
    recall_sum = 0.0
    for uid, target_list in targets.items():
        hits = _hits(recs[uid], set(target_list), k)
        n_pos = min(k, len(target_list))  # official: duplicate targets are counted
        ndcg_sum += _dcg(hits) / _dcg([1] * n_pos)
        recall_sum += sum(hits) / n_pos

    covered = set()
    for user_recs in recs.values():
        covered.update(user_recs[:k])

    n_users = len(targets)
    return {
        f"ndcg@{k}": ndcg_sum / n_users,
        f"recall@{k}": recall_sum / n_users,
        f"coverage@{k}": len(covered) / num_items,
    }