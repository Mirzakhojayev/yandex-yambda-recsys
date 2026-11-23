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
    paper_compat: bool = True,
) -> dict[str, float]:
    """recs: uid -> ranked item ids. targets: uid -> test-day item ids.

    paper_compat=True reproduces the Yambda paper's tables:
        NDCG = sum of DCG / sum of IDCG over users, Recall = hits / len(targets).
    paper_compat=False matches the current official benchmark code:
        NDCG = mean of per-user DCG/IDCG, Recall = hits / min(k, len(targets)).
    """
    dcg_sum = idcg_sum = ndcg_sum = recall_sum = 0.0
    for uid, target_list in targets.items():
        hits = _hits(recs[uid], set(target_list), k)
        n_pos = min(k, len(target_list))  # duplicate targets are counted
        dcg = _dcg(hits)
        idcg = _dcg([1] * n_pos)
        dcg_sum += dcg
        idcg_sum += idcg
        ndcg_sum += dcg / idcg
        recall_sum += sum(hits) / (len(target_list) if paper_compat else n_pos)

    covered = set()
    for user_recs in recs.values():
        covered.update(user_recs[:k])

    n_users = len(targets)
    return {
        f"ndcg@{k}": dcg_sum / idcg_sum if paper_compat else ndcg_sum / n_users,
        f"recall@{k}": recall_sum / n_users,
        f"coverage@{k}": len(covered) / num_items,
    }
