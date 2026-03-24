from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def _auc_binary(labels: Sequence[int], scores: Sequence[float]) -> float:
    n = len(labels)
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])

    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sum_rank_pos = sum(ranks[idx] for idx in range(n) if pairs[idx][1] == 1)
    u = sum_rank_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    order = sorted(range(len(labels)), key=lambda i: scores[i], reverse=True)
    total_pos = sum(labels)
    if total_pos == 0:
        return float("nan")

    tp = 0
    precision_sum = 0.0
    for rank, idx in enumerate(order, start=1):
        if labels[idx] == 1:
            tp += 1
            precision_sum += tp / rank
    return precision_sum / total_pos


def classification_metrics(labels: Sequence[int], probs: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    preds: List[int] = [1 if p >= threshold else 0 for p in probs]

    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)

    total = len(labels) if labels else 1
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    auc = _auc_binary(labels, probs)
    ap = _average_precision(labels, probs)

    return {
        "auc": auc,
        "auprc": ap,
        "accuracy": accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }
