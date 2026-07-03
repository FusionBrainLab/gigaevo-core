"""Word-level ROUGE-L F1 (LCS over lowercased whitespace tokens)."""

from __future__ import annotations


def rouge_l_f1(candidate: str, reference: str) -> float:
    cand = candidate.lower().split()
    ref = reference.lower().split()
    if not cand or not ref:
        return 0.0
    dp = [[0] * (len(ref) + 1) for _ in range(len(cand) + 1)]
    for i, c in enumerate(cand):
        for j, r in enumerate(ref):
            dp[i + 1][j + 1] = (
                dp[i][j] + 1 if c == r else max(dp[i][j + 1], dp[i + 1][j])
            )
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)
