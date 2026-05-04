SPECIAL_IDS = frozenset(range(5))   # 0:<s>  1:<pad>  2:</s>  3:<unk>  4:<mask>


def levenshtein_distance(pred: list, ref: list) -> int:
    m, n = len(pred), len(ref)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        new_dp = [i] + [0] * n
        for j in range(1, n + 1):
            if pred[i - 1] == ref[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp
    return dp[n]


def calculate_wer(predictions: list[list], references: list[list]) -> float:
    """
    Word Error Rate (WER) trên chuỗi gloss.

    Lọc bỏ special tokens (id < 5) trước khi tính.
    """
    total_edit = 0
    total_ref = 0
    for pred, ref in zip(predictions, references):
        pred_f = [t for t in pred if t not in SPECIAL_IDS]
        ref_f  = [t for t in ref  if t not in SPECIAL_IDS]
        total_edit += levenshtein_distance(pred_f, ref_f)
        total_ref  += len(ref_f)
    return total_edit / max(total_ref, 1)
