import numpy as np
import torch

def levenshtein_distance(ref, hyp):
    """
    Tính khoảng cách chỉnh sửa giữa chuỗi tham chiếu (ref) và chuỗi dự đoán (hyp).
    """
    m, n = len(ref), len(hyp)
    if m == 0: return n
    if n == 0: return m

    dp = np.zeros((m + 1, n + 1))
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    return dp[m][n]

def calculate_wer(predictions, targets):
    """
    Tính WER cho một batch.
    Args:
        predictions (list of list): Danh sách các chuỗi ID dự đoán.
        targets (list of list): Danh sách các chuỗi ID gốc.
    """
    total_distance = 0
    total_words = 0

    for ref, hyp in zip(targets, predictions):
        # Loại bỏ các token đặc biệt như <pad>, <s>, </s> nếu có
        ref = [i for i in ref if i > 4] 
        hyp = [i for i in hyp if i > 4]

        total_distance += levenshtein_distance(ref, hyp)
        total_words += len(ref)

    if total_words == 0: return 1
    return total_distance / total_words