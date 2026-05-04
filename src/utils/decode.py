"""
CTC Decoders: greedy + prefix beam search.

Beam search dựa trên thuật toán "Prefix Beam Search" (Hannun et al., 2014):
  - Mỗi beam giữ 2 score: prob_blank (kết thúc bằng blank) và prob_non_blank (kết thúc bằng char)
  - Tại mỗi frame, tách trường hợp: extend bằng blank, extend bằng char khác, extend bằng char trùng
  - Pruning: chỉ xét top-K char tại mỗi frame (giảm O(C) → O(K))

Không dùng LM cho đơn giản (có thể plug LM bằng cách cộng log-LM-score vào prefix score).
"""

import math
from collections import defaultdict
from typing import List

import numpy as np
import torch


NEG_INF = -float('inf')


def _logaddexp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


# ---------------------------------------------------------------------------
# Greedy
# ---------------------------------------------------------------------------

def ctc_greedy_decode(log_probs: torch.Tensor, blank_id: int = 0) -> List[List[int]]:
    """
    log_probs: B × T × C
    Returns: list of decoded ID sequences (collapsed + blank-removed).
    """
    pred = log_probs.argmax(dim=-1).cpu().tolist()      # B × T
    out = []
    for seq in pred:
        decoded, prev = [], None
        for tok in seq:
            if tok != blank_id and tok != prev:
                decoded.append(tok)
            prev = tok
        out.append(decoded)
    return out


# ---------------------------------------------------------------------------
# Prefix beam search
# ---------------------------------------------------------------------------

def _beam_search_single(
    log_probs: np.ndarray,
    beam_size: int,
    blank_id: int,
    top_k: int,
) -> List[int]:
    """
    Beam search cho một sequence.

    log_probs: T × C  (numpy, float32, log-probabilities)
    Returns: list[int]  — decoded token IDs.
    """
    T, C = log_probs.shape
    top_k = min(top_k, C)

    # beams: dict[tuple[int]] -> [prob_blank, prob_non_blank]   (log-space)
    beams = {(): [0.0, NEG_INF]}

    for t in range(T):
        # Top-K chars at frame t (giảm tải từ C → top_k)
        topk_idx = np.argpartition(-log_probs[t], top_k - 1)[:top_k]

        new_beams: dict = defaultdict(lambda: [NEG_INF, NEG_INF])

        for prefix, (pb, pnb) in beams.items():
            # Tổng prob của prefix hiện tại (trước khi extend)
            for c in topk_idx:
                lp = float(log_probs[t, c])
                c  = int(c)

                if c == blank_id:
                    # Extend bằng blank → prefix giữ nguyên, score gộp vào pb
                    delta = _logaddexp(pb + lp, pnb + lp)
                    new_beams[prefix][0] = _logaddexp(new_beams[prefix][0], delta)
                else:
                    if len(prefix) > 0 and prefix[-1] == c:
                        # Char trùng cuối prefix:
                        #   - Stay (không extend): cộng vào pnb (chỉ từ pnb cũ)
                        new_beams[prefix][1] = _logaddexp(new_beams[prefix][1], pnb + lp)
                        #   - Extend (sau blank): từ pb cũ
                        new_prefix = prefix + (c,)
                        new_beams[new_prefix][1] = _logaddexp(new_beams[new_prefix][1], pb + lp)
                    else:
                        # Char khác → luôn extend
                        new_prefix = prefix + (c,)
                        delta = _logaddexp(pb + lp, pnb + lp)
                        new_beams[new_prefix][1] = _logaddexp(new_beams[new_prefix][1], delta)

        # Prune theo log-prob tổng = logaddexp(pb, pnb)
        scored = [
            (prefix, _logaddexp(probs[0], probs[1]))
            for prefix, probs in new_beams.items()
        ]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        beams = {prefix: new_beams[prefix] for prefix, _ in scored[:beam_size]}

    # Pick best
    best_prefix, _ = max(
        ((p, _logaddexp(probs[0], probs[1])) for p, probs in beams.items()),
        key=lambda kv: kv[1],
    )
    return list(best_prefix)


def ctc_beam_search_decode(
    log_probs: torch.Tensor,
    beam_size: int = 5,
    blank_id: int = 0,
    top_k: int = 10,
    input_lengths: torch.Tensor = None,
) -> List[List[int]]:
    """
    Decode batch bằng prefix beam search.

    Args:
        log_probs:     B × T × C  (torch tensor)
        beam_size:     số beam giữ lại tại mỗi frame
        blank_id:      ID của CTC blank (mặc định 0)
        top_k:         pruning ngang frame, chỉ xét top-k char/frame
        input_lengths: B  — độ dài thực mỗi sequence (tránh decode trên padding)

    Returns: list các chuỗi ID đã decode.
    """
    log_probs_np = log_probs.detach().cpu().numpy().astype(np.float32)
    B = log_probs_np.shape[0]

    if input_lengths is None:
        lengths = [log_probs_np.shape[1]] * B
    else:
        lengths = input_lengths.cpu().tolist()

    out = []
    for i in range(B):
        T_i = lengths[i]
        seq = _beam_search_single(log_probs_np[i, :T_i], beam_size, blank_id, top_k)
        out.append(seq)
    return out
