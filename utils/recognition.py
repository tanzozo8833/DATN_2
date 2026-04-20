import torch
import numpy as np


def ctc_greedy_decoder(logits, tokenizer):
    """Greedy decoder - chọn argmax mỗi timestep, collapse CTC"""
    indices = torch.argmax(logits, dim=-1).cpu().numpy()
    prediction = []
    last_idx = -1
    for idx in indices:
        if idx != last_idx and idx != 0:
            prediction.append(idx)
        last_idx = idx
    return tokenizer.decode(prediction)


def ctc_beam_search_decoder(logits, tokenizer, beam_size=10):
    """
    Beam Search Decoder với proper CTC collapse
    - Xử lý blank (id=0) đúng cách
    - Merge repeated tokens
    - Giữ beam_size paths tốt nhất
    """
    probs = torch.nn.functional.softmax(logits, dim=-1).cpu().detach().numpy()
    T, V = probs.shape

    beams = {(): 1.0}

    for t in range(T):
        new_beams = {}

        for char_id in range(V):
            p = probs[t, char_id]
            if p < 0.001:
                continue

            for path, prob in beams.items():
                if char_id == 0:
                    new_path = path
                    new_beams[new_path] = new_beams.get(new_path, 0) + prob * p
                elif len(path) > 0 and char_id == path[-1]:
                    new_path = path
                    new_beams[new_path] = new_beams.get(new_path, 0) + prob * p
                else:
                    new_path = path + (char_id,)
                    new_beams[new_path] = new_beams.get(new_path, 0) + prob * p

        beams = dict(
            sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_size]
        )

    best_path = max(beams, key=beams.get)
    return tokenizer.decode(list(best_path))


def ctc_beam_search_with_lm(logits, tokenizer, beam_size=10, lm=None):
    """
    Beam Search với Language Model (nếu có)
    LM giúp cải thiện WER bằng cách ưu tiên transition hợp lý
    """
    probs = torch.nn.functional.softmax(logits, dim=-1).cpu().detach().numpy()
    T, V = probs.shape

    beams = {(): 1.0}

    for t in range(T):
        new_beams = {}

        for char_id in range(V):
            p = probs[t, char_id]
            if p < 0.001:
                continue

            for path, prob in beams.items():
                lm_score = 0
                if lm is not None and len(path) > 0:
                    prev_token = path[-1]
                    lm_score = lm.get_score(prev_token, char_id)

                if char_id == 0:
                    new_path = path
                    new_beams[new_path] = new_beams.get(new_path, 0) + prob * p
                elif len(path) > 0 and char_id == path[-1]:
                    new_path = path
                    new_beams[new_path] = new_beams.get(new_path, 0) + prob * p
                else:
                    new_path = path + (char_id,)
                    score = prob * p * (10**lm_score)
                    new_beams[new_path] = new_beams.get(new_path, 0) + score

        beams = dict(
            sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_size]
        )

    best_path = max(beams, key=beams.get)
    return tokenizer.decode(list(best_path))
