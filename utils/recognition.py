import torch
import numpy as np

def ctc_beam_search_decoder(logits, tokenizer, beam_size=5):
    """
    Giải mã Beam Search để tìm chuỗi Gloss tối ưu nhất.
    logits: (Time, Classes)
    """
    probs = torch.nn.functional.softmax(logits, dim=-1).cpu().detach().numpy()
    T, V = probs.shape
    beams = { (): 1.0 }
    
    for t in range(T):
        new_beams = {}
        for char_id in range(V):
            p = probs[t, char_id]
            if p < 0.005: continue 
            
            for path, prob in beams.items():
                if len(path) > 0 and char_id == path[-1]:
                    new_path = path
                else:
                    new_path = path + (char_id,) if char_id != 0 else path
                new_beams[new_path] = new_beams.get(new_path, 0) + prob * p
        
        beams = dict(sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_size])

    best_path = max(beams, key=beams.get)
    return tokenizer.decode(list(best_path))

def greedy_decoder(logits, tokenizer):
    """Giải mã nhanh để theo dõi trong lúc train (nếu cần)"""
    indices = torch.argmax(logits, dim=-1).cpu().numpy()
    prediction = []
    last_idx = -1
    for idx in indices:
        if idx != last_idx and idx != 0:
            prediction.append(idx)
        last_idx = idx
    return tokenizer.decode(prediction)