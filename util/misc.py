import re
import torch
import torch.nn.functional as F

def extract_numbers(data):
    flat_data = [item for sublist in data for item in sublist]
    numbers = []
    for item in flat_data:
        matches = re.findall(r'-?\d+\.\d+', item)
        if matches:
            numbers.extend(matches)
    return [float(number) for number in numbers]

def l_align_uniform(user_emb, item_emb):
    align = alignment(user_emb, item_emb)
    uniform = (uniformity(user_emb) + uniformity(item_emb)) / 2
    return align, uniform
    
def alignment(x, y, alpha=2):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniformity(x, t=2):
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()