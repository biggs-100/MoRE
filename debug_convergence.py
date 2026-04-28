import torch
from real_dataset import TextDataset
from more_demo import MoRE

loader = TextDataset()
data = loader.get_data()
X = data['train_x']
y = data['train_y']

model = MoRE(3, 384, 32)
winners, fam, y_g, g, scores = model.forward(X[:5])
print(f"Initial Winners: {winners}")
print(f"Initial Familiarity: {fam}")
print(f"Gating: {g}")

# Train for 10 steps manually
lr = 0.5
for _ in range(10):
    winners, fam, _, _, all_attn = model.forward(X)
    reward = torch.where(winners == y, 1.0, -1.0)
    for i in range(len(X)):
        model.experts[winners[i]].update_local(X[i:i+1], reward[i:i+1], all_attn[i], lr=lr)

winners, fam, _, _, _ = model.forward(X[:5])
print(f"After 10 steps Winners: {winners}")
print(f"After 10 steps Familiarity: {fam}")
