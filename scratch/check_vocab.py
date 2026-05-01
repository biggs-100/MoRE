from tiny_stories_loader import TinyStoriesLoader
loader = TinyStoriesLoader('tinystories_subset.txt')
print(f"Vocab size: {len(loader.itos)}")
import torch
checkpoint = torch.load("more_gpt_bifurcated.pth", weights_only=True)
print(f"Checkpoint keys: {list(checkpoint.keys())[:10]}")
print(f"Checkpoint vocab_size: {checkpoint.get('vocab_size')}")
if 'state_dict' in checkpoint:
    print(f"State dict keys: {list(checkpoint['state_dict'].keys())[:5]}")


