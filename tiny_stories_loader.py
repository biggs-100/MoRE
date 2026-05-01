import torch
import os
import requests

class TinyStoriesLoader:
    """
    Minimal character-level loader for TinyStories or arbitrary text.
    """
    def __init__(self, file_path=None, batch_size=32, seq_len=128, data_str=None):
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        if data_str:
            self.data = data_str
        elif file_path and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                self.data = f.read()
        else:
            # Fallback to a very small synthetic sample if nothing provided
            self.data = "Once upon a time, there was a small robot named MoRE. It loved to learn and evolve."
            
        # Character-level vocab
        chars = sorted(list(set(self.data)))
        self.vocab_size = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        
        self.tokens = torch.tensor(self.encode(self.data), dtype=torch.long)
        self.current_idx = 0

    def filter_by_keywords(self, keywords):
        """
        Creates a new loader containing only stories that match the keywords.
        """
        stories = self.data.split("<|endoftext|>") 
        filtered_stories = [s for s in stories if any(k.lower() in s.lower() for k in keywords)]
        filtered_data = "<|endoftext|>".join(filtered_stories)
        return TinyStoriesLoader(data_str=filtered_data, batch_size=self.batch_size, seq_len=self.seq_len)



    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, l):
        if isinstance(l, torch.Tensor):
            l = l.tolist()
        return ''.join([self.itos[i] for i in l])

    def get_batch(self):
        """
        Returns a batch of (x, y) where y is x shifted by 1.
        """
        if self.current_idx + self.batch_size * self.seq_len + 1 > len(self.tokens):
            self.current_idx = 0 # Loop back
            
        # Simple non-overlapping windows for now to keep state-resetting clean
        ix = torch.arange(self.current_idx, self.current_idx + self.batch_size * self.seq_len, self.seq_len)
        
        x = torch.stack([self.tokens[i:i+self.seq_len] for i in ix])
        y = torch.stack([self.tokens[i+1:i+self.seq_len+1] for i in ix])
        
        self.current_idx += self.batch_size * self.seq_len
        return x, y

def download_tinystories(target_path="tinystories_subset.txt", limit_chars=100000):
    """
    Downloads a small subset of TinyStories if not present.
    """
    if os.path.exists(target_path):
        return
        
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
    print(f"Downloading TinyStories subset from {url}...")
    
    # We use streaming to get only a subset
    response = requests.get(url, stream=True)
    with open(target_path, "w", encoding="utf-8") as f:
        count = 0
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                f.write(decoded_line + "\n")
                count += len(decoded_line)
                if count >= limit_chars:
                    break
    print(f"Downloaded {count} characters to {target_path}.")

if __name__ == "__main__":
    download_tinystories()
    loader = TinyStoriesLoader("tinystories_subset.txt")
    print(f"Vocab size: {loader.vocab_size}")
    x, y = loader.get_batch()
    print(f"Batch X shape: {x.shape}")
    print(f"Sample X: {loader.decode(x[0][:20])}")
    print(f"Sample Y: {loader.decode(y[0][:20])}")
