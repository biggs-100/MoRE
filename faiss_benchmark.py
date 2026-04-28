"""
FAISS Benchmark: Comparing Exact (Dense) vs. Approximate (FAISS) performance.
"""

import torch
import time
import numpy as np
import faiss
from rperceptron import RPerceptron
from rich.console import Console
from rich.table import Table

console = Console()

def run_benchmark(M_values, d=384, topk=5):
    results = []
    
    for M in M_values:
        console.log(f"Testing M = {M:,}...")
        
        # 1. Initialize RPerceptron
        # Threshold set to 1 so we can force FAISS even for small M for parity check
        model = RPerceptron(d, M, topk=topk, faiss_threshold=1, use_faiss=True)
        
        # Test sample
        x = torch.randn(1, d)
        
        # 2. Exact (Dense) Path
        # We manually trigger dense by setting use_faiss=False temporarily
        model.use_faiss = False
        start_t = time.perf_counter()
        w_exact, s_exact, _, _, _ = model.forward(x)
        t_exact = (time.perf_counter() - start_t) * 1000 # ms
        
        # 3. FAISS Path
        model.use_faiss = True
        # For high-M, use IVF to show real speedup
        if M > 5000:
            nlist = int(np.sqrt(M))
            quantizer = faiss.IndexFlatIP(d)
            model.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            keys_np = model.keys.detach().cpu().numpy().astype('float32')
            model.index.train(keys_np)
            model.index.add(keys_np)
            model.index.nprobe = 10
        else:
            model._rebuild_index()
            
        start_t = time.perf_counter()
        w_faiss, s_faiss, _, _, _ = model.forward(x)
        t_faiss = (time.perf_counter() - start_t) * 1000 # ms
        
        # 4. Parity Check
        match = (w_exact == w_faiss).item()
        error = torch.abs(s_exact - s_faiss).item()
        
        results.append({
            'M': M,
            't_exact': t_exact,
            't_faiss': t_faiss,
            'match': "YES" if match else "NO",
            'error': error,
            'speedup': t_exact / t_faiss
        })

    # Display results
    table = Table(title=f"MoRE Scalability Benchmark (d={d}, topk={topk})")
    table.add_column("Prototypes (M)", justify="right", style="cyan")
    table.add_column("Dense (ms)", justify="right", style="magenta")
    table.add_column("FAISS (ms)", justify="right", style="green")
    table.add_column("Speedup", justify="right", style="bold yellow")
    table.add_column("Parity", justify="center")
    table.add_column("Score Err", justify="right")

    for res in results:
        table.add_row(
            f"{res['M']:,}",
            f"{res['t_exact']:.4f}",
            f"{res['t_faiss']:.4f}",
            f"{res['speedup']:.2f}x",
            res['match'],
            f"{res['error']:.2e}"
        )

    console.print(table)

if __name__ == "__main__":
    # Test scale from small to massive
    M_SCALES = [100, 1000, 10000, 100000]
    run_benchmark(M_SCALES)
