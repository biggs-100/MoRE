import subprocess
import sys

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # 10 valores para una curva suave (Prioridad Alta para Paper)
    thetas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    mode = "split_mnist"
    n_tasks = 5
    samples = 1000
    
    python_exe = ".\\venv\\Scripts\\python.exe"
    
    print(f"Starting High-Density Theta Sweep for MoRE-3 on {mode}...")
    
    for theta in thetas:
        cmd = f"uv run --python 3.11 --with torch --with torchvision --with numpy --with requests --with matplotlib --with rich --with faiss-cpu py run_benchmark.py --mode {mode} --n_tasks {n_tasks} --model more --theta {theta} --n_experts 5 --samples {samples} --seed 101"
        try:
            run_cmd(cmd)
        except Exception as e:
            print(f"Error during theta={theta}: {e}")
            
    print("\nSweep complete. Generating Pareto plot...")
    run_cmd(f"uv run --python 3.11 --with torch --with torchvision --with numpy --with requests --with matplotlib --with rich --with faiss-cpu py -m benchmark.visualizer")

if __name__ == "__main__":
    main()
