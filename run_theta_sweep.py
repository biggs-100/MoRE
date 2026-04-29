import subprocess
import sys

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    thetas = [0.3, 0.5, 0.7, 0.9]
    mode = "permuted_mnist"
    n_tasks = 5
    samples = 1000
    
    print(f"Starting Theta Sweep for MoRE-3 on {mode}...")
    
    for theta in thetas:
        cmd = f"py run_benchmark.py --mode {mode} --n_tasks {n_tasks} --model more --theta {theta} --n_experts 5 --samples {samples}"
        try:
            run_cmd(cmd)
        except Exception as e:
            print(f"Error during theta={theta}: {e}")
            
    print("\nSweep complete. Generating Pareto plot...")
    run_cmd("py -m benchmark.visualizer")

if __name__ == "__main__":
    main()
