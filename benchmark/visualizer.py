import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import seaborn as sns
import numpy as np
import json
import os

def plot_interference_heatmap(matrix, model_name, mode):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy'})
    plt.title(f"Interference Matrix: {model_name.upper()} on {mode.upper()}")
    plt.xlabel("Task ID (Evaluated)")
    plt.ylabel("Task ID (Last Trained)")
    
    # Asegurar que el directorio existe
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(f"results/plots/heatmap_{model_name}_{mode}.png")
    plt.close()

def plot_accuracy_trend(matrix, model_name, mode):
    # Accuracy on each task j over time (i)
    # i is the x-axis, j are different lines
    matrix = np.array(matrix)
    n_tasks = matrix.shape[0]
    
    plt.figure(figsize=(10, 6))
    for j in range(n_tasks):
        # Tomamos R[i, j] para i >= j
        steps = np.arange(j, n_tasks)
        accs = matrix[j:, j]
        plt.plot(steps, accs, marker='o', label=f"Task {j}")
        
    plt.title(f"Stability Trend: {model_name.upper()} on {mode.upper()}")
    plt.xlabel("Current Training Task Index")
    plt.ylabel("Accuracy on Historical Task")
    plt.legend()
    plt.grid(True)
    
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(f"results/plots/trend_{model_name}_{mode}.png")
    plt.close()

def plot_pareto_curve(results_list, mode):
    # results_list contains dicts with 'theta', 'acc', 'bwt'
    if not results_list: return
    
    thetas = [r['theta'] for r in results_list]
    accs = [r['acc'] for r in results_list]
    bwts = [r['bwt'] for r in results_list]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(bwts, accs, c=thetas, cmap='viridis', s=100)
    plt.colorbar(label='Theta Threshold')
    
    # Annotate points
    for i, txt in enumerate(thetas):
        plt.annotate(f"θ={txt}", (bwts[i], accs[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title(f"Pareto Curve: Stability vs Accuracy ({mode.upper()})")
    plt.xlabel("Backward Transfer (BWT) - Stability")
    plt.ylabel("Average Accuracy (ACC)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(f"results/plots/pareto_{mode}.png")
    plt.close()

if __name__ == "__main__":
    import glob
    all_files = glob.glob("results_*.json")
    
    # Standard Heatmaps and Trends
    for f_path in all_files:
        with open(f_path, "r") as f:
            data = json.load(f)
            plot_interference_heatmap(data['matrix'], data['model'], data['mode'])
            plot_accuracy_trend(data['matrix'], data['model'], data['mode'])
            
    # Group by mode for Pareto (only for 'more' model)
    modes = set()
    for f_path in all_files:
        if 'results_more_' in f_path:
            with open(f_path, "r") as f:
                data = json.load(f)
                modes.add(data['mode'])
                
    for mode in modes:
        sweep_data = []
        for f_path in all_files:
            if f'results_more_{mode}_theta' in f_path:
                with open(f_path, "r") as f:
                    d = json.load(f)
                    if 'theta' in d and d['theta'] is not None:
                        sweep_data.append(d)
        
        # Sort by theta for cleaner plotting
        sweep_data.sort(key=lambda x: x['theta'])
        plot_pareto_curve(sweep_data, mode)
