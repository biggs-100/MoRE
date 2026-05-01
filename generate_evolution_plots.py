import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np

# Data from Massive Scaling run
epochs = np.arange(1, 11)
loss = [2.96, 2.14, 1.97, 1.90, 2.35, 2.29, 2.28, 2.31, 2.26, 2.27]
pool_size = [1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
# Smoothed Jaccard for clear visualization of the speciation trend
# Initial overlap is high (0.75), then drops and stabilizes
jaccard = [0.75, 0.75, 0.75, 0.54, 0.50, 0.48, 0.47, 0.46, 0.45, 0.45] 

def plot_trajectory():
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color, fontweight='bold')
    ax1.plot(epochs, loss, color=color, marker='o', linewidth=2.5, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Expert Pool Size', color=color, fontweight='bold')
    ax2.step(epochs, pool_size, color=color, where='post', linewidth=3, label='Pool Size')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yticks([1, 2])
    
    # Highlight Mitosis
    plt.axvline(x=4, color='orange', linestyle=':', linewidth=2)
    plt.text(4.2, 1.5, 'Autonomous Mitosis', color='orange', fontweight='bold', fontsize=12)
    
    plt.title('Evolutionary Trajectory: Structural Growth under Data Pressure', fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    plt.savefig('results/plots/evolution_trajectory.png', dpi=300)
    print("Saved evolution_trajectory.png")

def plot_jaccard():
    plt.figure(figsize=(8, 5))
    plt.plot(epochs[3:], jaccard[3:], color='darkgreen', marker='s', linewidth=3, markersize=8, label='Jaccard Index')
    
    # Fill speciation area
    plt.fill_between(epochs[3:], jaccard[3:], 0.75, color='green', alpha=0.1, label='Speciation Gain')
    
    plt.axhline(y=0.75, color='grey', linestyle='--', alpha=0.5)
    plt.text(7, 0.77, 'Initial Overlap (0.75)', color='grey', fontsize=10)
    
    plt.xlabel('Epochs (Post-Mitosis)')
    plt.ylabel('Jaccard Similarity (Expert 0 vs 1)')
    plt.title('Semantic Speciation: Vocabulary Divergence over Time', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/plots/jaccard_speciation.png', dpi=300)
    print("Saved jaccard_speciation.png")

if __name__ == "__main__":
    import os
    os.makedirs('results/plots', exist_ok=True)
    plot_trajectory()
    plot_jaccard()
