import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    csv_path = "results/plot_data.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run summarize_results.py first.")
        return

    # FIXED: removed invalid 'index_index' and used index_col=0
    df = pd.read_csv(csv_path, index_col=0)
    
    # Ensure Checkpoint is a column for plotting
    df = df.reset_index()
    
    # Professional styling
    plt.style.use('seaborn-v0_8-muted')
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- Plotting the Pass@1 Scores ---
    # Using 'zorder' to ensure lines are above bars
    ax1.plot(df['Checkpoint'], df['3shot'], marker='o', markersize=8, linestyle='-', 
             linewidth=2.5, label='3-Shot (RL Tuned)', color='#1F77B4', zorder=3)
    ax1.plot(df['Checkpoint'], df['0shot'], marker='s', markersize=6, linestyle='--', 
             linewidth=2, label='0-Shot (RL Tuned)', color='#FF7F0E', zorder=3)

    # Formatting Primary Axis
    ax1.set_xlabel('Training Steps / Checkpoints', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MBPP Pass@1 (Accuracy)', fontsize=12, fontweight='bold')
    ax1.set_title('GRPO Performance Evolution: Llama-3-8B-Instruct on MBPP', fontsize=14, pad=20)
    
    # Baseline Reference
    try:
        base_val = df.loc[df['Checkpoint'] == 'base', '3shot'].values[0]
        ax1.axhline(y=base_val, color='red', linestyle=':', alpha=0.7, label=f'Base Baseline ({base_val:.3f})')
    except IndexError:
        pass

    # --- Secondary Axis for Improvement ---
    if 'Improvement (%)' in df.columns:
        ax2 = ax1.twinx()
        # Bar chart for the Delta
        ax2.bar(df['Checkpoint'], df['Improvement (%)'], alpha=0.2, color='green', 
                label='Improvement %', zorder=1)
        ax2.set_ylabel('Relative Improvement over Base (%)', color='darkgreen', fontsize=10)
        ax2.set_ylim(min(df['Improvement (%)'].min() - 1, 0), df['Improvement (%)'].max() + 2)
        ax2.grid(False)

    # Clean up layout
    ax1.legend(loc='upper left', frameon=True)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save outputs
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/learning_curve.png"
    plt.savefig(save_path, dpi=300)
    
    print(f"\n✅ Plot successfully saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()