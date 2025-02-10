import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Basic setup
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

# Custom color palette - using vibrant, distinguishable colors
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#D4A5A5']
sns.set_palette(colors)

# Create output folder
output_folder = "./hindi_evaluation_results/plots"
os.makedirs(output_folder, exist_ok=True)

# Load data
results_file = "./hindi_evaluation_results/evaluation_summary.csv"
df = pd.read_csv(results_file)

def create_plot(data, x, y, hue, title):
    plt.figure()
    
    # Create main plot with larger markers
    sns.lineplot(data=data, x=x, y=y, hue=hue, marker='o', markersize=10)
    
    # Highlight best performers with a different color and larger stars
    top_points = data.nlargest(3, y)
    plt.scatter(top_points[x], top_points[y], c='#FFD700', s=200,  # Brighter gold color
               zorder=5, label='Best Performance', marker='*',
               edgecolor='black', linewidth=1)  # Added black edge for better visibility
    
    # Add labels and title
    plt.xlabel(x)
    plt.ylabel(y.replace('-', ' ').title())
    plt.title(title)
    
    # Add legend with better placement
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    return plt.gcf()

# List of metrics to plot
metrics = ["SimLex-Correlation", "WordSim-Correlation", "Clustering-Silhouette"]

# Plot for each normalization method
for norm in df["Normalization"].unique():
    df_norm = df[df["Normalization"] == norm]
    
    for metric in metrics:
        # Create and save dimension plots
        title = f"{metric} vs Dimensions\n{norm} Normalization"
        fig = create_plot(df_norm, "Dimensions", metric, "Window", title)
        
        filename = f"{metric.replace('-', '_')}_dims_{norm}.png"
        fig.savefig(os.path.join(output_folder, filename), bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")

# Plot for fixed dimension
fixed_dim = 100
df_fixed = df[df["Dimensions"] == fixed_dim]

for metric in metrics:
    # Create and save window size plots
    title = f"{metric} vs Window Size\nDimensions = {fixed_dim}"
    fig = create_plot(df_fixed, "Window", metric, "Normalization", title)
    
    filename = f"{metric.replace('-', '_')}_window_dim{fixed_dim}.png"
    fig.savefig(os.path.join(output_folder, filename), bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

print("All plots have been saved in the 'plots' folder.")