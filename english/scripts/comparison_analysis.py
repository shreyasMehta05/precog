#!/usr/bin/env python3
import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_html_report(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate a comprehensive HTML report with visualizations and analysis.
    
    Args:
        df (pd.DataFrame): DataFrame containing the metrics
        output_dir (Path): Directory to save the report
    """
    metrics = ['SimLex-Correlation', 'WordSim-Correlation', 'Analogy-Accuracy', 'Clustering-Silhouette']
    
    # Calculate summary statistics
    summary_stats = {}
    for method in df['Method'].unique():
        method_data = df[df['Method'] == method]
        summary_stats[method] = {
            metric: {
                'mean': method_data[metric].mean(),
                'std': method_data[metric].std(),
                'min': method_data[metric].min(),
                'max': method_data[metric].max()
            } for metric in metrics
        }

    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Word Embeddings Evaluation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            .metric-card {{
                border: 1px solid #ddd;
                padding: 15px;
                margin: 10px 0;
                border-radius: 4px;
            }}
            .stat-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 10px 0;
            }}
            .stat-box {{
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 4px;
            }}
            .visualization {{
                margin: 20px 0;
                text-align: center;
            }}
            .visualization img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            th, td {{
                padding: 8px;
                border: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background-color: #f8f9fa;
            }}
            .highlight {{
                background-color: #e3f2fd;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Word Embeddings Evaluation Report</h1>
            <p>This report compares the performance of different word embedding methods across multiple evaluation metrics.</p>
            
            <h2>Overview</h2>
            <p>Number of methods compared: {len(df['Method'].unique())}</p>
            <p>Total evaluations: {len(df)}</p>
            
            <h2>Performance Summary</h2>
    """
    
    # Add metric-specific analysis
    for metric in metrics:
        best_method = df.groupby('Method')[metric].mean().idxmax()
        html_content += f"""
            <div class="metric-card">
                <h3>{metric}</h3>
                <div class="stat-grid">
        """
        
        for method in summary_stats:
            stats = summary_stats[method][metric]
            highlight = 'highlight' if method == best_method else ''
            html_content += f"""
                    <div class="stat-box {highlight}">
                        <h4>{method}</h4>
                        <p>Mean: {stats['mean']:.3f}</p>
                        <p>Std: {stats['std']:.3f}</p>
                        <p>Range: [{stats['min']:.3f}, {stats['max']:.3f}]</p>
                    </div>
            """
        
        html_content += """
                </div>
            </div>
        """
    
    # Add visualizations
    html_content += """
            <h2>Visualizations</h2>
    """
    
    # List of visualization files to include in the report.
    # Adjust these filenames if needed.
    visualization_files = [
        ('performance_heatmap_mpl.png', 'Performance Heatmap', 
         'Heatmap showing the average performance of each method across all metrics.'),
        ('radar_comparison_mpl.png', 'Radar Chart Comparison', 
         'Radar chart comparing the relative strengths of each method across metrics.')
    ]
    
    for metric in metrics:
        visualization_files.append(( 
            f"{metric.lower().replace('-', '_')}_comparison_mpl.png",
            f"{metric} Distribution",
            f"Distribution of {metric} scores across methods, showing individual data points, means, and overall spread."
        ))
    
    for filename, title, description in visualization_files:
        html_content += f"""
            <div class="visualization">
                <h3>{title}</h3>
                <p>{description}</p>
                <img src="{filename}" alt="{title}">
            </div>
        """
    
    # Add detailed statistics table
    html_content += """
            <h2>Detailed Statistics</h2>
            <table>
                <tr>
                    <th>Method</th>
                    <th>Metric</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
    """
    
    for method in summary_stats:
        for metric in metrics:
            stats = summary_stats[method][metric]
            html_content += f"""
                <tr>
                    <td>{method}</td>
                    <td>{metric}</td>
                    <td>{stats['mean']:.3f}</td>
                    <td>{stats['std']:.3f}</td>
                    <td>{stats['min']:.3f}</td>
                    <td>{stats['max']:.3f}</td>
                </tr>
            """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # Save the HTML report
    report_path = output_dir / "evaluation_report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Generated HTML report at {report_path}")

def create_metric_plots_matplotlib(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create metric plots using matplotlib/seaborn with improved point distribution.
    
    Args:
        df (pd.DataFrame): DataFrame containing the metrics.
        output_dir (Path): Directory to save plots.
    """
    metrics = ['SimLex-Correlation', 'WordSim-Correlation', 'Analogy-Accuracy', 'Clustering-Silhouette']
    
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        sns.violinplot(data=df, x='Method', y=metric, inner='box', color='lightgray')
        sns.stripplot(data=df, x='Method', y=metric, color='red', size=4, jitter=0.2, alpha=0.6)
        
        means = df.groupby('Method')[metric].mean()
        plt.plot(range(len(means)), means.values, 'D', color='blue', markersize=10, label='Mean')
        
        plt.title(f"{metric} Comparison", pad=20, fontsize=14)
        plt.xlabel('Method', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        
        filename = f"{metric.lower().replace('-', '_')}_comparison_mpl"
        save_plot(plt.gcf(), output_dir, filename)
        plt.close()

def create_heatmap_matplotlib(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create enhanced heatmap using matplotlib/seaborn.
    
    Args:
        df (pd.DataFrame): DataFrame containing the metrics.
        output_dir (Path): Directory to save plots.
    """
    metrics = ['SimLex-Correlation', 'WordSim-Correlation', 'Analogy-Accuracy', 'Clustering-Silhouette']
    methods = df['Method'].unique()
    
    mean_values = []
    for method in methods:
        method_means = df[df['Method'] == method][metrics].mean()
        mean_values.append(method_means.values)
    
    plt.figure(figsize=(14, 8))
    
    sns.heatmap(mean_values, 
                xticklabels=metrics,
                yticklabels=methods,
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Score'})
    
    plt.title("Performance Heatmap: All Metrics", pad=20, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    save_plot(plt.gcf(), output_dir, "performance_heatmap_mpl")
    plt.close()

def create_additional_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create additional visualization types for better comparison.
    
    Args:
        df (pd.DataFrame): DataFrame containing the metrics.
        output_dir (Path): Directory to save plots.
    """
    metrics = ['SimLex-Correlation', 'WordSim-Correlation', 'Analogy-Accuracy', 'Clustering-Silhouette']
    
    # Create radar plot
    plt.figure(figsize=(10, 10))
    means = df.groupby('Method')[metrics].mean()
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    ax = plt.subplot(111, projection='polar')
    
    for method in means.index:
        values = means.loc[method].values
        values = np.concatenate((values, [values[0]]))
        ax.plot(angles, values, 'o-', linewidth=2, label=method)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    plt.title("Metric Comparison Radar Chart", pad=20, fontsize=14)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 1.1))
    
    save_plot(plt.gcf(), output_dir, "radar_comparison_mpl")
    plt.close()

def save_plot(fig, output_path: Path, filename: str) -> None:
    """
    Save a plot with enhanced error handling.
    
    Args:
        fig: A matplotlib figure.
        output_path (Path): Directory to save the plot.
        filename (str): Base filename without extension.
    """
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        png_path = output_path / f"{filename}.png"
        fig.savefig(png_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved matplotlib figure to {png_path}")
    except Exception as e:
        logger.error(f"Error saving plot {filename}: {str(e)}")

def load_results(file_path: str) -> pd.DataFrame:
    """
    Load evaluation results from a JSON or CSV file into a DataFrame.
    
    Args:
        file_path (str): Path to the input file.
        
    Returns:
        pd.DataFrame: Loaded data.
        
    Raises:
        ValueError: If file format is not supported.
        FileNotFoundError: If file doesn't exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        elif file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}. Only .json and .csv are allowed.")
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        raise

def load_all_neural_json_results(neural_dir: str) -> pd.DataFrame:
    """
    Load and merge all neural_evaluation_summary.json files.
    
    Args:
        neural_dir (str): Directory containing neural evaluation results.
        
    Returns:
        pd.DataFrame: Merged results.
    """
    neural_path = Path(neural_dir)
    json_files = list(neural_path.glob("**/neural_evaluation_summary.json"))
    
    if not json_files:
        logger.warning(f"No neural evaluation JSON files found in {neural_dir}")
        return pd.DataFrame()
        
    df_list = []
    for jf in json_files:
        try:
            df = load_results(str(jf))
            df['Configuration'] = jf.parent.name
            df_list.append(df)
        except Exception as e:
            logger.error(f"Error loading {jf}: {str(e)}")
            continue
            
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

def compare_trends(cooccurrence_file: str, neural_dir: str, output_dir: str):
    """
    Load evaluation summaries and generate comprehensive comparison visualizations.
    
    Args:
        cooccurrence_file (str): Path to co-occurrence results.
        neural_dir (str): Directory containing neural results.
        output_dir (str): Output directory for visualizations.
    """
    try:
        df_cooc = load_results(cooccurrence_file)
        df_neural = load_all_neural_json_results(neural_dir)
        
        df_cooc['Method'] = 'Co-occurrence'
        df_neural['Method'] = 'Neural'
        
        df_all = pd.concat([df_cooc, df_neural], ignore_index=True)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        merged_csv = output_path / "merged_evaluation_summary.csv"
        df_all.to_csv(merged_csv, index=False)
        
        # Generate visualizations using matplotlib/seaborn.
        create_metric_plots_matplotlib(df_all, output_path)
        create_heatmap_matplotlib(df_all, output_path)
        create_additional_visualizations(df_all, output_path)
        
        # Generate HTML report.
        generate_html_report(df_all, output_path)
        
        logger.info(f"All visualizations and report saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in comparison process: {str(e)}")
        raise

def main():
    try:
        cooccurrence_file = "./evaluation_results/evaluation_summary.csv"
        neural_dir = "./neural_evaluation_results"
        output_dir = "./comparison_results"
        
        if not Path(cooccurrence_file).exists():
            raise FileNotFoundError(f"Co-occurrence evaluation file not found at {cooccurrence_file}")
        if not Path(neural_dir).exists():
            raise FileNotFoundError(f"Neural evaluation directory not found at {neural_dir}")
        
        compare_trends(cooccurrence_file, neural_dir, output_dir)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
