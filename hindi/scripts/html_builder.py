import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_html_report(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate a comprehensive HTML report with visualizations and analysis.
    
    Args:
        df (pd.DataFrame): DataFrame containing the metrics.
        output_dir (Path): Directory to save the report.
    """
    metrics = ['SimLex-Correlation', 'WordSim-Correlation', 'Analogy-Accuracy', 'Clustering-Silhouette']
    
    # Calculate summary statistics for each method
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
    
    # Begin HTML content
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
            iframe {{
                border: none;
                width: 100%;
                height: 500px;
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
    
    # Add metric-specific analysis cards
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
    
    # Add visualizations (bar plots and heatmap images)
    html_content += """
            <h2>Visualizations</h2>
    """
    visualization_files = [
        ('performance_heatmap_mpl.png', 'Performance Heatmap', 
         'Heatmap showing the average performance of each method across all metrics.'),
        ('radar_comparison_mpl.png', 'Radar Chart Comparison (Static Image)', 
         'Static image of the radar chart comparing the relative strengths of each method across metrics.')
    ]
    for metric in metrics:
        visualization_files.append((f"{metric.lower().replace('-', '_')}_comparison_mpl.png",
                                      f"{metric} Distribution",
                                      f"Distribution of {metric} scores across methods."))
    for filename, title, description in visualization_files:
        html_content += f"""
            <div class="visualization">
                <h3>{title}</h3>
                <p>{description}</p>
                <img src="{filename}" alt="{title}">
            </div>
        """
    
    # Embed the interactive radar chart (generated as HTML) via an iframe
    html_content += """
            <h2>Interactive Radar Chart</h2>
            <p>The following interactive radar chart allows you to explore the relative performance across methods.</p>
            <iframe src="radar_comparison.html"></iframe>
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
    
    report_path = output_dir / "evaluation_report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    logger.info(f"Generated HTML report at {report_path}")

def load_results(cooccurrence_file: str) -> pd.DataFrame:
    """
    Load co-occurrence evaluation results from a CSV file.
    The CSV is assumed to have columns (e.g., Window, Normalization, Dimensions, 
    SimLex-Correlation, SimLex-Coverage, WordSim-Correlation, WordSim-Coverage, 
    Analogy-Accuracy, Clustering-Silhouette).
    """
    try:
        df = pd.read_csv(cooccurrence_file)
        # Add a column to indicate these results come from the co-occurrence method.
        df['Method'] = 'Co-occurrence'
        return df
    except Exception as e:
        logger.error(f"Error loading co-occurrence results: {str(e)}")
        raise

def load_all_neural_json_results(neural_dir: str) -> pd.DataFrame:
    """
    Load all neural model evaluation summaries stored as JSON files from a directory.
    Each JSON file is assumed to contain a list of dictionaries.
    """
    try:
        neural_results = []
        for file in Path(neural_dir).glob("*.json"):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                neural_results.extend(data)
        df = pd.DataFrame(neural_results)
        # Mark these results as coming from the neural method.
        df['Method'] = 'Neural'
        return df
    except Exception as e:
        logger.error(f"Error loading neural results: {str(e)}")
        raise

def create_metric_plots_matplotlib(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create and save metric plots using matplotlib.
    One bar plot is created for each metric.
    """
    try:
        metrics = ['SimLex-Correlation', 'WordSim-Correlation', 'Analogy-Accuracy', 'Clustering-Silhouette']
        for metric in metrics:
            plt.figure(figsize=(8,6))
            sns.barplot(data=df, x='Method', y=metric)
            plt.title(f"{metric} by Method")
            plt.tight_layout()
            plot_path = output_dir / f"{metric.lower().replace('-', '_')}_comparison_mpl.png"
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved plot for {metric} at {plot_path}")
    except Exception as e:
        logger.error(f"Error creating metric plots: {str(e)}")

def create_heatmap_matplotlib(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create and save a heatmap of the average performance across methods.
    """
    try:
        metrics = ['SimLex-Correlation', 'WordSim-Correlation', 'Analogy-Accuracy', 'Clustering-Silhouette']
        pivot = df.groupby('Method')[metrics].mean().reset_index()
        plt.figure(figsize=(8,6))
        sns.heatmap(pivot[metrics].set_index(pivot['Method']), annot=True, cmap='viridis')
        plt.title("Average Performance Heatmap")
        heatmap_path = output_dir / "performance_heatmap_mpl.png"
        plt.savefig(heatmap_path)
        plt.close()
        logger.info(f"Saved heatmap at {heatmap_path}")
    except Exception as e:
        logger.error(f"Error creating heatmap: {str(e)}")

def create_additional_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create an interactive radar chart comparing average metrics across methods.
    Instead of exporting as a static image via Kaleido, we export the chart as an HTML file.
    """
    try:
        methods = df['Method'].unique()
        metrics = ['SimLex-Correlation', 'WordSim-Correlation', 'Analogy-Accuracy', 'Clustering-Silhouette']
        radar_data = []
        for method in methods:
            method_data = df[df['Method'] == method]
            means = [method_data[metric].mean() for metric in metrics]
            radar_data.append(go.Scatterpolar(
                r=means,
                theta=metrics,
                fill='toself',
                name=method
            ))
        fig = go.Figure(data=radar_data)
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[df[metrics].min().min(), df[metrics].max().max()]
                )
            ),
            showlegend=True,
            title="Radar Chart Comparison"
        )
        # Save the interactive chart as an HTML file
        radar_path = output_dir / "radar_comparison.html"
        with open(radar_path, 'w') as f:
            f.write(fig.to_html(include_plotlyjs='cdn'))
        logger.info(f"Saved interactive radar chart at {radar_path}")
    except Exception as e:
        logger.error(f"Error creating additional visualizations: {str(e)}")

def compare_trends(cooccurrence_file: str, neural_dir: str, output_dir: str):
    """
    Load evaluation summaries from co-occurrence and neural methods,
    merge them, generate visualizations, and create an HTML report.
    """
    try:
        # Load data from both sources
        df_cooc = load_results(cooccurrence_file)
        df_neural = load_all_neural_json_results(neural_dir)
        
        # Merge the two DataFrames
        df_all = pd.concat([df_cooc, df_neural], ignore_index=True)
        
        # Create (or ensure) the output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save the merged results as a CSV file
        merged_csv = output_path / "merged_evaluation_summary.csv"
        df_all.to_csv(merged_csv, index=False)
        logger.info(f"Merged evaluation results saved to {merged_csv}")
        
        # Generate visualizations
        create_metric_plots_matplotlib(df_all, output_path)
        create_heatmap_matplotlib(df_all, output_path)
        create_additional_visualizations(df_all, output_path)
        
        # Generate an HTML report
        generate_html_report(df_all, output_path)
        
        logger.info(f"All visualizations and report saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error in comparison process: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage:
    # Update the following paths according to your file locations.
    cooccurrence_file = "hindi_evaluation_results/evaluation_summary.csv"  # your co-occurrence CSV file
    neural_dir = "hindi_evaluation_results/fasttext"  # folder with neural JSON summaries
    output_dir = "comparison_results"  # directory where merged results and report will be saved
    
    compare_trends(cooccurrence_file, neural_dir, output_dir)
