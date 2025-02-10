#!/usr/bin/env python3

import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

def load_json_results(file_path: str) -> pd.DataFrame:
    """Load evaluation results from a JSON file into a DataFrame."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def compare_hindi_embeddings(hindi_cooccurrence_json: str, hindi_neural_json: str, output_dir: str):
    """
    Load evaluation summaries from co-occurrence and neural embeddings for Hindi,
    merge them, and generate comparison plots.
    """
    # Load the JSON evaluation results
    df_cooc = load_json_results(hindi_cooccurrence_json)
    df_neural = load_json_results(hindi_neural_json)
    
    # Add a column to identify the method
    df_cooc['Method'] = 'Co-occurrence (Hindi)'
    df_neural['Method'] = 'Neural (Hindi)'
    
    # Merge the two DataFrames
    df_all = pd.concat([df_cooc, df_neural], ignore_index=True)
    
    # Save merged results to CSV
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    merged_csv = Path(output_dir) / "hindi_merged_evaluation_summary.csv"
    df_all.to_csv(merged_csv, index=False)
    
    # Create comparison bar charts for key metrics
    metrics = ['SimLex-Correlation', 'WordSim-Correlation', 'Analogy-Accuracy', 'Clustering-Silhouette']
    fig = go.Figure()
    
    for metric in metrics:
        for method in df_all['Method'].unique():
            subset = df_all[df_all['Method'] == method]
            fig.add_trace(go.Bar(
                x=[method],
                y=[subset[metric].mean()],  # average value if multiple configurations exist
                name=f"{method} - {metric}"
            ))
    
    fig.update_layout(
        barmode='group',
        title="Comparison of Hindi Embeddings: Co-occurrence vs Neural",
        xaxis_title="Method",
        yaxis_title="Metric Value",
        width=800,
        height=600
    )
    
    # Save the comparison plot as HTML and PNG (requires kaleido)
    comparison_html = Path(output_dir) / "hindi_comparison_metrics.html"
    comparison_png = Path(output_dir) / "hindi_comparison_metrics.png"
    fig.write_html(comparison_html)
    fig.write_image(comparison_png)
    print(f"Comparison plots saved to {output_dir}")

def main():
    # Paths to the JSON summary files for Hindi embeddings
    hindi_cooccurrence_json = "./hindi_evaluation_results/hindi_cooccurrence_summary.json"
    hindi_neural_json = "./hindi_neural_evaluation_results/hindi_neural_summary.json"
    output_dir = "./hindi_comparison_results"
    
    if not os.path.exists(hindi_cooccurrence_json):
        print(f"Co-occurrence evaluation file not found at {hindi_cooccurrence_json}")
        return
    if not os.path.exists(hindi_neural_json):
        print(f"Neural evaluation file not found at {hindi_neural_json}")
        return
    
    compare_hindi_embeddings(hindi_cooccurrence_json, hindi_neural_json, output_dir)

if __name__ == "__main__":
    main()
