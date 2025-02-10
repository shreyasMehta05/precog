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
    
    # Add each visualization with description
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

def compare_trends(cooccurrence_file: str, neural_dir: str, output_dir: str):
    """
    Load evaluation summaries and generate comprehensive comparison visualizations.
    """
    try:
        # [Previous loading and data preparation code remains the same]
        
        # Load and prepare data
        df_cooc = load_results(cooccurrence_file)
        df_neural = load_all_neural_json_results(neural_dir)
        
        df_cooc['Method'] = 'Co-occurrence'
        df_neural['Method'] = 'Neural'
        
        df_all = pd.concat([df_cooc, df_neural], ignore_index=True)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save merged results
        merged_csv = output_path / "merged_evaluation_summary.csv"
        df_all.to_csv(merged_csv, index=False)
        
        # Generate visualizations using matplotlib
        create_metric_plots_matplotlib(df_all, output_path)
        create_heatmap_matplotlib(df_all, output_path)
        create_additional_visualizations(df_all, output_path)
        
        # Generate HTML report
        generate_html_report(df_all, output_path)
        
        logger.info(f"All visualizations and report saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in comparison process: {str(e)}")
        raise