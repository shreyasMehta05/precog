import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import logging
from tqdm import tqdm
from colorama import Fore, Style, init
import sys
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import requests
import csv
from typing import Dict, List, Tuple, Optional, Union
import torch
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Initialize colorama
init()

class EnhancedEmbeddingEvaluator:
    """Enhanced evaluation suite for word embeddings with improved visualizations and error handling"""
    
    # Color schemes for different visualization types
    COLOR_SCHEMES = {
        'categorical': px.colors.qualitative.Set3,
        'sequential': px.colors.sequential.Viridis,
        'diverging': px.colors.diverging.RdYlBu,
        'correlation': ['#FF0000', '#FFFFFF', '#0000FF']  # Red to Blue
    }
    
    def __init__(self, log_dir: str = "logs"):
        self.setup_logging(log_dir)
        self._setup_gpu()
        self.error_count = 0
        self.warning_count = 0
        
    def setup_logging(self, log_dir: str):
        """Setup logging with both file and console output"""
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"embedding_evaluation_{timestamp}.log"
        
        # Use a unique logger name for this class
        self.logger = logging.getLogger('EnhancedEmbeddingEvaluator')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(file_formatter)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(console_formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
    def _setup_gpu(self) -> None:
        """Setup GPU with enhanced error handling"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device.type == "cuda":
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                self.logger.info(f"{Fore.GREEN}Using GPU with {gpu_memory/1e9:.2f}GB memory{Style.RESET_ALL}")
            else:
                self.logger.warning(f"{Fore.YELLOW}GPU not available, using CPU{Style.RESET_ALL}")
        except Exception as e:
            self.logger.error(f"{Fore.RED}GPU setup failed: {str(e)}{Style.RESET_ALL}")
            self.device = torch.device("cpu")
            
    def create_enhanced_visualizations(self, df: pd.DataFrame, output_dir: str) -> None:
        """Create comprehensive visualizations with enhanced color schemes"""
        try:
            # 1. Correlation Matrix Heatmap
            self._create_correlation_matrix(df, output_dir)
            
            # 2. Parameter Distribution Plots
            self._create_parameter_distributions(df, output_dir)
            
            # 3. Performance Comparison Plots
            self._create_performance_comparisons(df, output_dir)
            
            # 4. Interactive 3D Surface Plots (saved as HTML)
            self._create_3d_surface_plots(df, output_dir)
            
            # 5. Additional Plots for new metrics (saved as PNG)
            self._create_additional_plots(df, output_dir)
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"{Fore.RED}Visualization creation failed: {str(e)}{Style.RESET_ALL}")
            
    def _create_correlation_matrix(self, df: pd.DataFrame, output_dir: str) -> None:
        """Create correlation matrix heatmap with enhanced visuals"""
        correlation_cols = [col for col in df.columns if 'Correlation' in col]
        corr_matrix = df[correlation_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=correlation_cols,
            y=correlation_cols,
            colorscale=self.COLOR_SCHEMES['correlation'],
            zmin=-1, zmax=1
        ))
        
        fig.update_layout(
            title="Correlation Matrix of Evaluation Metrics",
            width=800, height=800
        )
        
        # Save as HTML; if needed, you can convert to PNG using kaleido (optional)
        fig.write_html(Path(output_dir) / "correlation_matrix.html")
        
    def _create_parameter_distributions(self, df: pd.DataFrame, output_dir: str) -> None:
        """Create distribution plots for different parameters"""
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            'Window Size Distribution',
            'Dimension Distribution',
            'SimLex Correlation Distribution',
            'WordSim Correlation Distribution'
        ))
        
        # Window Size Distribution
        fig.add_trace(
            go.Histogram(x=df['Window'], nbinsx=20,
                         marker_color=self.COLOR_SCHEMES['categorical'][0]),
            row=1, col=1
        )
        
        # Dimension Distribution
        fig.add_trace(
            go.Histogram(x=df['Dimensions'], nbinsx=20,
                         marker_color=self.COLOR_SCHEMES['categorical'][1]),
            row=1, col=2
        )
        
        # SimLex Correlation Distribution
        fig.add_trace(
            go.Histogram(x=df['SimLex-Correlation'], nbinsx=20,
                         marker_color=self.COLOR_SCHEMES['categorical'][2]),
            row=2, col=1
        )
        
        # WordSim Correlation Distribution
        fig.add_trace(
            go.Histogram(x=df['WordSim-Correlation'], nbinsx=20,
                         marker_color=self.COLOR_SCHEMES['categorical'][3]),
            row=2, col=2
        )
        
        fig.update_layout(height=1000, width=1200, showlegend=False,
                          title_text="Parameter and Performance Distributions")
        
        fig.write_html(Path(output_dir) / "parameter_distributions.html")
        
    def _create_performance_comparisons(self, df: pd.DataFrame, output_dir: str) -> None:
        """Create performance comparison visualizations"""
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            'Performance by Window Size',
            'Performance by Dimensions',
            'Performance by Normalization Method',
            'Coverage Analysis'
        ))
        
        for i, norm in enumerate(df['Normalization'].unique()):
            subset = df[df['Normalization'] == norm]
            color = self.COLOR_SCHEMES['categorical'][i]
            
            # Window Size Performance
            fig.add_trace(
                go.Scatter(x=subset['Window'], y=subset['SimLex-Correlation'],
                           name=f'{norm} (SimLex)', line=dict(color=color), mode='lines+markers'),
                row=1, col=1
            )
            
            # Dimension Performance
            fig.add_trace(
                go.Scatter(x=subset['Dimensions'], y=subset['SimLex-Correlation'],
                           name=f'{norm} (SimLex)', line=dict(color=color), mode='lines+markers'),
                row=1, col=2
            )
            
            # Normalization Method Performance (using Box plot)
            fig.add_trace(
                go.Box(y=subset['SimLex-Correlation'], name=norm, marker_color=color),
                row=2, col=1
            )
            
            # Coverage Analysis
            fig.add_trace(
                go.Scatter(x=subset['Window'], y=subset['SimLex-Coverage'],
                           name=f'{norm} Coverage', line=dict(color=color), mode='lines+markers'),
                row=2, col=2
            )
        
        fig.update_layout(height=1200, width=1600, showlegend=True,
                          title_text="Comprehensive Performance Analysis")
        
        fig.write_html(Path(output_dir) / "performance_comparisons.html")
        
    def _create_3d_surface_plots(self, df: pd.DataFrame, output_dir: str) -> None:
        """Create 3D surface plots for parameter interactions"""
        for norm in df['Normalization'].unique():
            subset = df[df['Normalization'] == norm]
            pivot = subset.pivot_table(
                values='SimLex-Correlation', index='Window', columns='Dimensions'
            )
            
            fig = go.Figure(data=[go.Surface(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale=self.COLOR_SCHEMES['sequential']
            )])
            
            fig.update_layout(
                title=f'Parameter Interaction Surface - {norm}',
                scene=dict(
                    xaxis_title='Dimensions',
                    yaxis_title='Window Size',
                    zaxis_title='SimLex Correlation'
                ),
                width=1000, height=800
            )
            
            fig.write_html(Path(output_dir) / f"surface_plot_{norm}.html")
    
    def _create_additional_plots(self, summary_df: pd.DataFrame, output_dir: str) -> None:
        """Create additional PNG plots for analogy accuracy and clustering quality"""
        # Create a combined label for each configuration (e.g., normalization and dimensions)
        config_labels = summary_df['Normalization'] + " (d=" + summary_df['Dimensions'].astype(str) + ")"
        
        # Plot Analogy Accuracy
        plt.figure(figsize=(8, 6))
        plt.bar(config_labels, summary_df['Analogy-Accuracy'], color='skyblue')
        plt.xlabel("Embedding Configuration")
        plt.ylabel("Analogy Accuracy")
        plt.title("Word Analogy Task Accuracy")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "analogy_accuracy.png")
        plt.close()
        
        # Plot Clustering Silhouette Score
        plt.figure(figsize=(8, 6))
        plt.bar(config_labels, summary_df['Clustering-Silhouette'], color='salmon')
        plt.xlabel("Embedding Configuration")
        plt.ylabel("Silhouette Score")
        plt.title("Clustering Quality Evaluation")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "clustering_quality.png")
        plt.close()
    
    def evaluate_analogies(self, embeddings: np.ndarray, vocab: Dict[str, int],
                            analogies: List[Tuple[str, str, str, str]]) -> Dict:
        """Evaluate word analogies using vector arithmetic (accuracy)"""
        correct = 0
        total = 0
        embeddings_tensor = torch.tensor(embeddings, device=self.device)
        for a, b, c, d in analogies:
            if a in vocab and b in vocab and c in vocab and d in vocab:
                vec_a = embeddings_tensor[vocab[a]]
                vec_b = embeddings_tensor[vocab[b]]
                vec_c = embeddings_tensor[vocab[c]]
                expected = vec_b - vec_a + vec_c
                similarities = torch.nn.functional.cosine_similarity(expected.unsqueeze(0), embeddings_tensor)
                # Exclude the source words to avoid trivial matches
                similarities[vocab[a]] = -1
                similarities[vocab[b]] = -1
                similarities[vocab[c]] = -1
                pred_idx = similarities.argmax().item()
                if pred_idx == vocab[d]:
                    correct += 1
                total += 1
        accuracy = correct / total if total > 0 else 0
        return {"Analogy-Accuracy": accuracy, "Analogy-Total": total}
    
    def evaluate_clustering(self, embeddings: np.ndarray, n_clusters: int = 10) -> Dict:
        """Evaluate clustering quality using K-Means and silhouette score"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            return {"Clustering-Silhouette": score, "Clusters": n_clusters}
        except Exception as e:
            self.logger.error(f"Clustering evaluation error: {e}")
            return {"Clustering-Silhouette": None, "Clusters": n_clusters}
    
    def evaluate_embeddings(self, embeddings_dir: str, vocab_file: str, output_dir: str) -> None:
        """Main evaluation function with enhanced error handling and visualizations"""
        try:
            start_time = time.perf_counter()
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            vocab = self._load_vocab(vocab_file)
            datasets = self.load_evaluation_datasets()
            
            results = self._process_embeddings(embeddings_dir, vocab, datasets)
            summary_df = self._create_summary_dataframe(results)
            
            self.create_enhanced_visualizations(summary_df, output_dir)
            self._save_results(summary_df, results, output_dir)
            
            processing_time = time.perf_counter() - start_time
            self._log_completion_stats(processing_time)
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"{Fore.RED}Evaluation failed: {str(e)}{Style.RESET_ALL}")
            raise
    
    def _load_vocab(self, vocab_file: str) -> Dict:
        """Load vocabulary from JSON file with error handling"""
        try:
            with open(vocab_file, 'r') as f:
                vocab = json.load(f)
            return vocab
        except Exception as e:
            self.logger.error(f"{Fore.RED}Failed to load vocabulary: {str(e)}{Style.RESET_ALL}")
            raise
    
    def load_evaluation_datasets(self) -> Dict:
        """Load evaluation datasets (example implementation)"""
        try:
            datasets = {}
            # Load SimLex-999 dataset
            datasets['simlex'] = pd.read_csv("test_data/SimLex-999.txt", sep='\t')
            # Load WordSim-353 dataset
            datasets['wordsim'] = pd.read_csv("test_data/wordsim/combined.csv")
            return datasets
        except Exception as e:
            self.logger.error(f"{Fore.RED}Failed to load evaluation datasets: {str(e)}{Style.RESET_ALL}")
            raise
    
    def _process_embeddings(self, embeddings_dir: str, vocab: Dict, datasets: Dict) -> List[Dict]:
        """Process embedding files and compute evaluation metrics"""
        results = []
        embedding_files = list(Path(embeddings_dir).glob("embeddings_*.npy"))
        # Define default analogy questions (ensure these words exist in your vocabulary)
        default_analogies = [
            ("king", "queen", "man", "woman"),
            ("paris", "france", "london", "england"),
            ("big", "biggest", "small", "smallest")
        ]
        for emb_file in tqdm(embedding_files, desc="Processing embedding files"):
            self.logger.info(f"\nEvaluating {emb_file.name}")
            try:
                embeddings = np.load(emb_file)
                params = self._parse_embedding_params(emb_file.stem)
                # (Placeholders for SimLex and WordSim metrics; replace with your actual implementations)
                simlex_corr = float(np.random.rand())
                simlex_cov = float(np.random.rand())
                wordsim_corr = float(np.random.rand())
                wordsim_cov = float(np.random.rand())
                
                # Evaluate analogies and clustering quality
                analogy_results = self.evaluate_analogies(embeddings, vocab, default_analogies)
                clustering_results = self.evaluate_clustering(embeddings, n_clusters=10)
                
                result = {**params,
                          "SimLex-Correlation": simlex_corr,
                          "SimLex-Coverage": simlex_cov,
                          "WordSim-Correlation": wordsim_corr,
                          "WordSim-Coverage": wordsim_cov,
                          "Analogy-Accuracy": analogy_results["Analogy-Accuracy"],
                          "Clustering-Silhouette": clustering_results["Clustering-Silhouette"]}
                results.append(result)
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"{Fore.RED}Failed processing {emb_file.name}: {str(e)}{Style.RESET_ALL}")
        return results
    
    def _create_summary_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Create a summary DataFrame from the results"""
        try:
            summary_df = pd.DataFrame(results)
            return summary_df
        except Exception as e:
            self.logger.error(f"{Fore.RED}Failed to create summary DataFrame: {str(e)}{Style.RESET_ALL}")
            raise
    
    def _save_results(self, summary_df: pd.DataFrame, results: List[Dict], output_dir: str) -> None:
        """Save evaluation results and summary to files"""
        try:
            summary_csv_path = Path(output_dir) / "evaluation_summary.csv"
            summary_df.to_csv(summary_csv_path, index=False)
            self.logger.info(f"Saved evaluation summary CSV to {summary_csv_path}")
        except Exception as e:
            self.logger.error(f"{Fore.RED}Failed to save results: {str(e)}{Style.RESET_ALL}")
            raise
    
    def _log_completion_stats(self, processing_time: float) -> None:
        """Log completion statistics with color coding"""
        self.logger.info(f"\n{Fore.GREEN}Evaluation Summary:{Style.RESET_ALL}")
        self.logger.info(f"Processing Time: {processing_time:.2f} seconds")
        self.logger.info(f"Errors Encountered: {self.error_count}")
        self.logger.info(f"Warnings Generated: {self.warning_count}")
        
        if self.error_count == 0 and self.warning_count == 0:
            self.logger.info(f"{Fore.GREEN}Evaluation completed successfully!{Style.RESET_ALL}")
        else:
            self.logger.warning(f"{Fore.YELLOW}Evaluation completed with issues.{Style.RESET_ALL}")
    
    def _parse_embedding_params(self, stem: str) -> Dict:
        """
        Parse parameters from the embedding filename.
        For example, for a filename like 'embeddings_w10_ppmi_d100',
        extract the window size, normalization method, and dimensions.
        """
        parts = stem.split('_')
        try:
            window_size = int(parts[1][1:])  # remove leading 'w'
            norm_method = parts[2]
            dims = int(parts[3][1:])         # remove leading 'd'
        except Exception as e:
            self.logger.error(f"Error parsing parameters from filename {stem}: {e}")
            window_size, norm_method, dims = None, None, None
        return {"Window": window_size, "Normalization": norm_method, "Dimensions": dims}

def main():
    try:
        evaluator = EnhancedEmbeddingEvaluator()
        evaluator.evaluate_embeddings(
            embeddings_dir="./processed_data/embeddings",
            vocab_file="./processed_data/cooccurrence_matrices/vocabulary.json",
            output_dir="./evaluation_results"
        )
    except Exception as e:
        print(f"{Fore.RED}Fatal error: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main()
