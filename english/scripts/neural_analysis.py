#!/usr/bin/env python3
import os
import time
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from colorama import Fore, Style, init
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# Initialize colorama (with auto reset so colors do not persist)
init(autoreset=True)

class NeuralEmbeddingEvaluator:
    """Evaluation suite for pre-trained neural word embeddings."""
    
    # Define color schemes for visualizations
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
        """Setup logging with file and console outputs."""
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"neural_embedding_evaluation_{timestamp}.log"
        
        self.logger = logging.getLogger('NeuralEmbeddingEvaluator')
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
        """Setup GPU with error handling."""
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
            

    def load_neural_embeddings(self, embedding_file: str) -> (np.ndarray, dict):
        """
        Load pre-trained neural embeddings using gensim.
        For GloVe files (which are text files), load directly with no_header=True.
        """
        try:
            self.logger.info(f"Loading neural embeddings from {embedding_file}")
            # Load directly using no_header=True for GloVe text files.
            model = KeyedVectors.load_word2vec_format(embedding_file, binary=False, no_header=True)
            embeddings = model.vectors  # shape: (vocab_size, vector_dim)
            vocab = model.key_to_index  # dict mapping word -> index
            self.logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
            return embeddings, vocab
        except Exception as e:
            self.logger.error(f"{Fore.RED}Failed to load neural embeddings: {str(e)}{Style.RESET_ALL}")
            sys.exit(1)


    
    def load_evaluation_datasets(self) -> dict:
        """
        Load evaluation datasets.
        Here we assume you have a SimLex-999 and WordSim-353 dataset stored in 'test_data/'.
        """
        try:
            datasets = {}
            datasets['simlex'] = pd.read_csv("test_data/SimLex-999.txt", sep='\t')
            datasets['wordsim'] = pd.read_csv("test_data/wordsim/combined.csv")
            return datasets
        except Exception as e:
            self.logger.error(f"{Fore.RED}Failed to load evaluation datasets: {str(e)}{Style.RESET_ALL}")
            sys.exit(1)
    
    def evaluate_analogies(self, embeddings: np.ndarray, vocab: dict,
                           analogies: list) -> dict:
        """
        Evaluate word analogies using vector arithmetic.
        Analogies is a list of tuples like (a, b, c, d) where a:b :: c:d.
        """
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
                similarities[vocab[a]] = -1
                similarities[vocab[b]] = -1
                similarities[vocab[c]] = -1
                pred_idx = similarities.argmax().item()
                if pred_idx == vocab[d]:
                    correct += 1
                total += 1
        accuracy = correct / total if total > 0 else 0
        return {"Analogy-Accuracy": accuracy, "Analogy-Total": total}
    
    def evaluate_clustering(self, embeddings: np.ndarray, n_clusters: int = 10) -> dict:
        """Evaluate clustering quality using K-Means and the silhouette score."""
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            return {"Clustering-Silhouette": score, "Clusters": n_clusters}
        except Exception as e:
            self.logger.error(f"Clustering evaluation error: {e}")
            return {"Clustering-Silhouette": None, "Clusters": n_clusters}
    
    def create_enhanced_visualizations(self, df: pd.DataFrame, output_dir: str) -> None:
        """Create visualizations (plots and heatmaps) from evaluation results."""
        try:
            self._create_correlation_matrix(df, output_dir)
            self._create_parameter_distributions(df, output_dir)
            self._create_performance_comparisons(df, output_dir)
            self._create_3d_surface_plots(df, output_dir)
            self._create_additional_plots(df, output_dir)
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"{Fore.RED}Visualization creation failed: {str(e)}{Style.RESET_ALL}")
    
    def _create_correlation_matrix(self, df: pd.DataFrame, output_dir: str) -> None:
        """Generate a correlation matrix heatmap."""
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
        fig.write_html(Path(output_dir) / "correlation_matrix.html")
    
    def _create_parameter_distributions(self, df: pd.DataFrame, output_dir: str) -> None:
        """Generate histograms for various evaluation parameters."""
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            'SimLex Correlation Distribution',
            'WordSim Correlation Distribution',
            'Analogy Accuracy Distribution',
            'Clustering Silhouette Distribution'
        ))
        fig.add_trace(
            go.Histogram(x=df['SimLex-Correlation'], nbinsx=20,
                         marker_color=self.COLOR_SCHEMES['categorical'][0]),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=df['WordSim-Correlation'], nbinsx=20,
                         marker_color=self.COLOR_SCHEMES['categorical'][1]),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=df['Analogy-Accuracy'], nbinsx=20,
                         marker_color=self.COLOR_SCHEMES['categorical'][2]),
            row=2, col=1
        )
        fig.add_trace(
            go.Histogram(x=df['Clustering-Silhouette'], nbinsx=20,
                         marker_color=self.COLOR_SCHEMES['categorical'][3]),
            row=2, col=2
        )
        fig.update_layout(height=1000, width=1200, showlegend=False,
                          title_text="Evaluation Metric Distributions")
        fig.write_html(Path(output_dir) / "parameter_distributions.html")
    
    def _create_performance_comparisons(self, df: pd.DataFrame, output_dir: str) -> None:
        """Generate comparison plots for evaluation metrics."""
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            'Performance Comparison (SimLex)',
            'Performance Comparison (WordSim)',
            'Analogy Accuracy',
            'Clustering Silhouette'
        ))
        for metric, row, col in [
            ('SimLex-Correlation', 1, 1),
            ('WordSim-Correlation', 1, 2),
            ('Analogy-Accuracy', 2, 1),
            ('Clustering-Silhouette', 2, 2)
        ]:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[metric],
                           mode='lines+markers',
                           name=metric),
                row=row, col=col
            )
        fig.update_layout(height=1200, width=1600, title_text="Comprehensive Performance Analysis")
        fig.write_html(Path(output_dir) / "performance_comparisons.html")
    
    def _create_3d_surface_plots(self, df: pd.DataFrame, output_dir: str) -> None:
        """Generate 3D surface plots (if applicable)."""
        if 'Window' in df.columns and 'Dimensions' in df.columns:
            for norm in df['Normalization'].unique():
                subset = df[df['Normalization'] == norm]
                try:
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
                            yaxis_title='Window',
                            zaxis_title='SimLex Correlation'
                        ),
                        width=1000, height=800
                    )
                    fig.write_html(Path(output_dir) / f"surface_plot_{norm}.html")
                except Exception as ex:
                    self.logger.warning(f"Could not create 3D surface for {norm}: {ex}")
    
    def _create_additional_plots(self, summary_df: pd.DataFrame, output_dir: str) -> None:
        """Generate additional PNG plots for analogy accuracy and clustering quality."""
        config_labels = summary_df['Normalization'] + " (d=" + summary_df['Dimensions'].astype(str) + ")"
        plt.figure(figsize=(8, 6))
        plt.bar(config_labels, summary_df['Analogy-Accuracy'], color='skyblue')
        plt.xlabel("Embedding Configuration")
        plt.ylabel("Analogy Accuracy")
        plt.title("Word Analogy Task Accuracy")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "analogy_accuracy.png")
        plt.close()
        
        plt.figure(figsize=(8, 6))
        plt.bar(config_labels, summary_df['Clustering-Silhouette'], color='salmon')
        plt.xlabel("Embedding Configuration")
        plt.ylabel("Silhouette Score")
        plt.title("Clustering Quality Evaluation")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "clustering_quality.png")
        plt.close()
    
    def evaluate_embedding(self, embedding_file: str, output_dir: str) -> None:
        """
        Main evaluation function.
          - Loads neural embeddings.
          - Loads evaluation datasets.
          - Computes evaluation metrics (using placeholder values for some metrics).
          - Evaluates analogy and clustering tasks.
          - Creates visualizations and saves summary CSV and JSON.
        """
        try:
            start_time = time.perf_counter()
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Load neural embeddings using gensim
            embeddings, vocab = self.load_neural_embeddings(embedding_file)
            
            # Load evaluation datasets (SimLex, WordSim)
            datasets = self.load_evaluation_datasets()
            
            # For demonstration, we use placeholder (random) values for SimLex/WordSim metrics.
            simlex_corr = float(np.random.rand())
            wordsim_corr = float(np.random.rand())
            
            # Define some default analogy questions (ensure these words are in your vocab)
            default_analogies = [
                ("king", "queen", "man", "woman"),
                ("paris", "france", "london", "england"),
                ("big", "biggest", "small", "smallest")
            ]
            analogy_results = self.evaluate_analogies(embeddings, vocab, default_analogies)
            clustering_results = self.evaluate_clustering(embeddings, n_clusters=10)
            
            # For demonstration, include dummy parameters for configuration.
            result = {
                "Normalization": "neural",
                "Dimensions": embeddings.shape[1],
                "SimLex-Correlation": simlex_corr,
                "WordSim-Correlation": wordsim_corr,
                "Analogy-Accuracy": analogy_results["Analogy-Accuracy"],
                "Analogy-Total": analogy_results["Analogy-Total"],
                "Clustering-Silhouette": clustering_results["Clustering-Silhouette"]
            }
            results = [result]
            summary_df = pd.DataFrame(results)
            
            # Save summary results as CSV and JSON
            summary_csv_path = Path(output_dir) / "neural_evaluation_summary.csv"
            summary_df.to_csv(summary_csv_path, index=False)
            json_path = Path(output_dir) / "neural_evaluation_summary.json"
            summary_df.to_json(json_path, orient='records', indent=4)
            self.logger.info(f"Saved evaluation summary to CSV and JSON in {output_dir}")
            
            # Create visualizations
            self.create_enhanced_visualizations(summary_df, output_dir)
            
            processing_time = time.perf_counter() - start_time
            self.logger.info(f"\n{Fore.GREEN}Evaluation completed in {processing_time:.2f} seconds{Style.RESET_ALL}")
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"{Fore.RED}Evaluation failed: {str(e)}{Style.RESET_ALL}")
            sys.exit(1)

def main():
    try:
        evaluator = NeuralEmbeddingEvaluator()
        # Define the folder that contains all GloVe files
        glove_folder = "./pretrained_models/neural_embeddings.bin"
        # Define a base output directory for neural evaluations
        output_base_dir = "./neural_evaluation_results"
        os.makedirs(output_base_dir, exist_ok=True)
        
        # Process each GloVe file (assumed to be .txt files) in the folder
        glove_files = sorted(Path(glove_folder).glob("*.txt"))
        for glove_file in glove_files:
            evaluator.logger.info(f"\n{Fore.CYAN}Evaluating embeddings for {glove_file}{Style.RESET_ALL}")
            # Create a separate output folder for each file based on its stem
            current_output_dir = os.path.join(output_base_dir, glove_file.stem)
            evaluator.evaluate_embedding(str(glove_file), current_output_dir)
    except Exception as e:
        print(f"{Fore.RED}Fatal error: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main()
