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
import fasttext
import fasttext.util

# Initialize colorama
init(autoreset=True)

class HindiNeuralEmbeddingEvaluator:
    """Evaluation suite for Hindi pre-trained neural word embeddings."""
    
    COLOR_SCHEMES = {
        'categorical': px.colors.qualitative.Set3,
        'sequential': px.colors.sequential.Viridis,
        'diverging': px.colors.diverging.RdYlBu,
        'correlation': ['#FF0000', '#FFFFFF', '#0000FF']
    }
    
    def __init__(self, log_dir: str = "logs"):
        self.setup_logging(log_dir)
        self._setup_gpu()  # GPU setup (or remove if not needed)
        self.error_count = 0
        self.warning_count = 0
        self.supported_models = {
            'fasttext': 'cc.hi.300.bin',  # Hindi FastText
            'word2vec': 'hi_vectors_word2vec_format.bin',  # Hindi Word2Vec
            'glove': 'hi.840B.300d.txt'  # Hindi GloVe
        }
    
    def _setup_gpu(self):
        """Simple GPU setup (optional)"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

    def setup_logging(self, log_dir: str):
        """Setup logging with enhanced Hindi support"""
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"hindi_neural_evaluation_{timestamp}.log"
        
        self.logger = logging.getLogger('HindiNeuralEmbeddingEvaluator')
        self.logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(file_formatter)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(console_formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def load_hindi_embeddings(self, model_path: str, model_type: str) -> (np.ndarray, dict):
        """
        Load pre-trained Hindi embeddings based on model type.
        Supports FastText, Word2Vec, and GloVe formats.
        """
        try:
            self.logger.info(f"Loading Hindi embeddings from {model_path}")
            
            if model_type == 'fasttext':
                model = fasttext.load_model(model_path)
                # Load vocabulary from JSON file (update the path if needed)
                with open('./processed_data/hindi_cooccurrence_matrices/hindi_vocabulary.json', 'r', encoding='utf-8') as f:
                    vocab_dict = json.load(f)
                # Extract words from the JSON keys (you may sort by index if required)
                words = list(vocab_dict.keys())
                vectors = [model.get_word_vector(word) for word in words]
                embeddings = np.array(vectors)
                # Recreate the mapping (or use the provided indices if preferred)
                vocab = {word: idx for idx, word in enumerate(words)}
                
            elif model_type == 'word2vec':
                model = KeyedVectors.load_word2vec_format(model_path, binary=True)
                embeddings = model.vectors
                vocab = model.key_to_index
                
            elif model_type == 'glove':
                temp_converted = model_path + '.word2vec'
                glove2word2vec(model_path, temp_converted)
                model = KeyedVectors.load_word2vec_format(temp_converted)
                embeddings = model.vectors
                vocab = model.key_to_index
                os.remove(temp_converted)
                
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            self.logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
            return embeddings, vocab
            
        except Exception as e:
            self.logger.error(f"{Fore.RED}Failed to load Hindi embeddings: {str(e)}{Style.RESET_ALL}")
            sys.exit(1)

    def load_hindi_evaluation_datasets(self) -> dict:
        """
        Load Hindi-specific evaluation datasets.
        This includes a Hindi WordSim file (for similarity scoring) and a Hindi analogies dataset.
        If the analogies file is not found, default analogies are used.
        """
        try:
            datasets = {}
            # Load Hindi WordSim dataset from CSV.
            # Read the CSV file and rename columns appropriately.
            wordsim_df = pd.read_csv("test_data/Dataset-Hindi-WordSim353.csv", encoding='utf-8')
            # Assume the CSV has columns: "S. No.", "Word Pair", an empty column, and "Similarity Score"
            # Rename columns to: "S.No.", "word1", "word2", "similarity"
            wordsim_df.columns = ["S.No.", "word1", "word2", "similarity"]
            datasets['hindi_wordsim'] = wordsim_df
            
            # Load Hindi analogies dataset
            try:
                analogies = []
                with open("test_data/hindi_analogies.txt", 'r', encoding='utf-8') as f:
                    for line in f:
                        words = line.strip().split()
                        if len(words) == 4:
                            analogies.append(tuple(words))
                datasets['hindi_analogies'] = analogies
            except FileNotFoundError:
                self.logger.warning("Hindi analogies file not found. Using default analogies.")
                # Define your own default analogies here (using Hindi examples)
                datasets['hindi_analogies'] = [
                    ("राजा", "रानी", "पुरुष", "स्त्री"),
                    ("पेरिस", "फ्रांस", "लंदन", "इंग्लैंड"),
                    ("बड़ा", "सबसे बड़ा", "छोटा", "सबसे छोटा")
                ]
            
            return datasets
        except Exception as e:
            self.logger.error(f"{Fore.RED}Failed to load Hindi evaluation datasets: {str(e)}{Style.RESET_ALL}")
            sys.exit(1)

    def evaluate_hindi_analogies(self, embeddings: np.ndarray, vocab: dict,
                                 analogies: list) -> dict:
        """Evaluate Hindi word analogies with cultural and linguistic considerations"""
        correct = 0
        total = 0
        skipped = 0
        embeddings_tensor = torch.tensor(embeddings, device=self.device)
        
        for a, b, c, d in analogies:
            try:
                if a in vocab and b in vocab and c in vocab and d in vocab:
                    vec_a = embeddings_tensor[vocab[a]]
                    vec_b = embeddings_tensor[vocab[b]]
                    vec_c = embeddings_tensor[vocab[c]]
                    expected = vec_b - vec_a + vec_c
                    
                    similarities = torch.nn.functional.cosine_similarity(
                        expected.unsqueeze(0),
                        embeddings_tensor
                    )
                    
                    # Exclude source words from the search
                    similarities[vocab[a]] = -1
                    similarities[vocab[b]] = -1
                    similarities[vocab[c]] = -1
                    
                    pred_idx = similarities.argmax().item()
                    if pred_idx == vocab[d]:
                        correct += 1
                    total += 1
                else:
                    skipped += 1
            except Exception as e:
                self.logger.warning(f"Error processing analogy {(a, b, c, d)}: {str(e)}")
                skipped += 1
                
        accuracy = correct / total if total > 0 else 0
        coverage = total / (total + skipped) if (total + skipped) > 0 else 0
        
        return {
            "Analogy-Accuracy": accuracy,
            "Analogy-Coverage": coverage,
            "Analogy-Total": total,
            "Analogy-Skipped": skipped
        }
    
    def evaluate_clustering(self, embeddings: np.ndarray, n_clusters: int = 10) -> dict:
        """Perform a dummy clustering evaluation using KMeans and compute silhouette score"""
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            silhouette = silhouette_score(embeddings, labels)
            self.logger.info(f"Clustering silhouette score: {silhouette:.4f}")
            return {"Clustering-Silhouette": silhouette}
        except Exception as e:
            self.logger.error(f"Clustering evaluation error: {str(e)}")
            return {"Clustering-Silhouette": None}
    
    def create_enhanced_visualizations(self, df: pd.DataFrame, output_dir: str) -> None:
        """Create an HTML table visualization of the evaluation results"""
        try:
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(df.columns)),
                cells=dict(values=[df[col] for col in df.columns])
            )])
            fig.write_html(os.path.join(output_dir, "evaluation_summary.html"))
            self.logger.info("Created enhanced visualizations.")
        except Exception as e:
            self.logger.error(f"Visualization creation error: {str(e)}")
    
    def save_results(self, df: pd.DataFrame, output_dir: str) -> None:
        """Save evaluation results as CSV and JSON with proper UTF-8 encoding"""
        try:
            csv_path = Path(output_dir) / "hindi_evaluation_summary.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            json_path = Path(output_dir) / "hindi_evaluation_summary.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(df.to_dict('records'), f, ensure_ascii=False, indent=4)
                
            self.logger.info(f"Saved evaluation results to {output_dir}")
        except Exception as e:
            self.logger.error(f"{Fore.RED}Failed to save results: {str(e)}{Style.RESET_ALL}")

    def evaluate_embedding(self, model_path: str, model_type: str, output_dir: str) -> None:
        """Main evaluation function for Hindi embeddings"""
        try:
            start_time = time.perf_counter()
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Load embeddings and evaluation datasets
            embeddings, vocab = self.load_hindi_embeddings(model_path, model_type)
            datasets = self.load_hindi_evaluation_datasets()
            
            # Evaluate analogies
            analogy_results = self.evaluate_hindi_analogies(
                embeddings,
                vocab,
                datasets['hindi_analogies']
            )
            
            # Evaluate clustering
            clustering_results = self.evaluate_clustering(embeddings, n_clusters=10)
            
            # Combine results into a summary dictionary
            result = {
                "Model-Type": model_type,
                "Dimensions": embeddings.shape[1],
                "Vocabulary-Size": len(vocab),
                "Analogy-Accuracy": analogy_results["Analogy-Accuracy"],
                "Analogy-Coverage": analogy_results["Analogy-Coverage"],
                "Clustering-Silhouette": clustering_results["Clustering-Silhouette"]
            }
            
            # Save results
            results_df = pd.DataFrame([result])
            self.save_results(results_df, output_dir)
            
            # Create visualizations
            self.create_enhanced_visualizations(results_df, output_dir)
            
            processing_time = time.perf_counter() - start_time
            self.logger.info(f"\n{Fore.GREEN}Evaluation completed in {processing_time:.2f} seconds{Style.RESET_ALL}")
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"{Fore.RED}Evaluation failed: {str(e)}{Style.RESET_ALL}")
            sys.exit(1)

def main():
    try:
        evaluator = HindiNeuralEmbeddingEvaluator()
        
        # Set the base directories for the pretrained models and output results
        model_base_dir = "./pretrained_models/hindi_neural_embeddings"
        output_base_dir = "./hindi_evaluation_results"
        
        # Here we target only the FastText model (cc.hi.300.bin) available in your directory.
        model_type = 'fasttext'
        model_file = evaluator.supported_models[model_type]
        
        evaluator.logger.info(f"\n{Fore.CYAN}Evaluating Hindi {model_type} embeddings{Style.RESET_ALL}")
        model_path = os.path.join(model_base_dir, model_file)
        output_dir = os.path.join(output_base_dir, model_type)
        evaluator.evaluate_embedding(model_path, model_type, output_dir)
            
    except Exception as e:
        print(f"{Fore.RED}Fatal error: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main()
