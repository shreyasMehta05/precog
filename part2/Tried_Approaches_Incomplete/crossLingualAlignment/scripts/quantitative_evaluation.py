import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from gensim.models import KeyedVectors
import pandas as pd
import os
import json
from tqdm import tqdm

class CrossLingualEvaluator:
    def __init__(self, aligned_dir, tgt_emb_path, test_dict_path, similarity_dataset_path, use_gpu=True):
        """
        Initialize evaluator with paths to aligned embeddings and evaluation data
        
        Args:
            aligned_dir: Directory containing aligned embeddings and vocabulary
            tgt_emb_path: Path to target (English) embeddings
            test_dict_path: Path to test dictionary for word translation
            similarity_dataset_path: Path to cross-lingual similarity dataset
            use_gpu: Whether to use GPU acceleration
        """
        # Determine device
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.vocab = self._load_vocabulary(os.path.join(aligned_dir, 'vocabulary.txt'))
        
        # Load alignments to GPU
        self.alignments = {
            method: torch.tensor(np.load(os.path.join(aligned_dir, f'aligned_{method}.npy')), 
                                 dtype=torch.float32).to(self.device)
            for method in ['procrustes', 'cca', 'optimal_transport']
        }
        
        print("Loading target embeddings...")
        self.tgt_emb = KeyedVectors.load_word2vec_format(tgt_emb_path)
        print(f"Loaded {len(self.tgt_emb.index_to_key)} target words.")
        
        # Convert target embeddings to GPU tensor
        self.tgt_vectors = torch.tensor(self.tgt_emb.vectors, dtype=torch.float32).to(self.device)
        
        self.test_dict = self._load_test_dictionary(test_dict_path)
        self.similarity_data = self._load_similarity_dataset(similarity_dataset_path)
    
    def _load_vocabulary(self, vocab_path):
        """Load vocabulary file (one word per line)."""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def _load_test_dictionary(self, dict_path):
        """Load test dictionary for word translation. Expected format: src_word <tab> tgt_word per line."""
        word_pairs = []
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    src, tgt = parts[:2]
                    # Ensure both words are present
                    if src in self.vocab and tgt in self.tgt_emb.key_to_index:
                        word_pairs.append((src, tgt))
        return word_pairs
    
    def _load_similarity_dataset(self, dataset_path):
        """Load cross-lingual similarity dataset. Expected format: src_word <tab> tgt_word <tab> score per line."""
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    src_word, tgt_word, score_str = parts[:3]
                    try:
                        score = float(score_str)
                    except ValueError:
                        continue
                    # Ensure words exist in our vocabulary
                    if src_word in self.vocab and tgt_word in self.tgt_emb.key_to_index:
                        data.append((src_word, tgt_word, score))
        return data
    
    def evaluate_word_translation(self, method, k=(1, 5, 10)):
        """
        Evaluate word translation accuracy using precision@k with GPU acceleration.
        
        Args:
            method: Alignment method to evaluate.
            k: Tuple of k values for precision@k.
        
        Returns:
            Dictionary containing precision@k for each value.
        """
        aligned_vectors = self.alignments[method]
        results = {}
        
        for k_val in k:
            correct = 0
            total = 0
            
            for src_word, tgt_word in tqdm(self.test_dict, desc=f"Evaluating word translation (P@{k_val})"):
                # Ensure the word is in the vocabulary
                if src_word not in self.vocab:
                    continue
                
                src_idx = self.vocab.index(src_word)
                src_vec = aligned_vectors[src_idx].unsqueeze(0)
                
                # Compute cosine similarities using GPU
                sims = F.cosine_similarity(src_vec, self.tgt_vectors).cpu().numpy()
                
                # Find k nearest neighbors
                top_k_indices = np.argsort(-sims)[:k_val]
                top_k_words = [self.tgt_emb.index_to_key[idx] for idx in top_k_indices]
                
                if tgt_word in top_k_words:
                    correct += 1
                total += 1
            
            results[f'P@{k_val}'] = correct / total if total > 0 else 0
        
        return results

    def evaluate_semantic_similarity(self, method):
        """
        Evaluate semantic similarity preservation with GPU acceleration.
        
        Args:
            method: Alignment method to evaluate.
        
        Returns:
            Dictionary containing Spearman correlation and mean absolute error.
        """
        aligned_vectors = self.alignments[method]
        predicted_scores = []
        true_scores = []
        
        for src_word, tgt_word, score in tqdm(self.similarity_data, desc="Evaluating semantic similarity"):
            if src_word not in self.vocab:
                continue
            
            src_idx = self.vocab.index(src_word)
            src_vec = aligned_vectors[src_idx].unsqueeze(0)
            tgt_vec = torch.tensor(self.tgt_emb[tgt_word], dtype=torch.float32).to(self.device).unsqueeze(0)
            
            # Compute cosine similarity using GPU
            pred_score = F.cosine_similarity(src_vec, tgt_vec).item()
            predicted_scores.append(pred_score)
            true_scores.append(score)
        
        correlation, _ = spearmanr(predicted_scores, true_scores)
        mae = np.mean(np.abs(np.array(predicted_scores) - np.array(true_scores)))
        
        return {'spearman_correlation': correlation, 'mean_absolute_error': mae}
    
    def evaluate_hubness(self, method, k=10):
        """
        Evaluate hubness in the cross-lingual space with GPU acceleration.
        
        Args:
            method: Alignment method to evaluate.
            k: Number of nearest neighbors to consider.
        
        Returns:
            Dictionary containing hubness statistics.
        """
        aligned_vectors = self.alignments[method]
        
        # Compute cosine similarity matrix using GPU
        sim_matrix = F.cosine_similarity(
            aligned_vectors.unsqueeze(1), 
            self.tgt_vectors.unsqueeze(0), 
            dim=-1
        ).cpu().numpy()
        
        top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :k]
        hub_counts = np.bincount(top_k_indices.flatten())
        
        return {
            'max_hub_size': int(np.max(hub_counts)),
            'mean_hub_size': float(np.mean(hub_counts)),
            'std_hub_size': float(np.std(hub_counts)),
            'skewness': float(np.mean(((hub_counts - np.mean(hub_counts)) / np.std(hub_counts)) ** 3))
        }
    
    def evaluate_all(self, output_path):
        """
        Run all evaluations and save the results.
        
        Args:
            output_path: Path to save evaluation results (JSON format).
        
        Returns:
            Dictionary with evaluation results for each alignment method.
        """
        results = {}
        
        for method in self.alignments.keys():
            method_results = {}
            # Evaluate word translation
            translation_results = self.evaluate_word_translation(method)
            method_results['word_translation'] = translation_results
            
            # Evaluate semantic similarity
            similarity_results = self.evaluate_semantic_similarity(method)
            method_results['semantic_similarity'] = similarity_results
            
            # Evaluate hubness
            hubness_results = self.evaluate_hubness(method)
            method_results['hubness'] = hubness_results
            
            results[method] = method_results
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

if __name__ == "__main__":
    # Example usage with GPU configuration
    evaluator = CrossLingualEvaluator(
        aligned_dir="output_alignments",
        tgt_emb_path="data/english_embeddings.vec",
        test_dict_path="data/hi_en_dictionary.txt",
        similarity_dataset_path="data/similarity_dataset.txt",
        use_gpu=True  # Easy GPU toggle
    )
    
    results = evaluator.evaluate_all("evaluation_results.json")
    
    # Print summary of results
    for method, scores in results.items():
        print(f"\nResults for {method}:")
        print(f"P@1: {scores['word_translation']['P@1']:.3f}")
        print(f"Semantic similarity correlation: {scores['semantic_similarity']['spearman_correlation']:.3f}")
        print(f"Hubness skewness: {scores['hubness']['skewness']:.3f}")