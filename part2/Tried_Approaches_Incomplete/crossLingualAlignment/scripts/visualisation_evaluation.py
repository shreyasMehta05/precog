import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import KeyedVectors
import os
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class AlignmentVisualizer:
    def __init__(self, aligned_dir, tgt_emb_path, concept_groups_path):
        """
        Initialize visualizer with paths to aligned embeddings and concept groups
        
        Args:
            aligned_dir: Directory containing aligned embeddings and vocabulary
            tgt_emb_path: Path to target (English) embeddings
            concept_groups_path: Path to file containing grouped concepts for visualization
        """
        self.vocab = self._load_vocabulary(os.path.join(aligned_dir, 'vocabulary.txt'))
        self.alignments = {
            'procrustes': np.load(os.path.join(aligned_dir, 'aligned_procrustes.npy')),
            'cca': np.load(os.path.join(aligned_dir, 'aligned_cca.npy')),
            'optimal_transport': np.load(os.path.join(aligned_dir, 'aligned_optimal_transport.npy')),
            'iterative': np.load(os.path.join(aligned_dir, 'aligned_iterative.npy'))
        }
        
        self.tgt_emb = KeyedVectors.load_word2vec_format(tgt_emb_path)
        self.concept_groups = self._load_concept_groups(concept_groups_path)
        
        # Set up plotting style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def _load_concept_groups(self, groups_path):
        """
        Load concept groups from file
        Expected format: category\tsrc_word\ttgt_word
        """
        groups = {}
        with open(groups_path, 'r', encoding='utf-8') as f:
            for line in f:
                category, src_word, tgt_word = line.strip().split('\t')
                if src_word in self.vocab and tgt_word in self.tgt_emb:
                    if category not in groups:
                        groups[category] = []
                    groups[category].append((src_word, tgt_word))
        return groups
    
    def visualize_embeddings(self, method, output_dir, n_components=2, perplexity=30):
        """
        Create t-SNE visualization of aligned embeddings
        
        Args:
            method: Alignment method to visualize
            output_dir: Directory to save visualizations
            n_components: Number of components for dimensionality reduction
            perplexity: Perplexity parameter for t-SNE
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect vectors for visualization
        vectors = []
        labels = []
        languages = []
        categories = []
        
        for category, word_pairs in self.concept_groups.items():
            for src_word, tgt_word in word_pairs:
                # Add source word vector
                src_idx = self.vocab.index(src_word)
                vectors.append(self.alignments[method][src_idx])
                labels.append(src_word)
                languages.append('Hindi')
                categories.append(category)
                
                # Add target word vector
                vectors.append(self.tgt_emb[tgt_word])
                labels.append(tgt_word)
                languages.append('English')
                categories.append(category)
        
        vectors = np.vstack(vectors)
        
        # Reduce dimensionality
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        reduced_vectors = tsne.fit_transform(vectors)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot points
        for i, category in enumerate(set(categories)):
            mask = np.array(categories) == category
            
            # Plot Hindi words
            hindi_mask = mask & (np.array(languages) == 'Hindi')
            plt.scatter(
                reduced_vectors[hindi_mask, 0],
                reduced_vectors[hindi_mask, 1],
                label=f'{category} (Hindi)',
                marker='o'
            )
            
            # Plot English words
            english_mask = mask & (np.array(languages) == 'English')
            plt.scatter(
                reduced_vectors[english_mask, 0],
                reduced_vectors[english_mask, 1],
                label=f'{category} (English)',
                marker='^'
            )
        
        plt.title(f'Cross-lingual Word Embeddings ({method})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'tsne_{method}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_similarity_heatmap(self, method, output_dir):
        """
        Create heatmap of cross-lingual similarities within concept groups
        
        Args:
            method: Alignment method to visualize
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for category, word_pairs in self.concept_groups.items():
            src_words = [pair[0] for pair in word_pairs]
            tgt_words = [pair[1] for pair in word_pairs]
            
            # Compute similarities
            src_vectors = np.vstack([self.alignments[method][self.vocab.index(w)] for w in src_words])
            tgt_vectors = np.vstack([self.tgt_emb[w] for w in tgt_words])
            
            similarities = cosine_similarity(src_vectors, tgt_vectors)
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                similarities,
                xticklabels=tgt_words,
                yticklabels=src_words,
                cmap='YlOrRd',
                annot=True,
                fmt='.2f'
            )
            plt.title(f'Cross-lingual Similarities - {category} ({method})')
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f'heatmap_{method}_{category}.png'),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
    
    def visualize_all(self, output_dir):
        """
        Generate all visualizations for all methods
        
        Args:
            output_dir: Directory to save visualizations
        """
        for method in self.alignments.keys():
            print(f"Generating visualizations for {method}...")
            self.visualize_embeddings(method, output_dir)
            self.plot_similarity_heatmap(method, output_dir)

if __name__ == "__main__":
    visualizer = AlignmentVisualizer(
        aligned_dir="output_alignments",
        tgt_emb_path="data/english_embeddings.vec",
        concept_groups_path="data/concept_groups.txt"
    )
    
    visualizer.visualize_all("visualization_results")