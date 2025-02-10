"""
Static Word Embeddings Bias Evaluation Script
Evaluates harmful associations in static word embeddings using WEAT-like metrics.

Requirements:
pip install gensim numpy tqdm torch
"""

import argparse
import numpy as np
from gensim.models import KeyedVectors
from typing import List, Set, Dict
import torch
from tqdm import tqdm

class BiasEvaluator:
    def __init__(self, model_path: str, device: str = 'cuda'):
        """Initialize the evaluator with a word embeddings model."""
        print(f"Loading word embeddings from {model_path}...")
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.device = device
        self.missing_words = set()

    def get_embedding(self, word: str) -> np.ndarray:
        """Retrieve the embedding of a word, return None if word is missing."""
        try:
            return self.model[word]
        except KeyError:
            self.missing_words.add(word)
            return None

    def compute_weat_effect_size(self, 
                                 target_set_1: List[str],
                                 target_set_2: List[str],
                                 attribute_set_1: List[str],
                                 attribute_set_2: List[str]) -> Dict:
        """
        Compute the WEAT effect size between two target sets and two attribute sets.
        Returns the effect size along with additional statistics.
        """
        # Convert word sets to embeddings, excluding missing words
        def get_embeddings_batch(word_list: List[str]) -> torch.Tensor:
            embeddings = [self.get_embedding(word) for word in word_list]
            valid_embeddings = [embedding for embedding in embeddings if embedding is not None]
            return torch.tensor(valid_embeddings, device=self.device)

        # Get embeddings for all word sets
        t1_embeddings = get_embeddings_batch(target_set_1)
        t2_embeddings = get_embeddings_batch(target_set_2)
        a1_embeddings = get_embeddings_batch(attribute_set_1)
        a2_embeddings = get_embeddings_batch(attribute_set_2)

        # Calculate mean cosine similarity between embeddings
        def mean_cosine_similarity(embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
            normalized1 = torch.nn.functional.normalize(embeddings1, dim=1)
            normalized2 = torch.nn.functional.normalize(embeddings2, dim=1)
            similarities = torch.mm(normalized1, normalized2.t())
            return similarities.mean(dim=1)

        # Calculate associations for each target set
        t1_a1_similarity = mean_cosine_similarity(t1_embeddings, a1_embeddings)
        t1_a2_similarity = mean_cosine_similarity(t1_embeddings, a2_embeddings)
        t2_a1_similarity = mean_cosine_similarity(t2_embeddings, a1_embeddings)
        t2_a2_similarity = mean_cosine_similarity(t2_embeddings, a2_embeddings)

        # Compute the differential association for each set
        t1_diff = t1_a1_similarity - t1_a2_similarity
        t2_diff = t2_a1_similarity - t2_a2_similarity

        # Compute the effect size
        total_size = len(t1_diff) + len(t2_diff)
        effect_size = (t1_diff.mean() - t2_diff.mean()) / torch.cat([t1_diff, t2_diff]).std()

        return {
            'effect_size': effect_size.item(),
            'missing_words': list(self.missing_words),
            't1_mean_diff': t1_diff.mean().item(),
            't2_mean_diff': t2_diff.mean().item(),
            'sample_size': total_size
        }

def main():
    parser = argparse.ArgumentParser(description='Evaluate bias in static word embeddings')
    parser.add_argument('--model_path', required=True, help='Path to word2vec format embeddings file')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    # Example word sets (modifiable or can be loaded from external files)
    target_set_1 = ['doctor', 'engineer', 'scientist', 'programmer']
    target_set_2 = ['nurse', 'teacher', 'librarian', 'homemaker']
    attribute_set_1 = ['he', 'man', 'his', 'male']
    attribute_set_2 = ['she', 'woman', 'her', 'female']

    evaluator = BiasEvaluator(args.model_path, args.device)
    results = evaluator.compute_weat_effect_size(
        target_set_1, target_set_2, attribute_set_1, attribute_set_2
    )

    print("\nResults:")
    print(f"Effect Size: {results['effect_size']:.3f}")
    print(f"Sample Size: {results['sample_size']}")
    print(f"Target Set 1 Mean Differential Association: {results['t1_mean_diff']:.3f}")
    print(f"Target Set 2 Mean Differential Association: {results['t2_mean_diff']:.3f}")
    
    if results['missing_words']:
        print("\nWarning: The following words were not found in the vocabulary:")
        print(", ".join(results['missing_words']))

if __name__ == '__main__':
    main()
