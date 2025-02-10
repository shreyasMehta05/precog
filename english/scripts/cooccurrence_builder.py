import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from scipy.sparse import save_npz, csr_matrix
import pickle
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
from colorama import Fore, Style, init
import json
import sys

# Initialize colorama
init()

class CooccurrenceMatrixBuilder:
    """Builds co-occurrence matrices with GPU acceleration and different window sizes"""
    
    def __init__(self, log_file: str = "cooccurrence_builder.log"):
        self.setup_logging(log_file)
        self._check_gpu()
        
    def _check_gpu(self) -> None:
        """Check GPU availability and memory"""
        try:
            mem_info = cp.cuda.runtime.memGetInfo()
            free_memory = mem_info[0] / 1024**3  # Convert to GB
            total_memory = mem_info[1] / 1024**3
            self.logger.info(f"{Fore.GREEN}GPU Memory: {free_memory:.2f}GB free / {total_memory:.2f}GB total{Style.RESET_ALL}")
        except Exception as e:
            self.logger.error(f"{Fore.RED}GPU check failed: {str(e)}{Style.RESET_ALL}")
            raise

    def setup_logging(self, log_file: str) -> None:
        """Setup logging configuration"""
        self.logger = logging.getLogger('CooccurrenceBuilder')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def build_vocabulary(self, tokenized_sentences: List[List[str]], 
                        min_freq: int = 5) -> Dict[str, int]:
        """Build vocabulary from tokenized sentences with minimum frequency threshold"""
        try:
            word_freq = {}
            
            # Count word frequencies
            self.logger.info(f"{Fore.CYAN}Counting word frequencies...{Style.RESET_ALL}")
            for sentence in tqdm(tokenized_sentences, desc="Building vocabulary"):
                for token in sentence:
                    word_freq[token] = word_freq.get(token, 0) + 1
            
            # Filter by minimum frequency and sort by frequency
            sorted_words = sorted(
                [(word, freq) for word, freq in word_freq.items() if freq >= min_freq],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Create vocabulary with indices
            vocab = {word: idx for idx, (word, _) in enumerate(sorted_words)}
            
            self.logger.info(f"{Fore.GREEN}Vocabulary size: {len(vocab)}{Style.RESET_ALL}")
            return vocab
            
        except Exception as e:
            self.logger.error(f"{Fore.RED}Vocabulary building failed: {str(e)}{Style.RESET_ALL}")
            raise

    def build_cooccurrence_matrix(self, 
                                tokenized_sentences: List[List[str]],
                                vocab: Dict[str, int],
                                window_size: int,
                                normalize: bool = False) -> cp_csr_matrix:
        """Build co-occurrence matrix using GPU acceleration"""
        try:
            vocab_size = len(vocab)
            # Use lists for initial collection to save memory
            row_indices = []
            col_indices = []
            values = []
            
            self.logger.info(f"{Fore.CYAN}Building co-occurrence matrix with window size {window_size}...{Style.RESET_ALL}")
            
            # Process each sentence
            for sentence in tqdm(tokenized_sentences, desc="Processing sentences"):
                # Convert tokens to indices, filtering unknown words
                indices = [vocab[token] for token in sentence if token in vocab]
                
                if len(indices) < 2:  # Skip sentences with fewer than 2 valid tokens
                    continue
                    
                # Slide window over sentence
                for i, center in enumerate(indices):
                    start_idx = max(0, i - window_size)
                    end_idx = min(len(indices), i + window_size + 1)
                    
                    for j in range(start_idx, end_idx):
                        if i != j:  # Skip self-co-occurrences
                            context = indices[j]
                            # Verify indices are within bounds
                            if center < vocab_size and context < vocab_size:
                                row_indices.append(center)
                                col_indices.append(context)
                                # Apply distance weighting if desired
                                weight = 1.0 / abs(i - j) if normalize else 1.0
                                values.append(weight)
            
            # Verify we have data to process
            if not row_indices:
                raise ValueError("No valid co-occurrences found in the corpus")
                
            # Convert to GPU arrays
            rows = cp.array(row_indices, dtype=cp.int32)
            cols = cp.array(col_indices, dtype=cp.int32)
            vals = cp.array(values, dtype=cp.float32)
            
            # Verify dimensions
            max_row = int(cp.max(rows))
            max_col = int(cp.max(cols))
            if max_row >= vocab_size or max_col >= vocab_size:
                raise ValueError(f"Index out of bounds. Max indices: {max_row}, {max_col}. Vocab size: {vocab_size}")
            
            # Create sparse matrix on GPU
            matrix = cp_csr_matrix((vals, (rows, cols)), 
                                 shape=(vocab_size, vocab_size),
                                 dtype=cp.float32)
            
            if normalize:
                # Normalize by row sums
                row_sums = cp.array(matrix.sum(axis=1)).flatten()
                row_sums[row_sums == 0] = 1  # Avoid division by zero
                matrix = matrix.multiply(1 / row_sums.reshape(-1, 1))
            
            self.logger.info(f"{Fore.GREEN}Matrix shape: {matrix.shape}, Non-zero elements: {matrix.nnz}{Style.RESET_ALL}")
            return matrix
            
        except Exception as e:
            self.logger.error(f"{Fore.RED}Co-occurrence matrix building failed: {str(e)}{Style.RESET_ALL}")
            raise

    def process_multiple_windows(self,
                               input_file: str,
                               output_dir: str,
                               window_sizes: List[int],
                               min_freq: int = 5,
                               normalize: bool = False) -> None:
        """Process corpus with multiple window sizes and save results"""
        try:
            start_time = time.perf_counter()
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Load tokenized sentences
            self.logger.info(f"{Fore.CYAN}Loading tokenized corpus from {input_file}{Style.RESET_ALL}")
            with open(input_file, 'rb') as f:
                tokenized_sentences = pickle.load(f)
            
            # Build vocabulary
            vocab = self.build_vocabulary(tokenized_sentences, min_freq)
            
            # Save vocabulary
            vocab_file = Path(output_dir) / "vocabulary.json"
            with open(vocab_file, 'w', encoding='utf-8') as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)
            
            # Process each window size
            for window_size in window_sizes:
                self.logger.info(f"{Fore.CYAN}Processing window size {window_size}{Style.RESET_ALL}")
                
                # Build matrix
                matrix = self.build_cooccurrence_matrix(
                    tokenized_sentences, vocab, window_size, normalize)
                
                # Convert to CPU and save
                matrix_cpu = csr_matrix(matrix.get())
                matrix_file = Path(output_dir) / f"cooc_matrix_w{window_size}.npz"
                save_npz(str(matrix_file), matrix_cpu)
                
                # Save matrix info
                matrix_info = {
                    "window_size": window_size,
                    "shape": matrix.shape,
                    "nonzero": int(matrix.nnz),
                    "normalized": normalize,
                    "vocabulary_size": len(vocab)
                }
                info_file = Path(output_dir) / f"matrix_info_w{window_size}.json"
                with open(info_file, 'w') as f:
                    json.dump(matrix_info, f, indent=2)
            
            processing_time = time.perf_counter() - start_time
            self.logger.info(f"{Fore.GREEN}All matrices processed in {processing_time:.2f} seconds{Style.RESET_ALL}")
            
        except Exception as e:
            self.logger.error(f"{Fore.RED}Matrix processing failed: {str(e)}{Style.RESET_ALL}")
            raise

if __name__ == "__main__":
    try:
        # Example usage
        builder = CooccurrenceMatrixBuilder(log_file="./logs/cooccurrence_builder.log")
        builder.process_multiple_windows(
            input_file="./processed_data/tokenized_corpus.pkl",
            output_dir="./processed_data/cooccurrence_matrices",
            window_sizes=[2, 4, 6, 8, 10],
            min_freq=5,
            normalize=True
        )
        print(f"{Fore.GREEN}Successfully built co-occurrence matrices{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.RED}Failed to build matrices: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)