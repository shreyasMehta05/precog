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

class HindiCooccurrenceMatrixBuilder:
    """Builds co-occurrence matrices for Hindi text with GPU acceleration and different window sizes"""
    
    def __init__(self, log_file: str = "hindi_cooccurrence_builder.log"):
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
        self.logger = logging.getLogger('HindiCooccurrenceBuilder')
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
                        min_freq: int = 3) -> Dict[str, int]:  # Lower min_freq for Hindi
        """Build vocabulary from tokenized Hindi sentences with minimum frequency threshold"""
        try:
            word_freq = {}
            
            # Count word frequencies
            self.logger.info(f"{Fore.CYAN}Counting Hindi word frequencies...{Style.RESET_ALL}")
            for sentence in tqdm(tokenized_sentences, desc="Building Hindi vocabulary"):
                for token in sentence:
                    # Verify token is in Devanagari script
                    if any('\u0900' <= char <= '\u097F' for char in token):
                        word_freq[token] = word_freq.get(token, 0) + 1
            
            # Filter by minimum frequency and sort by frequency
            sorted_words = sorted(
                [(word, freq) for word, freq in word_freq.items() if freq >= min_freq],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Create vocabulary with indices
            vocab = {word: idx for idx, (word, _) in enumerate(sorted_words)}
            
            # Save frequency information
            freq_info = {word: freq for word, freq in word_freq.items() if freq >= min_freq}
            
            self.logger.info(f"{Fore.GREEN}Hindi vocabulary size: {len(vocab)}{Style.RESET_ALL}")
            return vocab, freq_info
            
        except Exception as e:
            self.logger.error(f"{Fore.RED}Hindi vocabulary building failed: {str(e)}{Style.RESET_ALL}")
            raise

    def build_cooccurrence_matrix(self, 
                                tokenized_sentences: List[List[str]],
                                vocab: Dict[str, int],
                                window_size: int,
                                normalize: bool = False,
                                distance_weighting: bool = True) -> cp_csr_matrix:
        """Build co-occurrence matrix for Hindi text using GPU acceleration"""
        try:
            vocab_size = len(vocab)
            row_indices = []
            col_indices = []
            values = []
            
            self.logger.info(f"{Fore.CYAN}Building Hindi co-occurrence matrix with window size {window_size}...{Style.RESET_ALL}")
            
            # Process each sentence
            for sentence in tqdm(tokenized_sentences, desc="Processing Hindi sentences"):
                # Convert tokens to indices, filtering non-Hindi and unknown words
                indices = [vocab[token] for token in sentence 
                         if token in vocab and any('\u0900' <= char <= '\u097F' for char in token)]
                
                if len(indices) < 2:
                    continue
                    
                # Slide window over sentence
                for i, center in enumerate(indices):
                    start_idx = max(0, i - window_size)
                    end_idx = min(len(indices), i + window_size + 1)
                    
                    for j in range(start_idx, end_idx):
                        if i != j:
                            context = indices[j]
                            if center < vocab_size and context < vocab_size:
                                row_indices.append(center)
                                col_indices.append(context)
                                
                                # Apply distance weighting if enabled
                                if distance_weighting:
                                    weight = 1.0 / abs(i - j)
                                else:
                                    weight = 1.0
                                    
                                values.append(weight)
            
            if not row_indices:
                raise ValueError("No valid Hindi co-occurrences found in the corpus")
                
            # Convert to GPU arrays
            rows = cp.array(row_indices, dtype=cp.int32)
            cols = cp.array(col_indices, dtype=cp.int32)
            vals = cp.array(values, dtype=cp.float32)
            
            # Create sparse matrix on GPU
            matrix = cp_csr_matrix((vals, (rows, cols)), 
                                 shape=(vocab_size, vocab_size),
                                 dtype=cp.float32)
            
            if normalize:
                # Normalize by row sums
                row_sums = cp.array(matrix.sum(axis=1)).flatten()
                row_sums[row_sums == 0] = 1
                matrix = matrix.multiply(1 / row_sums.reshape(-1, 1))
            
            self.logger.info(f"{Fore.GREEN}Hindi matrix shape: {matrix.shape}, Non-zero elements: {matrix.nnz}{Style.RESET_ALL}")
            return matrix
            
        except Exception as e:
            self.logger.error(f"{Fore.RED}Hindi co-occurrence matrix building failed: {str(e)}{Style.RESET_ALL}")
            raise

    def process_multiple_windows(self,
                               input_file: str,
                               output_dir: str,
                               window_sizes: List[int],
                               min_freq: int = 3,
                               normalize: bool = False,
                               distance_weighting: bool = True) -> None:
        """Process Hindi corpus with multiple window sizes and save results"""
        try:
            start_time = time.perf_counter()
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Load tokenized sentences
            self.logger.info(f"{Fore.CYAN}Loading tokenized Hindi corpus from {input_file}{Style.RESET_ALL}")
            with open(input_file, 'rb') as f:
                tokenized_sentences = pickle.load(f)
            
            # Build vocabulary with frequency information
            vocab, freq_info = self.build_vocabulary(tokenized_sentences, min_freq)
            
            # Save vocabulary and frequency information
            vocab_file = Path(output_dir) / "hindi_vocabulary.json"
            freq_file = Path(output_dir) / "hindi_word_frequencies.json"
            
            with open(vocab_file, 'w', encoding='utf-8') as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)
            with open(freq_file, 'w', encoding='utf-8') as f:
                json.dump(freq_info, f, ensure_ascii=False, indent=2)
            
            # Process each window size
            for window_size in window_sizes:
                self.logger.info(f"{Fore.CYAN}Processing window size {window_size}{Style.RESET_ALL}")
                
                # Build matrix
                matrix = self.build_cooccurrence_matrix(
                    tokenized_sentences, vocab, window_size, normalize, distance_weighting)
                
                # Convert to CPU and save
                matrix_cpu = csr_matrix(matrix.get())
                matrix_file = Path(output_dir) / f"hindi_cooc_matrix_w{window_size}.npz"
                save_npz(str(matrix_file), matrix_cpu)
                
                # Save matrix info with Hindi-specific details
                matrix_info = {
                    "window_size": window_size,
                    "shape": matrix.shape,
                    "nonzero": int(matrix.nnz),
                    "normalized": normalize,
                    "distance_weighted": distance_weighting,
                    "vocabulary_size": len(vocab),
                    "min_frequency": min_freq,
                    "language": "hindi"
                }
                info_file = Path(output_dir) / f"hindi_matrix_info_w{window_size}.json"
                with open(info_file, 'w') as f:
                    json.dump(matrix_info, f, indent=2)
            
            processing_time = time.perf_counter() - start_time
            self.logger.info(f"{Fore.GREEN}All Hindi matrices processed in {processing_time:.2f} seconds{Style.RESET_ALL}")
            
        except Exception as e:
            self.logger.error(f"{Fore.RED}Hindi matrix processing failed: {str(e)}{Style.RESET_ALL}")
            raise

if __name__ == "__main__":
    try:
        # Example usage
        builder = HindiCooccurrenceMatrixBuilder(log_file="./logs/hindi_cooccurrence_builder.log")
        builder.process_multiple_windows(
            input_file="./processed_data/hindi_tokenized_corpus.pkl",
            output_dir="./processed_data/hindi_cooccurrence_matrices",
            window_sizes=[2, 4, 6, 8, 10],
            min_freq=3,  # Lower threshold for Hindi
            normalize=True,
            distance_weighting=True
        )
        print(f"{Fore.GREEN}Successfully built Hindi co-occurrence matrices{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.RED}Failed to build Hindi matrices: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)