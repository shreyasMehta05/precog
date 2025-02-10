import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg import svds
import json
import time
from pathlib import Path
import logging
from tqdm import tqdm
from colorama import Fore, Style, init
import sys
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Initialize colorama
init()

class MatrixReducer:
    """Handles matrix normalization and dimensionality reduction with detailed logging"""
    
    def __init__(self, log_dir: str = "logs"):
        self.setup_logging(log_dir)
        self._check_gpu()
        
    def _check_gpu(self) -> None:
        """Check GPU availability and memory"""
        try:
            mem_info = cp.cuda.runtime.memGetInfo()
            free_memory = mem_info[0] / 1024**3
            total_memory = mem_info[1] / 1024**3
            self.logger.info(f"{Fore.GREEN}GPU Memory: {free_memory:.2f}GB free / {total_memory:.2f}GB total{Style.RESET_ALL}")
        except Exception as e:
            self.logger.error(f"{Fore.RED}GPU check failed: {str(e)}{Style.RESET_ALL}")
            raise

    def setup_logging(self, log_dir: str) -> None:
        """Setup logging with both file and console handlers"""
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"matrix_reduction_{timestamp}.log"
        
        self.logger = logging.getLogger('MatrixReducer')
        self.logger.setLevel(logging.INFO)
        
        # File handler with detailed formatting
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(file_formatter)
        
        # Console handler with color
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(console_formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    # ... (previous imports remain the same)

    def apply_normalization(self, 
                          matrix: cp_csr_matrix,
                          method: str) -> cp_csr_matrix:
        """Apply different normalization techniques to the matrix"""
        try:
            self.logger.info(f"{Fore.CYAN}Applying {method} normalization...{Style.RESET_ALL}")
            
            if method == "ppmi":
                # Positive Pointwise Mutual Information
                row_sums = matrix.sum(axis=1)
                col_sums = matrix.sum(axis=0)
                total_sum = float(matrix.sum())
                
                # Convert to probability space
                prob_matrix = matrix.multiply(1/total_sum)
                row_probs = row_sums / total_sum
                col_probs = col_sums / total_sum
                
                # Calculate PMI (log(P(x,y)/(P(x)P(y))))
                expected_probs = cp.outer(row_probs, col_probs)
                pmi_matrix = prob_matrix.multiply(1/expected_probs)
                pmi_matrix.data = cp.log(pmi_matrix.data)
                
                # Convert negative values to 0 for PPMI
                pmi_matrix.data = cp.maximum(pmi_matrix.data, 0)
                
                return pmi_matrix
                
            elif method == "tfidf":
                # Term Frequency-Inverse Document Frequency variant
                N = matrix.shape[0]
                
                # Convert matrix to binary (occurrence matrix)
                binary_matrix = matrix.copy()
                binary_matrix.data = cp.ones_like(binary_matrix.data)
                
                # Calculate document frequency (number of rows where term appears)
                doc_freq = cp.array(binary_matrix.sum(axis=0)).flatten()
                
                # Calculate IDF
                idf = cp.log(N / (doc_freq + 1))
                
                # Apply IDF weights to original matrix
                normalized = matrix.multiply(idf)
                
                return normalized
                
            elif method == "row_normalize":
                # Simple row normalization
                row_sums = matrix.sum(axis=1)
                # If the result has the .A attribute (e.g., if it's a sparse matrix), use it;
                # otherwise, work directly with the ndarray.
                if hasattr(row_sums, "A"):
                    row_sums = row_sums.A.ravel()
                else:
                    row_sums = row_sums.ravel()
                    
                # Avoid division by zero by setting zeros to 1
                row_sums[row_sums == 0] = 1

                # If matrix is sparse (has a multiply method), use it; otherwise, use standard division.
                if hasattr(matrix, "multiply"):
                    normalized = matrix.multiply(1 / row_sums.reshape(-1, 1))
                else:
                    normalized = matrix / row_sums.reshape(-1, 1)
                
                return normalized

                
            else:
                raise ValueError(f"Unknown normalization method: {method}")
                
        except Exception as e:
            self.logger.error(f"{Fore.RED}Normalization failed: {str(e)}{Style.RESET_ALL}")
            raise


    def reduce_dimensionality(self,
                            matrix: cp_csr_matrix,
                            d: int) -> np.ndarray:
        """Perform truncated SVD for dimensionality reduction"""
        try:
            self.logger.info(f"{Fore.CYAN}Performing SVD with d={d}...{Style.RESET_ALL}")
            
            # Convert to CPU for SVD (cupy's SVD implementation might be unstable for large sparse matrices)
            matrix_cpu = matrix.get()
            
            # Perform truncated SVD
            U, Sigma, Vt = svds(matrix_cpu, k=d)
            
            # Create word embeddings using U * sqrt(Sigma)
            embeddings = U * np.sqrt(Sigma.reshape(1, -1))
            
            self.logger.info(f"{Fore.GREEN}SVD complete. Embedding shape: {embeddings.shape}{Style.RESET_ALL}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"{Fore.RED}SVD failed: {str(e)}{Style.RESET_ALL}")
            raise

    def process_matrices(self,
                        input_dir: str,
                        output_dir: str,
                        d_values: List[int],
                        normalization_methods: List[str]) -> None:
        """Process matrices with different normalizations and d-values"""
        try:
            start_time = time.perf_counter()
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Get all matrix files
            matrix_files = list(Path(input_dir).glob("cooc_matrix_w*.npz"))
            
            results = {}
            for matrix_file in matrix_files:
                window_size = int(matrix_file.stem.split('w')[1])
                self.logger.info(f"\n{Fore.CYAN}Processing matrix with window size {window_size}{Style.RESET_ALL}")
                
                # Load matrix
                matrix = cp_csr_matrix(load_npz(matrix_file))
                original_shape = matrix.shape
                
                for norm_method in normalization_methods:
                    self.logger.info(f"\n{Fore.CYAN}Applying {norm_method} normalization{Style.RESET_ALL}")
                    
                    # Apply normalization
                    normalized_matrix = self.apply_normalization(matrix, norm_method)
                    
                    for d in d_values:
                        self.logger.info(f"\n{Fore.CYAN}Reducing to d={d} dimensions{Style.RESET_ALL}")
                        
                        # Perform SVD
                        embeddings = self.reduce_dimensionality(normalized_matrix, d)
                        
                        # Save embeddings
                        if(norm_method == "row_normalize"):
                            output_file = Path(output_dir) / f"embeddings_w{window_size}_rownormalize_d{d}.npy"
                        else:
                            output_file = Path(output_dir) / f"embeddings_w{window_size}_{norm_method}_d{d}.npy"
                        np.save(output_file, embeddings)
                        
                        # Store results
                        result_key = f"w{window_size}_{norm_method}_d{d}"
                        results[result_key] = {
                            "window_size": window_size,
                            "normalization": norm_method,
                            "dimensions": d,
                            "original_shape": original_shape,
                            "embedding_shape": embeddings.shape,
                            "file_path": str(output_file)
                        }
            
            # Save results summary
            results_file = Path(output_dir) / "reduction_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            processing_time = time.perf_counter() - start_time
            self.logger.info(f"\n{Fore.GREEN}All processing complete in {processing_time:.2f} seconds{Style.RESET_ALL}")
            
        except Exception as e:
            self.logger.error(f"{Fore.RED}Processing failed: {str(e)}{Style.RESET_ALL}")
            raise

if __name__ == "__main__":
    try:
        reducer = MatrixReducer("logs")
        reducer.process_matrices(
            input_dir="./processed_data/cooccurrence_matrices",
            output_dir="./processed_data/embeddings",
            d_values=[50, 100, 200, 300],
            normalization_methods=["ppmi", "tfidf", "row_normalize"]
        )
        print(f"{Fore.GREEN}Successfully processed all matrices{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.RED}Failed to process matrices: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)