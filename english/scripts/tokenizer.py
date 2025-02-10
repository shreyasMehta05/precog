import time
import sys
from pathlib import Path
import pickle
from typing import List, Dict, Optional, Set

import cudf
import spacy
from tqdm import tqdm
from colorama import Fore, Style, init
import logging
import re

# Initialize colorama for cross-platform colored output
init()

class GPUTokenizer:
    """GPU-accelerated tokenizer with progress tracking and error handling"""
    
    def __init__(self, 
                 model_name: str = "en_core_web_sm", 
                 log_file: str = "tokenizer.log",
                 remove_stop_words: bool = True,
                 remove_punctuation: bool = True,
                 remove_numbers: bool = True,
                 min_token_length: int = 3):
        self.setup_logging(log_file)
        self.remove_stop_words = remove_stop_words
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.min_token_length = min_token_length
        self._setup_spacy(model_name)
        
    def _setup_spacy(self, model_name: str) -> None:
        """Setup spaCy with GPU if available"""
        try:
            if spacy.prefer_gpu():
                spacy.require_gpu()
                self.logger.info(f"{Fore.GREEN}GPU acceleration enabled for spaCy{Style.RESET_ALL}")
            else:
                self.logger.warning(f"{Fore.YELLOW}GPU not available for spaCy, falling back to CPU{Style.RESET_ALL}")
            
            self.nlp = spacy.load(model_name)
            # Create custom stop words set
            self.stop_words = set(self.nlp.Defaults.stop_words)
            self.logger.info(f"Loaded spaCy model: {model_name}")
        except Exception as e:
            self.logger.error(f"{Fore.RED}Failed to initialize spaCy: {str(e)}{Style.RESET_ALL}")
            raise
            
    def setup_logging(self, log_file: str) -> None:
        """Setup logging configuration"""
        self.logger = logging.getLogger('GPUTokenizer')
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

    def clean_text_gpu(self, text_series: cudf.Series) -> cudf.Series:
        """Clean text using GPU-accelerated operations"""
        try:
            clean_series = text_series.str.lower()
            
            if self.remove_punctuation:
                clean_series = clean_series.str.replace(r'[^\w\s]', '', regex=True)
                
            if self.remove_numbers:
                clean_series = clean_series.str.replace(r'\d+', '', regex=True)
                
            # Remove extra whitespace
            clean_series = clean_series.str.replace(r'\s+', ' ', regex=True)
            clean_series = clean_series.str.strip()
            
            return clean_series
        except Exception as e:
            self.logger.error(f"{Fore.RED}Text cleaning failed: {str(e)}{Style.RESET_ALL}")
            raise

    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """Filter tokens based on configured criteria"""
        filtered_tokens = []
        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_token_length:
                continue
                
            # Skip stop words if enabled
            if self.remove_stop_words and token in self.stop_words:
                continue
                
            # Skip tokens that are just numbers if enabled
            if self.remove_numbers and token.isdigit():
                continue
                
            filtered_tokens.append(token)
        return filtered_tokens

    def tokenize_batch(self, sentences: List[str], batch_size: int = 1000) -> List[List[str]]:
        """Tokenize sentences using spaCy's pipe for batch processing"""
        try:
            docs = list(tqdm(
                self.nlp.pipe(sentences, batch_size=batch_size),
                total=len(sentences),
                desc=f"{Fore.CYAN}Tokenizing batches{Style.RESET_ALL}"
            ))
            
            # Apply token filtering
            tokenized_sentences = []
            for doc in docs:
                tokens = [token.text for token in doc]
                filtered_tokens = self.filter_tokens(tokens)
                tokenized_sentences.append(filtered_tokens)
                
            return tokenized_sentences
        except Exception as e:
            self.logger.error(f"{Fore.RED}Tokenization failed: {str(e)}{Style.RESET_ALL}")
            raise

    def process_corpus(self, 
                      input_file: str, 
                      output_dir: str,
                      batch_size: int = 1000) -> Dict[str, List[List[str]]]:
        """Process entire corpus and save tokenized output"""
        start_time = time.perf_counter()
        self.logger.info(f"{Fore.GREEN}Starting corpus processing{Style.RESET_ALL}")
        
        try:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Load corpus
            self.logger.info(f"{Fore.CYAN}Loading corpus from {input_file}{Style.RESET_ALL}")
            df = cudf.read_csv(input_file, header=None, names=["sentence"])
            
            # Clean text
            self.logger.info(f"{Fore.CYAN}Cleaning text on GPU{Style.RESET_ALL}")
            df["clean"] = self.clean_text_gpu(df["sentence"])
            
            # Convert to pandas for tokenization
            sentences = df["clean"].to_pandas().tolist()
            
            # Tokenize
            tokenized_sentences = self.tokenize_batch(sentences, batch_size)
            
            # Generate statistics
            total_sentences = len(tokenized_sentences)
            total_tokens = sum(len(sent) for sent in tokenized_sentences)
            avg_tokens_per_sentence = total_tokens / total_sentences if total_sentences > 0 else 0
            
            stats = {
                "total_sentences": total_sentences,
                "total_tokens": total_tokens,
                "avg_tokens_per_sentence": avg_tokens_per_sentence
            }
            
            # Save results
            output_file = Path(output_dir) / "tokenized_corpus.pkl"
            stats_file = Path(output_dir) / "corpus_stats.pkl"
            
            with open(output_file, 'wb') as f:
                pickle.dump(tokenized_sentences, f)
            with open(stats_file, 'wb') as f:
                pickle.dump(stats, f)
            
            processing_time = time.perf_counter() - start_time
            self.logger.info(f"{Fore.GREEN}Processing complete in {processing_time:.2f} seconds{Style.RESET_ALL}")
            self.logger.info(f"{Fore.GREEN}Tokenized corpus saved to {output_file}{Style.RESET_ALL}")
            self.logger.info(f"Statistics: {stats}")
            
            return {
                "tokenized_sentences": tokenized_sentences,
                "stats": stats
            }
            
        except Exception as e:
            self.logger.error(f"{Fore.RED}Corpus processing failed: {str(e)}{Style.RESET_ALL}")
            raise

if __name__ == "__main__":
    # Example usage
    try:
        tokenizer = GPUTokenizer(
            remove_stop_words=True,
            remove_punctuation=True,
            remove_numbers=True,
            min_token_length=3
        )
        results = tokenizer.process_corpus(
            input_file="./dataset/raw/eng_300k/eng_news_2024_300k-sentences.txt",
            output_dir="./processed_data",
            batch_size=1000
        )
        print(f"{Fore.GREEN}Successfully processed corpus{Style.RESET_ALL}")
        print(f"Statistics: {results['stats']}")
        
    except Exception as e:
        print(f"{Fore.RED}Failed to process corpus: {str(e)}{Style.RESET_ALL}")
        sys.exit(1) 