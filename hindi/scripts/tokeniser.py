import re
import time
import sys
from typing import List, Optional, Dict, Any
from pathlib import Path
import pickle
import logging
from datetime import datetime

import stanza
import cudf
import torch
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored output
init(autoreset=True)

class EnhancedHindiGPUTokenizer:
    """Enhanced GPU-accelerated Hindi tokenizer with comprehensive logging and progress tracking."""
    
    def __init__(self, 
                 log_file: Optional[str] = None,
                 remove_foreign: bool = True,
                 remove_punctuation: bool = True,
                 remove_numbers: bool = True,
                 min_token_length: int = 2,
                 debug_mode: bool = False):
        
        self.remove_foreign = remove_foreign
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.min_token_length = min_token_length
        self.debug_mode = debug_mode
        
        # Initialize logging
        if log_file is None:
            log_file = f"hindi_tokenizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self._setup_logging(log_file)
        
        # Initialize components
        self._setup_device()
        self._setup_stanza()
        self._compile_regex_patterns()
        
        self.stats: Dict[str, Any] = {
            "total_processed": 0,
            "failed_tokens": 0,
            "processing_time": 0,
            "gpu_operations": 0,
            "preserved_tokens": set()
        }
    
    def _setup_logging(self, log_file: str) -> None:
        """Setup enhanced logging with colored output."""
        self.logger = logging.getLogger('EnhancedHindiTokenizer')
        self.logger.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / log_file
        
        # File handler with detailed formatting
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        fh.setFormatter(file_formatter)
        
        # Console handler with color formatting
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)
        
        class ColoredFormatter(logging.Formatter):
            """Custom formatter for colored console output."""
            
            COLORS = {
                'DEBUG': Fore.CYAN,
                'INFO': Fore.GREEN,
                'WARNING': Fore.YELLOW,
                'ERROR': Fore.RED,
                'CRITICAL': Fore.RED + Style.BRIGHT
            }
            
            def format(self, record):
                color = self.COLORS.get(record.levelname, '')
                record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
                return super().format(record)
        
        console_formatter = ColoredFormatter('%(message)s')
        ch.setFormatter(console_formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.logger.info(f"Logging initialized - writing to {log_path}")
    
    def _setup_device(self) -> None:
        """Setup and log GPU/CPU device information."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_info = f"Using device: {self.device}"
        if self.device.type == 'cuda':
            device_info += f" ({torch.cuda.get_device_name(0)})"
            device_info += f"\nAvailable GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        self.logger.info(device_info)
    
    def _compile_regex_patterns(self) -> None:
        """Compile and log regex patterns."""
        try:
            self.logger.debug("Compiling regex patterns...")
            
            self.patterns = {
                'leading_special': re.compile(r'^[•\'\"\$\-\.\s]+'),
                'number_with_units': re.compile(r'[\d\u0966-\u096F]+\s*(?:करोड़|लाख|रुपए|रूपए|साल|वर्ष|मिनट|हज़ार|घंटे)'),
                'parenthetical': re.compile(r'\(.*?\)'),
                'mixed_phrase': re.compile(r'[a-zA-Z]+[^\u0900-\u097F]*[\u0900-\u097F]+|[\u0900-\u097F]+[^\u0900-\u097F]*[a-zA-Z]+'),
                'sentence_end': re.compile(r'([।!?])\s*')
            }
            
            self.logger.debug("Successfully compiled all regex patterns")
            
        except Exception as e:
            self.logger.error(f"Failed to compile regex patterns: {str(e)}")
            raise
    
    def _setup_stanza(self) -> None:
        """Setup Stanza with error handling and logging."""
        try:
            self.logger.info("Initializing Stanza for Hindi...")
            
            # Download Hindi models if needed
            stanza.download('hi', verbose=False)
            
            # Initialize pipeline with detailed logging
            start_time = time.perf_counter()
            self.nlp = stanza.Pipeline(
                'hi',
                processors='tokenize',
                tokenize_no_ssplit=False,
                verbose=False,
                use_gpu=self.device.type == 'cuda'
            )
            setup_time = time.perf_counter() - start_time
            
            self.logger.info(f"Stanza initialization completed in {setup_time:.2f} seconds")
            
            # Initialize stop words
            self.stop_words = set(['का', 'की', 'के', 'में', 'से', 'को', 'पर', 'ने', 'एक', 'और'])
            self.logger.debug(f"Initialized {len(self.stop_words)} stop words")
            
        except Exception as e:
            self.logger.critical(f"Failed to initialize Stanza: {str(e)}")
            raise
    
    def clean_text_gpu(self, text_series) -> any:
        """Enhanced GPU-accelerated text cleaning with detailed logging."""
        start_time = time.perf_counter()
        self.logger.info("Starting GPU text cleaning...")
        
        try:
            # Convert to pandas for complex regex operations
            texts = text_series.to_pandas()
            total_texts = len(texts)
            
            cleaned_texts = []
            preserved_tokens = set()
            
            for idx, text in enumerate(tqdm(texts, desc=f"{Fore.CYAN}Cleaning texts{Style.RESET_ALL}")):
                if self.debug_mode and idx < 5:
                    self.logger.debug(f"Original text: {text}")
                
                # Clean text with pattern matching
                for pattern_name, pattern in self.patterns.items():
                    if pattern_name == 'number_with_units':
                        matches = pattern.findall(text)
                        preserved_tokens.update(matches)
                        for match in matches:
                            text = text.replace(match, f"__{match}__")
                    elif pattern_name == 'mixed_phrase':
                        matches = pattern.findall(text)
                        preserved_tokens.update(matches)
                    else:
                        text = pattern.sub('', text)
                
                if self.debug_mode and idx < 5:
                    self.logger.debug(f"Cleaned text: {text}")
                
                cleaned_texts.append(text)
            
            # Update statistics
            self.stats["preserved_tokens"].update(preserved_tokens)
            self.stats["gpu_operations"] += 1
            
            processing_time = time.perf_counter() - start_time
            self.stats["processing_time"] += processing_time
            
            self.logger.info(f"Text cleaning completed in {processing_time:.2f} seconds")
            self.logger.info(f"Preserved {len(preserved_tokens)} unique tokens")
            
            return cudf.Series(cleaned_texts)
            
        except Exception as e:
            self.logger.error(f"Text cleaning failed: {str(e)}")
            self.log_error_context(e)
            raise
    
    def tokenize_batch(self, sentences: List[str], batch_size: int = 1000) -> List[List[str]]:
        """Enhanced batch tokenization with progress tracking and error handling."""
        start_time = time.perf_counter()
        self.logger.info(f"Starting batch tokenization with size {batch_size}")
        
        try:
            tokenized_sentences = []
            total_tokens = 0
            
            for i in tqdm(range(0, len(sentences), batch_size),
                         desc=f"{Fore.CYAN}Tokenizing batches{Style.RESET_ALL}"):
                batch = sentences[i:i + batch_size]
                batch_start_time = time.perf_counter()
                
                for sentence in batch:
                    try:
                        # Tokenize with Stanza
                        doc = self.nlp(sentence)
                        
                        # Extract and filter tokens
                        tokens = []
                        for sent in doc.sentences:
                            sent_tokens = [word.text for word in sent.words]
                            filtered = self.filter_tokens(sent_tokens)
                            tokens.extend(filtered)
                            total_tokens += len(filtered)
                        
                        tokenized_sentences.append(tokens)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to tokenize sentence: {sentence[:100]}...")
                        self.stats["failed_tokens"] += 1
                        continue
                
                batch_time = time.perf_counter() - batch_start_time
                if self.debug_mode:
                    self.logger.debug(f"Batch processed in {batch_time:.2f} seconds")
            
            processing_time = time.perf_counter() - start_time
            self.stats["processing_time"] += processing_time
            self.stats["total_processed"] += len(sentences)
            
            self.logger.info(f"Tokenization completed in {processing_time:.2f} seconds")
            self.logger.info(f"Processed {len(sentences)} sentences, {total_tokens} tokens")
            
            return tokenized_sentences
            
        except Exception as e:
            self.logger.error("Batch tokenization failed")
            self.log_error_context(e)
            raise
    
    def log_error_context(self, error: Exception) -> None:
        """Log detailed error context for debugging."""
        self.logger.error("Error Context:")
        self.logger.error(f"Error Type: {type(error).__name__}")
        self.logger.error(f"Error Message: {str(error)}")
        self.logger.error(f"Current Stats: {self.stats}")
        
        # Log system info
        self.logger.error(f"Python Version: {sys.version}")
        self.logger.error(f"GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.error(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated")
    
    def save_stats(self, output_dir: Path) -> None:
        """Save processing statistics to file."""
        stats_file = output_dir / "hindi_corpus_stats.pkl"
        with open(stats_file, 'wb') as f:
            pickle.dump(self.stats, f)
        self.logger.info(f"Statistics saved to {stats_file}")
    
    def process_corpus(self, 
                      input_file: str, 
                      output_dir: str,
                      batch_size: int = 1000) -> Dict[str, Any]:
        """Process entire corpus with comprehensive logging and error handling."""
        start_time = time.perf_counter()
        self.logger.info(f"Starting corpus processing: {input_file}")
        
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Load corpus
            self.logger.info("Loading corpus...")
            df = cudf.read_csv(input_file, header=None, names=["sentence"])
            self.logger.info(f"Loaded {len(df)} sentences")
            
            # Process text
            self.logger.info("Cleaning text...")
            df["clean"] = self.clean_text_gpu(df["sentence"])
            
            # Tokenize
            sentences = df["clean"].to_pandas().tolist()
            tokenized_sentences = self.tokenize_batch(sentences, batch_size)
            
            # Calculate statistics
            total_sentences = len(tokenized_sentences)
            total_tokens = sum(len(sent) for sent in tokenized_sentences)
            avg_tokens = total_tokens / total_sentences if total_sentences > 0 else 0
            
            # Update and save statistics
            self.stats.update({
                "total_sentences": total_sentences,
                "total_tokens": total_tokens,
                "avg_tokens_per_sentence": avg_tokens,
                "total_processing_time": time.perf_counter() - start_time
            })
            
            # Save results
            output_file = output_path / "hindi_tokenized_corpus.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(tokenized_sentences, f)
            
            self.save_stats(output_path)
            
            self.logger.info(f"Processing completed in {self.stats['total_processing_time']:.2f} seconds")
            self.logger.info(f"Results saved to {output_file}")
            
            return {
                "tokenized_sentences": tokenized_sentences,
                "stats": self.stats
            }
            
        except Exception as e:
            self.logger.error("Corpus processing failed")
            self.log_error_context(e)
            raise
        
    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """Filter tokens based on length and stop words."""
        filtered_tokens = []
        for token in tokens:
            # Remove tokens that are shorter than the minimum token length
            if len(token) < self.min_token_length:
                continue
            # Remove stop words if needed
            if token in self.stop_words:
                continue
            # (Optionally add further filtering: punctuation, foreign words, etc.)
            filtered_tokens.append(token)
        return filtered_tokens


if __name__ == "__main__":
    try:
        # Parse command line arguments (could be enhanced with argparse)
        input_file = sys.argv[1] if len(sys.argv) > 1 else "./dataset/raw/hindi_300k/hin_news_2022_300K-sentences.txt"
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./processed_data"
        debug_mode = "--debug" in sys.argv
        
        # Initialize tokenizer
        tokenizer = EnhancedHindiGPUTokenizer(
            debug_mode=debug_mode,
            remove_foreign=True,
            remove_punctuation=True,
            remove_numbers=True,
            min_token_length=2
        )
        
        # Process corpus
        results = tokenizer.process_corpus(
            input_file=input_file,
            output_dir=output_dir,
            batch_size=1000
        )
        
        # Print summary
        print(f"\n{Fore.GREEN}Processing Summary{Style.RESET_ALL}")
        print(f"Total Sentences: {results['stats']['total_sentences']}")
        print(f"Total Tokens: {results['stats']['total_tokens']}")
        print(f"Avg Tokens/Sentence: {results['stats']['avg_tokens_per_sentence']:.2f}")
        print(f"Total Processing Time: {results['stats']['total_processing_time']:.2f} seconds")
        
    except Exception as e:
        
        print(f"{Fore.RED}Processing Failed{Style.RESET_ALL}")
        print(f"Error: {str(e)}")
        sys.exit(1)