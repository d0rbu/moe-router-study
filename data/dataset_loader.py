"""
Dataset loading utilities for different data types (pretraining, math, code, etc.).
"""

import torch
from datasets import load_dataset, Dataset
from typing import List, Dict, Optional, Tuple, Any
import random
import logging

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Load and preprocess datasets for MoE routing analysis.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize dataset loader.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        
    def load_fineweb_sample(
        self,
        num_samples: int = 1000,
        max_length: int = 512,
        split: str = "train"
    ) -> Tuple[List[str], List[str]]:
        """
        Load a sample from FineWeb pretraining data.
        
        Args:
            num_samples: Number of samples to load
            max_length: Maximum text length
            split: Dataset split to use
            
        Returns:
            Tuple of (texts, labels)
        """
        logger.info(f"Loading {num_samples} samples from FineWeb")
        
        try:
            # Load FineWeb dataset
            dataset = load_dataset(
                "HuggingFaceFW/fineweb", 
                name="sample-10BT",
                split=f"{split}[:{num_samples}]",
                streaming=False
            )
            
            texts = []
            labels = []
            
            for item in dataset:
                text = item["text"]
                
                # Truncate if too long
                if len(text) > max_length:
                    text = text[:max_length]
                    
                texts.append(text)
                labels.append("pretraining")
                
            logger.info(f"Loaded {len(texts)} FineWeb samples")
            return texts, labels
            
        except Exception as e:
            logger.error(f"Failed to load FineWeb: {e}")
            # Fallback to dummy data
            return self._generate_dummy_pretraining_data(num_samples), ["pretraining"] * num_samples
    
    def load_math_qa_data(
        self,
        num_samples: int = 500,
        dataset_name: str = "gsm8k"
    ) -> Tuple[List[str], List[str]]:
        """
        Load mathematical reasoning data.
        
        Args:
            num_samples: Number of samples to load
            dataset_name: Math dataset to use ("gsm8k", "math", "aqua_rat")
            
        Returns:
            Tuple of (texts, labels)
        """
        logger.info(f"Loading {num_samples} samples from {dataset_name}")
        
        try:
            if dataset_name == "gsm8k":
                dataset = load_dataset("gsm8k", "main", split=f"train[:{num_samples}]")
                texts = [item["question"] for item in dataset]
                
            elif dataset_name == "math":
                dataset = load_dataset("hendrycks/competition_math", split=f"train[:{num_samples}]")
                texts = [item["problem"] for item in dataset]
                
            elif dataset_name == "aqua_rat":
                dataset = load_dataset("aqua_rat", split=f"train[:{num_samples}]")
                texts = [item["question"] for item in dataset]
                
            else:
                raise ValueError(f"Unknown math dataset: {dataset_name}")
                
            labels = ["math"] * len(texts)
            logger.info(f"Loaded {len(texts)} math samples from {dataset_name}")
            return texts, labels
            
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            return self._generate_dummy_math_data(num_samples), ["math"] * num_samples
    
    def load_code_data(
        self,
        num_samples: int = 500,
        dataset_name: str = "codeparrot/github-code"
    ) -> Tuple[List[str], List[str]]:
        """
        Load code data for analysis.
        
        Args:
            num_samples: Number of samples to load
            dataset_name: Code dataset to use
            
        Returns:
            Tuple of (texts, labels)
        """
        logger.info(f"Loading {num_samples} code samples")
        
        try:
            if dataset_name == "codeparrot/github-code":
                # Load a subset of GitHub code
                dataset = load_dataset(
                    "codeparrot/github-code-clean",
                    split=f"train[:{num_samples}]",
                    streaming=False
                )
                texts = [item["code"] for item in dataset if len(item["code"]) > 50]
                
            elif dataset_name == "humaneval":
                dataset = load_dataset("openai_humaneval", split="test")
                texts = [item["prompt"] for item in dataset][:num_samples]
                
            else:
                # Fallback to dummy code data
                texts = self._generate_dummy_code_data(num_samples)
                
            labels = ["code"] * len(texts)
            logger.info(f"Loaded {len(texts)} code samples")
            return texts, labels
            
        except Exception as e:
            logger.error(f"Failed to load code data: {e}")
            return self._generate_dummy_code_data(num_samples), ["code"] * num_samples
    
    def load_mixed_dataset(
        self,
        pretraining_samples: int = 500,
        math_samples: int = 200,
        code_samples: int = 200,
        other_samples: int = 100
    ) -> Tuple[List[str], List[str]]:
        """
        Load a mixed dataset with different data types.
        
        Args:
            pretraining_samples: Number of pretraining samples
            math_samples: Number of math samples
            code_samples: Number of code samples
            other_samples: Number of other samples
            
        Returns:
            Tuple of (texts, labels)
        """
        all_texts = []
        all_labels = []
        
        # Load pretraining data
        if pretraining_samples > 0:
            texts, labels = self.load_fineweb_sample(pretraining_samples)
            all_texts.extend(texts)
            all_labels.extend(labels)
        
        # Load math data
        if math_samples > 0:
            texts, labels = self.load_math_qa_data(math_samples)
            all_texts.extend(texts)
            all_labels.extend(labels)
        
        # Load code data
        if code_samples > 0:
            texts, labels = self.load_code_data(code_samples)
            all_texts.extend(texts)
            all_labels.extend(labels)
        
        # Load other data types
        if other_samples > 0:
            texts, labels = self._load_other_data(other_samples)
            all_texts.extend(texts)
            all_labels.extend(labels)
        
        # Shuffle the combined dataset
        combined = list(zip(all_texts, all_labels))
        random.shuffle(combined)
        all_texts, all_labels = zip(*combined)
        
        logger.info(f"Created mixed dataset with {len(all_texts)} total samples")
        return list(all_texts), list(all_labels)
    
    def _load_other_data(self, num_samples: int) -> Tuple[List[str], List[str]]:
        """Load other types of data (news, books, etc.)."""
        try:
            # Load news data
            dataset = load_dataset("cnn_dailymail", "3.0.0", split=f"train[:{num_samples}]")
            texts = [item["article"][:500] for item in dataset]  # Truncate articles
            labels = ["news"] * len(texts)
            return texts, labels
        except:
            # Fallback to dummy data
            return self._generate_dummy_other_data(num_samples), ["other"] * num_samples
    
    def _generate_dummy_pretraining_data(self, num_samples: int) -> List[str]:
        """Generate dummy pretraining-like data."""
        templates = [
            "The quick brown fox jumps over the lazy dog. This is a common sentence used for testing.",
            "In the field of artificial intelligence, machine learning has become increasingly important.",
            "Climate change is one of the most pressing issues of our time, affecting ecosystems worldwide.",
            "The development of renewable energy sources is crucial for sustainable development.",
            "Scientific research continues to advance our understanding of the natural world.",
        ]
        
        return [random.choice(templates) + f" Sample {i}" for i in range(num_samples)]
    
    def _generate_dummy_math_data(self, num_samples: int) -> List[str]:
        """Generate dummy math problems."""
        templates = [
            "If x + 5 = 12, what is the value of x?",
            "A rectangle has length 8 and width 6. What is its area?",
            "Solve for y: 2y - 3 = 7",
            "What is 15% of 200?",
            "If a triangle has angles of 60° and 70°, what is the third angle?",
        ]
        
        return [random.choice(templates) + f" Problem {i}" for i in range(num_samples)]
    
    def _generate_dummy_code_data(self, num_samples: int) -> List[str]:
        """Generate dummy code snippets."""
        templates = [
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "class Node:\n    def __init__(self, data):\n        self.data = data\n        self.next = None",
            "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:",
            "import numpy as np\n\ndef matrix_multiply(A, B):\n    return np.dot(A, B)",
            "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]",
        ]
        
        return [random.choice(templates) + f"\n# Example {i}" for i in range(num_samples)]
    
    def _generate_dummy_other_data(self, num_samples: int) -> List[str]:
        """Generate dummy other data."""
        templates = [
            "Breaking news: Scientists discover new species in the Amazon rainforest.",
            "The stock market showed mixed results today with technology stocks leading gains.",
            "Local weather forecast predicts sunny skies with temperatures reaching 75°F.",
            "New restaurant opens downtown featuring fusion cuisine from around the world.",
            "University researchers publish findings on sustainable agriculture practices.",
        ]
        
        return [random.choice(templates) + f" Article {i}" for i in range(num_samples)]

