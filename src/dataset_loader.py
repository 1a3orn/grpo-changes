"""
Generic Dataset Loader

A reusable class for loading various datasets and converting them to JSONL format
with standardized question and canonical_answer fields.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union
from datasets import load_dataset
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenericDatasetLoader:
    """Reusable dataset loader for various dataset formats."""
    
    def __init__(self, output_dir: str = "data"):
        """
        Initialize the generic dataset loader.
        
        Args:
            output_dir: Directory to save processed datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self._data = []  # Store loaded data
    
    def __len__(self) -> int:
        """
        Return the number of items in the dataset.
        
        Returns:
            Number of items in the dataset
        """
        return len(self._data)
    
    def __getitem__(self, index: Union[int, slice]) -> Union[Dict[str, str], List[Dict[str, str]]]:
        """
        Get item(s) by index or slice.
        
        Args:
            index: Integer index or slice
            
        Returns:
            Single item or list of items
        """
        return self._data[index]
    
    def __iter__(self):
        """
        Make the loader iterable.
        
        Returns:
            Iterator over the dataset
        """
        return iter(self._data)
         
    def load_from_jsonl(self, file_path: Union[str, Path]) -> List[Dict[str, str]]:
        """
        Load data from an existing JSONL file.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            List of dictionaries with 'question' and 'canonical_answer' fields
        """
        file_path = Path(file_path)
        logger.info(f"Loading data from {file_path}")
        
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            
            self._data = data  # Store the loaded data
            logger.info(f"Loaded {len(data)} examples from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load JSONL file {file_path}: {e}")
            raise

    def load_from_json(self, file_path: Union[str, Path]) -> List[Dict[str, str]]:
        """
        Load data from a JSON file containing a list of dictionaries.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of dictionaries with 'question' and 'canonical_answer' fields
        """
        file_path = Path(file_path)
        logger.info(f"Loading data from {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Ensure data is a list
            if not isinstance(data, list):
                raise ValueError(f"JSON file {file_path} does not contain a list")
            
            self._data = data  # Store the loaded data
            logger.info(f"Loaded {len(data)} examples from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {e}")
            raise

    def load_data(self, file_path: Union[str, Path]) -> List[Dict[str, str]]:
        """
        Load data from a file, automatically detecting JSON or JSONL format.
        
        Args:
            file_path: Path to the JSON or JSONL file
            
        Returns:
            List of dictionaries with 'question' and 'canonical_answer' fields
        """
        file_path = Path(file_path)
        
        try:
            # Try to load as JSON first (list format)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                logger.info(f"Detected JSON list format in {file_path}")
                self._data = data
                logger.info(f"Loaded {len(data)} examples from {file_path}")
                return data
            else:
                raise ValueError(f"JSON file {file_path} does not contain a list")
                
        except json.JSONDecodeError:
            # If JSON parsing fails, try JSONL format
            logger.info(f"Detected JSONL format in {file_path}")
            return self.load_from_jsonl(file_path)
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            raise
    
    def validate_data(self, data: List[Dict[str, str]]) -> bool:
        """
        Validate that data has the correct format.
        
        Args:
            data: List of dictionaries to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        required_fields = {"question", "canonical_answer"}
        
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} is not a dictionary")
            
            missing_fields = required_fields - set(item.keys())
            if missing_fields:
                raise ValueError(f"Item {i} missing required fields: {missing_fields}")
            
            if not isinstance(item["question"], str) or not isinstance(item["canonical_answer"], str):
                raise ValueError(f"Item {i} has non-string question or answer")
        
        logger.info(f"Validated {len(data)} examples successfully")
        return True
    

def main():
    """Example usage of the GenericDatasetLoader."""
    loader = GenericDatasetLoader()

    # Use the new load_data method that handles both JSON and JSONL
    data = loader.load_data("data/gsm8k_test.jsonl")
    loader.validate_data(data)
    for i, item in enumerate(data):
        print("Question: ", item["question"])
        print("Answer: ", item["canonical_answer"])
        if i > 10:
            break


if __name__ == "__main__":
    main() 