"""
GSM8K Dataset Loader

Loads the GSM8K test dataset and converts it to standardized JSONL format
with question and canonical_answer fields.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GSM8KLoader:
    """Dedicated loader for GSM8K dataset."""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def _remove_angle_brackets_content(self, text: str) -> str:
        """
        Remove all text within <<...>> tags.
        """
        # Remove content within <<...>> tags, including the tags themselves
        cleaned_text = re.sub(r'<<[^>]*>>', '', text)
        # Clean up any extra whitespace that might be left
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text
        
    def load_test_set(self) -> List[Dict[str, str]]:
        """
        Load GSM8K test set and convert to standardized format.
        
        Returns:
            List of dictionaries with 'question' and 'canonical_answer' fields
        """
        logger.info("Loading GSM8K test dataset...")
        
        try:
            # Load the dataset from HuggingFace
            dataset = load_dataset("gsm8k", "main", split="test")
            logger.info(f"Loaded {len(dataset)} examples from GSM8K test set")
            
            # Convert to standardized format
            processed_data = []
            for example in dataset:
                canonical_answer = self._remove_angle_brackets_content(example["answer"]).split("####")[1].strip()
                processed_example = {
                    "question": self._remove_angle_brackets_content(example["question"]),
                    "canonical_answer": canonical_answer
                }
                processed_data.append(processed_example)
            
            output_path = self.output_dir / "gsm8k_test.jsonl"
            self._save_jsonl(processed_data, output_path)
            logger.info(f"Saved processed data to {output_path}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to load GSM8K dataset: {e}")
            raise
    
    def _save_jsonl(self, data: List[Dict[str, str]], output_path: Path) -> None:
        """
        Save data to JSONL format.
        
        Args:
            data: List of dictionaries to save
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"Saved {len(data)} examples to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save JSONL file {output_path}: {e}")
            raise

def main():
    """Load GSM8K test set and save to JSONL."""
    loader = GSM8KLoader()
    
    try:
        # Load GSM8K test set
        print("=== Loading GSM8K Test Set ===")
        gsm8k_data = loader.load_test_set()
        
        # Show first few examples
        print("\n=== First 3 Examples ===")
        for i, example in enumerate(gsm8k_data[:3]):
            print(f"\nExample {i+1}:")
            print(f"Question: {example['question'][:1000]}...")
            print(f"Answer: {example['canonical_answer']}")
        
    except Exception as e:
        logger.error(f"Failed to load GSM8K dataset: {e}")


if __name__ == "__main__":
    main() 