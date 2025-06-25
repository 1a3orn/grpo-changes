"""
Qwen3-1.7B Text Generation using VLLM

A clean, modular implementation for generating text continuations
using the Qwen/Qwen3-1.7B model with VLLM for efficient inference.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from vllm import LLM, SamplingParams
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation parameters."""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stop: Optional[List[str]] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class QwenGenerator:
    """Clean interface for generating text continuations with Qwen3-1.7B."""
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen3-1.7B",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        **kwargs
    ):
        """
        Initialize the Qwen generator.
        
        Args:
            model_name: HuggingFace model identifier
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            **kwargs: Additional VLLM initialization parameters
        """
        self.model_name = model_name
        self.llm = None
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.init_kwargs = kwargs
        
    def load_model(self) -> None:
        """Load the VLLM model."""
        if self.llm is not None:
            logger.info("Model already loaded")
            return
            
        logger.info(f"Loading model: {self.model_name}")
        try:
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                **self.init_kwargs
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(
        self, 
        prompts: List[str], 
        config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """
        Generate text continuations for the given prompts.
        
        Args:
            prompts: List of input prompts
            config: Generation configuration parameters
            
        Returns:
            List of generated text continuations
        """
        if self.llm is None:
            self.load_model()
            
        if config is None:
            config = GenerationConfig()
            
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            stop=config.stop,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
        )
        
        # Generate continuations
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Extract generated text
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
            
        return results
    
    def generate_single(
        self, 
        prompt: str, 
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate a single text continuation.
        
        Args:
            prompt: Input prompt
            config: Generation configuration parameters
            
        Returns:
            Generated text continuation
        """
        results = self.generate([prompt], config)
        return results[0] if results else ""
    
    def close(self) -> None:
        """Clean up resources."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            logger.info("Model unloaded")


def main():
    """Example usage of the QwenGenerator."""
    # Example prompts
    prompts = [
        "What is a Transformer in AI?",
        "The best way to learn programming is to do what?"
    ]
    
    # Create generator
    generator = QwenGenerator()
    
    try:
        # Generate continuations
        results = generator.generate(prompts)
        
        # Print results
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            print(f"\n--- Generation {i+1} ---")
            print(f"Prompt: {prompt}")
            print(f"Continuation: {result}")
            print(f"Full text: {prompt}{result}")
            
    except Exception as e:
        logger.error(f"Generation failed: {e}")
    finally:
        generator.close()


if __name__ == "__main__":
    main() 