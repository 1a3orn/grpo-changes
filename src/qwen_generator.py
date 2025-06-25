"""
Qwen3-1.7B Text Generation using VLLM

A clean, modular implementation for generating text continuations
using the Qwen/Qwen3-1.7B model with VLLM for efficient inference.
Supports both chat mode (with chat template) and prefill mode.
"""

from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation parameters."""
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None


class QwenGenerator:
    """Clean interface for generating text continuations with Qwen3-1.7B."""
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen3-1.7B",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Qwen generator.
        
        Args:
            model_name: HuggingFace model identifier
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            system_prompt: Optional system prompt to include in all conversations
            **kwargs: Additional VLLM initialization parameters
        """
        self.model_name = model_name
        self.llm = None
        self.tokenizer = None
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.system_prompt = system_prompt
        self.init_kwargs = kwargs
        
    def load_model(self) -> None:
        """Load the VLLM model and tokenizer."""
        if self.llm is not None:
            logger.info("Model already loaded")
            return
            
        logger.info(f"Loading model: {self.model_name}")
        try:
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load VLLM model
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                **self.init_kwargs
            )
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _apply_chat_template(self, question: str) -> str:
        """Apply the chat template to format the question."""
        if self.tokenizer is None:
            self.load_model()
            
        # Build messages list, starting with system prompt if provided
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": question})
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return formatted_prompt
    
    def _prepare_prefill_prompt(self, question: str, prefill: str) -> str:
        """Prepare prompt with prefill text."""
        if self.tokenizer is None:
            self.load_model()
            
        # Build messages list, starting with system prompt if provided
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": question})
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Add the prefill text
        return formatted_prompt + prefill
    
    def generate_chat(
        self, 
        questions: Union[str, List[str]], 
        config: Optional[GenerationConfig] = None
    ) -> Union[str, List[str]]:
        """
        Generate responses using chat mode (applies chat template).
        
        Args:
            questions: Question(s) to ask
            config: Generation configuration parameters
            
        Returns:
            Generated response(s)
        """
        if self.llm is None:
            self.load_model()
            
        if config is None:
            config = GenerationConfig()
            
        # Handle single question vs list
        single_input = isinstance(questions, str)
        if single_input:
            questions = [questions]
            
        # Apply chat template to all questions
        formatted_prompts = [self._apply_chat_template(q) for q in questions]
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            stop=config.stop,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
        )
        
        # Generate responses
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        
        # Extract generated text
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
            
        return results[0] if single_input else results
    
    def generate_with_prefill(
        self, 
        questions: Union[str, List[str]], 
        prefills: Union[str, List[str]],
        config: Optional[GenerationConfig] = None
    ) -> Union[str, List[str]]:
        """
        Generate responses with prefill text (includes prefill in output).
        
        Args:
            questions: Question(s) to ask
            prefills: Prefill text(s) to start the response with
            config: Generation configuration parameters
            
        Returns:
            Generated response(s) including the prefill text
        """
        if self.llm is None:
            self.load_model()
            
        if config is None:
            config = GenerationConfig()
            
        # Handle single input vs list
        single_input = isinstance(questions, str)
        if single_input:
            questions = [questions]
            prefills = [prefills]
            
        if len(questions) != len(prefills):
            raise ValueError("Number of questions must match number of prefills")
            
        # Prepare prompts with prefill
        formatted_prompts = [
            self._prepare_prefill_prompt(q, p) for q, p in zip(questions, prefills)
        ]
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            stop=config.stop,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
        )
        
        # Generate responses
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        
        # Extract generated text (includes prefill)
        results = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            # The generated text includes the prefill, so we return it as is
            results.append(generated_text)
            
        return results[0] if single_input else results
    
    def generate(
        self, 
        prompts: List[str], 
        config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """
        Legacy method for direct prompt generation (no chat template).
        
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
        Legacy method for single prompt generation (no chat template).
        
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
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        logger.info("Model and tokenizer unloaded")


def main():
    """Example usage of the QwenGenerator with both modes."""
    # Example questions
    questions = [
        "What is a Transformer in AI?",
        "The best way to learn programming is to do what?"
    ]
    
    # Example prefills
    prefills = [
        "A Transformer is a type of",
        "The best way to learn programming is to practice by"
    ]
    
    # Create generator
    generator = QwenGenerator()
    
    try:
        print("=== CHAT MODE ===")
        # Generate responses using chat mode
        chat_results = generator.generate_chat(questions)
        
        for i, (question, result) in enumerate(zip(questions, chat_results)):
            print(f"\n--- Chat Generation {i+1} ---")
            print(f"Question: {question}")
            print(f"Response: {result}")
        
        print("\n=== PREFILL MODE ===")
        # Generate responses with prefill
        prefill_results = generator.generate_with_prefill(questions, prefills)
        
        for i, (question, prefill, result) in enumerate(zip(questions, prefills, prefill_results)):
            print(f"\n--- Prefill Generation {i+1} ---")
            print(f"Question: {question}")
            print(f"Prefill: {prefill}")
            print(f"Full Response: {result}")
            
    except Exception as e:
        logger.error(f"Generation failed: {e}")
    finally:
        generator.close()


if __name__ == "__main__":
    main() 