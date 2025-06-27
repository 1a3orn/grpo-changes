import json
import random
import argparse
import re
from dataset_loader import GenericDatasetLoader
from qwen_generator import QwenGenerator, GenerationConfig
from utils_equality import compare_strings


class BoxedParser:
    """Simple parser to extract content from \\boxed{...} tags."""
    
    def __init__(self):
        self.boxed_pattern = re.compile(r'\\boxed\{([^}]*)\}')
    
    def parse(self, text):
        """
        Parse text and extract content from \\boxed{...} tags.
        
        Returns:
            tuple: (parsed_content, error_message)
            - parsed_content: The content inside \\boxed{...} if found, None otherwise
            - error_message: Error message if no \\boxed{...} found, None otherwise
        """
        match = self.boxed_pattern.search(text)
        if match:
            return match.group(1), None
        else:
            return None, "No \\boxed{...} tag found in the response"


# Hyperparameters - easily configurable at the top level
HYPERPARAMS = {
    "temperature": 0.85,
    "max_tokens": 12000,
    "top_p": 0.95,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}

# System prompt to guide Qwen's behavior
SYSTEM_PROMPT = """You are an intelligent assistant.

You will be given a question and you will need to answer it in the following format:
First, think step-by-step.
Then, give your answer inside a \\boxed{...} tag.

Make sure to:
- Break down complex problems into simpler steps
- Show all calculations clearly
- ONLY the final answer should be inside the \\boxed{...} tag.
- Use the exact tag format shown above.
"""

# Evaluation settings
BATCH_SIZE = 8  # Process 8 questions at a time


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run evaluation on a dataset with Qwen model')
    parser.add_argument('--num-runs', type=int, default=8, 
                       help='Number of runs per question (default: 8)')
    parser.add_argument('--datasource', type=str, default="src/generators/zebra/zebra_puzzles.json",
                       help='Path to the dataset file (default: src/generators/zebra/zebra_puzzles.json)')
    parser.add_argument('--num-questions', type=int, default=48,
                       help='Number of questions to evaluate (default: 48)')
    parser.add_argument('--output-file', type=str, default="eval_results_48.json",
                       help='Output file path (default: eval_results.json)')
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load questions from specified datasource
    loader = GenericDatasetLoader()
    data = loader.load_from_json(args.datasource)
    if len(data) > args.num_questions:
        data = random.sample(data, args.num_questions)
    else:
        data = data[:args.num_questions]

    # Initialize Qwen generator with custom config
    qwen = QwenGenerator(system_prompt=SYSTEM_PROMPT)
    generation_config = GenerationConfig(**HYPERPARAMS)
    boxed_parser = BoxedParser()

    results = []
    total_correct = 0
    total_attempts = 0

    for batch_start in range(0, len(data), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(data))
        batch_data = data[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//BATCH_SIZE + 1}/{(len(data) + BATCH_SIZE - 1)//BATCH_SIZE}")
        
        for run in range(args.num_runs):  # Use command-line argument for number of runs
            print(f"Run {run + 1}/{args.num_runs}")
            # Extract questions for this batch
            questions = [item["question"] for item in batch_data]
            
            # Generate Qwen outputs for the entire batch with custom config
            try:
                outputs = qwen.generate_chat(questions, config=generation_config)
            except Exception as e:
                outputs = [f"[GENERATION ERROR] {e}"] * len(questions)
            
            # Process each question in the batch
            for i, (item, output) in enumerate(zip(batch_data, outputs)):
                question = item["question"]
                canonical_answer = item["canonical_answer"]
                
                # Initialize results for this question if first run
                if run == 0:
                    results.append({
                        "question": question,
                        "canonical_answer": canonical_answer,
                        "qwen_outputs": [],
                        "parsed_outputs": [],
                        "correctness": []
                    })
                
                # Parse Qwen output using BoxedParser
                parsed_content, error = boxed_parser.parse(output)
                results[batch_start + i]["parsed_outputs"].append({"parsed": parsed_content, "error": error})
                results[batch_start + i]["qwen_outputs"].append(output)

                # Check correctness
                if parsed_content:
                    is_correct = compare_strings(parsed_content, canonical_answer)
                else:
                    is_correct = False
                results[batch_start + i]["correctness"].append(is_correct)
                
                if is_correct:
                    total_correct += 1
                total_attempts += 1
        
        # Print progress after each batch
        print(f"  Processed {batch_end} questions. Correct so far: {total_correct}/{total_attempts}")

    # Print final accuracy
    print(f"\Accuracy: {total_correct}/{total_attempts} = {total_correct/total_attempts:.2%}")
    print(f"Hyperparameters used: {HYPERPARAMS}")

    # Save results with hyperparameters included
    output_data = {
        "hyperparameters": HYPERPARAMS,
        "system_prompt": SYSTEM_PROMPT,
        "evaluation_settings": {
            "batch_size": BATCH_SIZE,
            "num_runs_per_question": args.num_runs,
            "total_questions": len(data),
            "datasource": args.datasource
        },
        "results": results
    }
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main() 