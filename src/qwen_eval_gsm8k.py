import json
import random
from dataset_loader import GenericDatasetLoader
from qwen_generator import QwenGenerator, GenerationConfig
from utils_tags import TagParser
from utils_equality import compare_numeric_strings


# Hyperparameters - easily configurable at the top level
HYPERPARAMS = {
    "temperature": 0.75,
    "max_tokens": 3000,
    "top_p": 0.95,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}

# System prompt to guide Qwen's behavior
SYSTEM_PROMPT = """You are an intelligent assistant.

You will be given a question and you will need to answer it in the following format:
First, think step-by-step.
Then, give your answer inside a \\boxed{...} tag, and format your answer as a Python list.
So the answer should be like this: \\boxed{['item1', 'item2', 'item3', ...]}.

Make sure to:
- Break down complex problems into simpler steps
- Show all calculations clearly
- ONLY the final answer should be inside the \\boxed{...} tag.
- Use the exact tag format shown above.
"""

# Evaluation settings
BATCH_SIZE = 8  # Process 4 questions at a time
NUM_RUNS_PER_QUESTION = 16  # Number of times to ask each question


def main():
    # Load 500 questions from GSM8K
    N = 100
    loader = GenericDatasetLoader()
    data = loader.load_from_jsonl("src/generators/zebra/zebra_puzzles.json")
    if len(data) > N:
        data = random.sample(data, N)
    else:
        data = data[:N]

    # Initialize Qwen generator with custom config
    qwen = QwenGenerator(system_prompt=SYSTEM_PROMPT)
    generation_config = GenerationConfig(**HYPERPARAMS)
    tag_parser = TagParser(["think", "answer"])

    results = []
    total_correct = 0
    total_attempts = 0

    for batch_start in range(0, len(data), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(data))
        batch_data = data[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//BATCH_SIZE + 1}/{(len(data) + BATCH_SIZE - 1)//BATCH_SIZE}")
        
        for run in range(NUM_RUNS_PER_QUESTION):  # 16 runs per question
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
                
                # Parse Qwen output
                parsed, error = tag_parser(output)
                results[batch_start + i]["parsed_outputs"].append({"parsed": parsed, "error": error})
                results[batch_start + i]["qwen_outputs"].append(output)

                # Check correctness
                if parsed and "answer" in parsed:
                    is_correct = compare_numeric_strings(parsed["answer"], canonical_answer)
                else:
                    is_correct = False
                results[batch_start + i]["correctness"].append(is_correct)
                
                if is_correct:
                    total_correct += 1
                total_attempts += 1
        
        # Print progress after each batch
        print(f"  Processed {batch_end} questions. Correct so far: {total_correct}/{total_attempts}")

    # Print final accuracy
    print(f"\nQwen accuracy: {total_correct}/{total_attempts} = {total_correct/total_attempts:.2%}")
    print(f"Hyperparameters used: {HYPERPARAMS}")

    # Save results with hyperparameters included
    output_data = {
        "hyperparameters": HYPERPARAMS,
        "system_prompt": SYSTEM_PROMPT,
        "evaluation_settings": {
            "batch_size": BATCH_SIZE,
            "num_runs_per_question": NUM_RUNS_PER_QUESTION,
            "total_questions": len(data)
        },
        "results": results
    }
    
    with open("qwen_gsm8k_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print("Results saved to qwen_gsm8k_eval_results.json")


if __name__ == "__main__":
    main() 