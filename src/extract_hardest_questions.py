import json
import numpy as np


def load_results(filename="qwen_gsm8k_eval_results.json"):
    """Load the evaluation results from the JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: Could not find {filename}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filename}")
        return None


def calculate_question_difficulty(results):
    """Calculate the proportion correct for each question and return with indices."""
    question_stats = []
    
    for i, result in enumerate(results):
        correctness = result["correctness"]
        total_attempts = len(correctness)
        correct_attempts = sum(correctness)
        proportion_correct = correct_attempts / total_attempts if total_attempts > 0 else 0
        
        question_stats.append({
            "index": i,
            "proportion_correct": proportion_correct
        })
    
    return question_stats


def get_hardest_questions(data, percentage=20):
    """Get the worst performing questions based on proportion correct."""
    question_stats = calculate_question_difficulty(data["results"])
    
    # Sort by proportion correct (ascending - hardest first)
    sorted_stats = sorted(question_stats, key=lambda x: x["proportion_correct"])
    
    # Calculate how many questions to include (worst 20%)
    n_questions = len(sorted_stats)
    n_hardest = int(n_questions * percentage / 100)
    
    print(f"Total questions: {n_questions}")
    print(f"Extracting worst {percentage}%: {n_hardest} questions")
    
    # Get indices of the hardest questions
    hardest_indices = [stats["index"] for stats in sorted_stats[:n_hardest]]
    
    # Print some statistics about the hardest questions
    hardest_proportions = [stats["proportion_correct"] for stats in sorted_stats[:n_hardest]]
    print(f"Hardest question accuracy: {min(hardest_proportions):.3f}")
    print(f"Easiest of hardest questions accuracy: {max(hardest_proportions):.3f}")
    print(f"Average accuracy of hardest questions: {np.mean(hardest_proportions):.3f}")
    
    return hardest_indices


def create_hardest_questions_file(data, hardest_indices, output_filename="qwen_hardest_20_percent.json"):
    """Create a new JSON file with only the hardest questions."""
    # Extract only the hardest questions
    hardest_results = [data["results"][i] for i in hardest_indices]
    
    # Create new data structure with same format
    new_data = {
        "hyperparameters": data.get("hyperparameters", {}),
        "system_prompt": data.get("system_prompt", ""),
        "evaluation_settings": data.get("evaluation_settings", {}),
        "results": hardest_results
    }
    
    # Update the evaluation settings to reflect the new number of questions
    new_data["evaluation_settings"]["total_questions"] = len(hardest_results)
    new_data["evaluation_settings"]["original_total_questions"] = len(data["results"])
    new_data["evaluation_settings"]["extraction_info"] = {
        "percentage_extracted": 20,
        "criteria": "worst performing questions by proportion correct"
    }
    
    # Save to file
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nHardest questions saved to: {output_filename}")
    print(f"New file contains {len(hardest_results)} questions (worst 20%)")
    
    return output_filename


def main():
    """Main function to extract the hardest questions."""
    # Load the original results
    data = load_results()
    if data is None:
        return
    
    # Get the hardest 20% of questions
    hardest_indices = get_hardest_questions(data, percentage=20)
    
    # Create the new file with hardest questions
    output_file = create_hardest_questions_file(data, hardest_indices)
    
    print(f"\nSuccessfully created {output_file} with the hardest 20% of questions!")
    print("The file maintains the same JSON structure as the original.")


if __name__ == "__main__":
    main() 