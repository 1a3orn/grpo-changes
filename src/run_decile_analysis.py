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


def calculate_proportions(results):
    """Calculate proportion correct for each question."""
    proportions = []
    
    for result in results:
        correctness = result["correctness"]
        total_attempts = len(correctness)
        correct_attempts = sum(correctness)
        proportion = correct_attempts / total_attempts if total_attempts > 0 else 0
        proportions.append(proportion)
    
    return proportions


def analyze_deciles(proportions):
    """Sort by difficulty and calculate decile statistics."""
    # Sort by proportion correct (easiest to hardest)
    sorted_proportions = sorted(proportions, reverse=True)
    
    n_questions = len(sorted_proportions)
    decile_size = n_questions // 10
    
    print(f"Total questions: {n_questions}")
    print(f"Decile size: {decile_size}")
    print()
    
    print("DECILE ANALYSIS (Easiest to Hardest)")
    print("=" * 50)
    print(f"{'Decile':<8} {'Avg Proportion':<15} {'Min':<8} {'Max':<8}")
    print("-" * 50)
    
    for decile in range(10):
        start_idx = decile * decile_size
        end_idx = start_idx + decile_size if decile < 9 else n_questions
        
        decile_proportions = sorted_proportions[start_idx:end_idx]
        
        if decile_proportions:
            avg_proportion = np.mean(decile_proportions)
            min_proportion = min(decile_proportions)
            max_proportion = max(decile_proportions)
            
            print(f"{decile + 1:<8} {avg_proportion:<15.3f} {min_proportion:<8.3f} {max_proportion:<8.3f}")


def main():
    """Main function."""
    # Load results
    data = load_results()
    if data is None:
        return
    
    # Calculate proportions
    proportions = calculate_proportions(data["results"])
    
    # Analyze deciles
    analyze_deciles(proportions)


if __name__ == "__main__":
    main() 