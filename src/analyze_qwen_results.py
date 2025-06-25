import json
import numpy as np
from collections import defaultdict


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
    """Calculate the proportion correct for each question."""
    question_stats = []
    
    for i, result in enumerate(results):
        question = result["question"]
        canonical_answer = result["canonical_answer"]
        correctness = result["correctness"]
        
        # Calculate proportion correct
        total_attempts = len(correctness)
        correct_attempts = sum(correctness)
        proportion_correct = correct_attempts / total_attempts if total_attempts > 0 else 0
        
        question_stats.append({
            "index": i,
            "question": question,
            "canonical_answer": canonical_answer,
            "total_attempts": total_attempts,
            "correct_attempts": correct_attempts,
            "proportion_correct": proportion_correct
        })
    
    return question_stats


def sort_by_difficulty(question_stats):
    """Sort questions by proportion correct (easiest to hardest)."""
    return sorted(question_stats, key=lambda x: x["proportion_correct"], reverse=True)


def calculate_deciles(sorted_stats):
    """Calculate decile statistics."""
    n_questions = len(sorted_stats)
    decile_size = n_questions // 10
    
    deciles = []
    for decile in range(10):
        start_idx = decile * decile_size
        end_idx = start_idx + decile_size if decile < 9 else n_questions
        
        decile_questions = sorted_stats[start_idx:end_idx]
        
        if decile_questions:
            avg_proportion = np.mean([q["proportion_correct"] for q in decile_questions])
            min_proportion = min([q["proportion_correct"] for q in decile_questions])
            max_proportion = max([q["proportion_correct"] for q in decile_questions])
            
            deciles.append({
                "decile": decile + 1,
                "num_questions": len(decile_questions),
                "avg_proportion_correct": avg_proportion,
                "min_proportion_correct": min_proportion,
                "max_proportion_correct": max_proportion,
                "questions": decile_questions
            })
    
    return deciles


def print_analysis(data):
    """Print the complete analysis."""
    print("=" * 80)
    print("QWEN GSM8K EVALUATION RESULTS ANALYSIS")
    print("=" * 80)
    
    # Print evaluation settings
    settings = data.get("evaluation_settings", {})
    print(f"\nEvaluation Settings:")
    print(f"  Total Questions: {settings.get('total_questions', 'N/A')}")
    print(f"  Runs per Question: {settings.get('num_runs_per_question', 'N/A')}")
    print(f"  Batch Size: {settings.get('batch_size', 'N/A')}")
    
    # Print hyperparameters
    hyperparams = data.get("hyperparameters", {})
    print(f"\nHyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    # Calculate question difficulty
    print(f"\nCalculating question difficulty...")
    question_stats = calculate_question_difficulty(data["results"])
    
    # Sort by difficulty
    sorted_stats = sort_by_difficulty(question_stats)
    
    # Calculate overall statistics
    overall_accuracy = np.mean([q["proportion_correct"] for q in question_stats])
    print(f"\nOverall Statistics:")
    print(f"  Average Accuracy: {overall_accuracy:.3f} ({overall_accuracy:.1%})")
    print(f"  Easiest Question: {sorted_stats[0]['proportion_correct']:.3f} ({sorted_stats[0]['proportion_correct']:.1%})")
    print(f"  Hardest Question: {sorted_stats[-1]['proportion_correct']:.3f} ({sorted_stats[-1]['proportion_correct']:.1%})")
    
    # Calculate and print deciles
    deciles = calculate_deciles(sorted_stats)
    
    print(f"\n" + "=" * 80)
    print("DECILE ANALYSIS (Easiest to Hardest)")
    print("=" * 80)
    print(f"{'Decile':<8} {'Questions':<10} {'Avg Acc':<10} {'Min Acc':<10} {'Max Acc':<10} {'Range':<10}")
    print("-" * 80)
    
    for decile in deciles:
        range_val = decile["max_proportion_correct"] - decile["min_proportion_correct"]
        print(f"{decile['decile']:<8} {decile['num_questions']:<10} "
              f"{decile['avg_proportion_correct']:<10.3f} "
              f"{decile['min_proportion_correct']:<10.3f} "
              f"{decile['max_proportion_correct']:<10.3f} "
              f"{range_val:<10.3f}")
    
    # Print detailed breakdown for each decile
    print(f"\n" + "=" * 80)
    print("DETAILED DECILE BREAKDOWN")
    print("=" * 80)
    
    for decile in deciles:
        print(f"\nDecile {decile['decile']} (Easiest {decile['decile']*10}%):")
        print(f"  Average Accuracy: {decile['avg_proportion_correct']:.3f} ({decile['avg_proportion_correct']:.1%})")
        print(f"  Range: {decile['min_proportion_correct']:.3f} - {decile['max_proportion_correct']:.3f}")
        print(f"  Number of Questions: {decile['num_questions']}")
        
        # Show a few example questions from this decile
        print(f"  Example Questions:")
        for i, q in enumerate(decile['questions'][:3]):  # Show first 3 questions
            print(f"    {i+1}. Accuracy: {q['proportion_correct']:.3f} - {q['question'][:80]}...")
        if len(decile['questions']) > 3:
            print(f"    ... and {len(decile['questions']) - 3} more questions")


def save_detailed_analysis(data, output_filename="qwen_analysis_detailed.json"):
    """Save detailed analysis to a JSON file."""
    question_stats = calculate_question_difficulty(data["results"])
    sorted_stats = sort_by_difficulty(question_stats)
    deciles = calculate_deciles(sorted_stats)
    
    analysis_data = {
        "metadata": {
            "hyperparameters": data.get("hyperparameters", {}),
            "evaluation_settings": data.get("evaluation_settings", {}),
            "overall_accuracy": np.mean([q["proportion_correct"] for q in question_stats])
        },
        "question_difficulty": sorted_stats,
        "decile_analysis": deciles
    }
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed analysis saved to {output_filename}")


def main():
    """Main function to run the analysis."""
    # Load the results
    data = load_results()
    if data is None:
        return
    
    # Print the analysis
    print_analysis(data)
    
    # Save detailed analysis
    save_detailed_analysis(data)


if __name__ == "__main__":
    main() 