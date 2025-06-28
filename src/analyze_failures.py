import json
import argparse
import os
from qwen_generator import QwenGenerator, GenerationConfig


class FailureAnalyzer:
    """Analyzes eval results to find failed questions and generate improvement principles."""
    
    def __init__(self):
        """Initialize the analyzer with Qwen model."""
        self.qwen = QwenGenerator()
        self.analysis_config = GenerationConfig(
            temperature=0.7,
            max_tokens=2000,
            top_p=0.95
        )
        
    def load_eval_file(self, filepath):
        """Load an eval JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def find_failed_questions(self, eval_data):
        """Find questions where the LLM didn't get a perfect score."""
        failed_questions = []
        
        for i, result in enumerate(eval_data['results']):
            # Check if all attempts were correct
            if not all(result['correctness']):
                failed_questions.append({
                    'index': i,
                    'question': result['question'],
                    'canonical_answer': result['canonical_answer'],
                    'qwen_outputs': result['qwen_outputs'],
                    'correctness': result['correctness']
                })
        
        return failed_questions
    
    def generate_analysis_prompt(self, question, canonical_answer, qwen_outputs, correctness):
        """Generate a prompt for analyzing a failed question."""
        
        # Find a failed attempt to analyze
        failed_attempts = [i for i, correct in enumerate(correctness) if not correct]
        if not failed_attempts:
            return None
        
        # Use the first failed attempt
        failed_idx = failed_attempts[0]
        failed_output = qwen_outputs[failed_idx]
        
        prompt = f"""Your task is to analyze reasoning errors in logic puzzles and mathematical problems.

I will show you:
1. A question that was given to an AI
2. The AI's chain-of-thought reasoning (which contains an error)
3. The correct answer

Your task is to:
1. Glance at the AI's reasoning
2. Identify the kind of thing that is going wrong in the reasoning -- broadly, in a vibe-based way
3. Generate a broad, general principle -- NOT specific to this kind of question -- that would help avoid this type of error
4. Put the single sentence giving a general principle for avoiding this type of mistake inside a <principle>...</principle> tag.

The principle should be something general and actionable, not specific to details of this question.
It should be the kind of thing that helps reasoning in general, not solving this kind of problem.

Below is the question, the correct answer, and the AI's reasoning.

Question: {question}

Correct Answer: {canonical_answer}

AI's Reasoning:
{failed_output}

Above was the question, the correct answer, and the AI's reasoning.

Again, please analyze the AI's reasoning, identify its error, and provide a general principle for avoiding this type of mistake.

Put the single sentence, giving a general principle for avoiding this type of mistake inside a <principle>...</principle> tag.

To repeat my instructions:
1. Carefully read the AI's reasoning
2. Identify, in a abroad vibe-based fashion, the kind of thing going wrong in the reasoning
3. Generate a broad, general principle -- not specific to this kind of question -- that would help avoid this type of error
4. Put the sentence giving a general principle for avoiding this type of mistake inside a <principle>...</principle> tag.

"""
        
        return prompt
    
    def analyze_failed_question(self, failed_question):
        """Analyze a single failed question and generate a principle."""
        prompt = self.generate_analysis_prompt(
            failed_question['question'],
            failed_question['canonical_answer'],
            failed_question['qwen_outputs'],
            failed_question['correctness']
        )
        
        if not prompt:
            return None
        
        try:
            # Generate analysis using Qwen
            response = self.qwen.generate_chat(prompt, config=self.analysis_config)
            
            # Extract the principle from the boxed response
            import re
            boxed_match = re.search(r'<principle>([^<]*)</principle>', response)
            if boxed_match:
                principle = boxed_match.group(1).strip()
            else:
                principle = response.strip()
            
            return {
                'raw_chain_of_thought': response,
                'principle': principle
            }
            
        except Exception as e:
            print(f"Error analyzing question: {e}")
            return {
                'raw_chain_of_thought': f"Error during analysis: {e}",
                'principle': "Error occurred during analysis"
            }
    
    def process_eval_file(self, input_filepath):
        """Process an eval file and generate analysis."""
        print(f"Loading eval file: {input_filepath}")
        eval_data = self.load_eval_file(input_filepath)
        
        print("Finding failed questions...")
        failed_questions = self.find_failed_questions(eval_data)
        print(f"Found {len(failed_questions)} questions with failures")
        
        # Analyze each failed question
        for i, failed_question in enumerate(failed_questions):
            print(f"Analyzing failed question {i+1}/{len(failed_questions)}...")
            analysis = self.analyze_failed_question(failed_question)
            print(i)
            if i > 100:
                break
            
            if analysis:
                # Add analysis to the original result
                original_index = failed_question['index']
                eval_data['results'][original_index]['failure_analysis'] = analysis
        
        # Generate output filename
        base_name = os.path.splitext(input_filepath)[0]
        output_filepath = f"{base_name}_and_extra_thoughts.json"
        
        # Save the enhanced eval data
        print(f"Saving enhanced results to: {output_filepath}")
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
        
        print(f"Analysis complete! Enhanced file saved to: {output_filepath}")
        return output_filepath


def main():
    parser = argparse.ArgumentParser(description='Analyze eval results and generate improvement principles')
    parser.add_argument('input_file', help='Path to the eval JSON file to analyze')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: File {args.input_file} does not exist")
        return
    
    analyzer = FailureAnalyzer()
    output_file = analyzer.process_eval_file(args.input_file)
    
    print(f"\nProcessing complete!")
    print(f"Input: {args.input_file}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main() 