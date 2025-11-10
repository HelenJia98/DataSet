import json
import time
import requests
import re
import os
import statistics
from typing import Dict, List, Tuple
from datetime import datetime

class GPQABenchmark:
    def __init__(self, api_url: str, model_name: str, output_dir: str = "gpqa_results", api_key: str = None):
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        self.output_dir = output_dir
        self.problems = self.load_problems()
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_problems(self) -> List[Dict]:
        """Load GPQA problem dataset"""
        problems = []
        try:
            with open('/home/externals/suanfabu/x00806807/dataset/gpqa_diamond.jsonl', 'r') as f:
                for line in f:
                    problem_data = json.loads(line)
                    # 适配GPQA格式
                    problem = {
                        'question': problem_data.get('question', ''),
                        'options': self.extract_options(problem_data.get('question', '')),
                        'answer': problem_data.get('answer', ''),
                        'question_id': problem_data.get('id', f'gpqa_{len(problems)+1}')
                    }
                    problems.append(problem)
            print(f"Loaded {len(problems)} GPQA problems")
        except FileNotFoundError:
            print("Error: gpqa_diamond.jsonl file not found")
        except Exception as e:
            print(f"Error loading problems: {e}")
            
        return problems
    
    def extract_options(self, question: str) -> Dict[str, str]:
        """Extract options from question text"""
        options = {}
        # 匹配选项模式：a) ... b) ... c) ... d) ...
        pattern = r'([a-d])\)\s*([^\n]+?)(?=\n[a-d]\)|\n[A-D]\.|\n*$)'
        matches = re.findall(pattern, question)
        
        for option_letter, option_text in matches:
            options[option_letter.lower()] = option_text.strip()
        
        return options
    
    def call_model(self, prompt: str) -> Tuple[str, float, int]:
        """
        Call the vLLM inference service
        Returns: (response content, time taken, number of tokens)
        """
        headers = {
            'Content-Type': 'application/json'
        }
        
        # vLLM API payload format
        payload = {
            'model': self.model_name,
            'prompt': prompt,
            'temperature': 0,
            'max_tokens': 1024,
            'stream': False
        }
        
        start_time = time.time()
        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            end_time = time.time()
            
            # Extract response content from vLLM format
            content = result['choices'][0]['text']
            
            # Token usage
            total_tokens = result.get('usage', {}).get('total_tokens', 0)
            
            return content, end_time - start_time, total_tokens
            
        except Exception as e:
            print(f"API call error: {e}")
            if 'response' in locals():
                print(f"Response: {response.text}")
            return "", 0, 0
    
    def extract_answer(self, response: str) -> str:
        """Extract answer from model response for multiple choice questions"""
        # Match various possible answer formats
        patterns = [
            r'[Aa]nswer:\s*([A-Da-d])',
            r'[Tt]he correct answer is\s*([A-Da-d])',
            r'[Ff]inal answer:\s*([A-Da-d])',
            r'\\boxed\{([A-Da-d])\}',
            r'[\[\(]\s*([A-Da-d])\s*[\]\)]',
            r'\b([A-Da-d])\b(?=\s*$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                # 返回最后一个匹配，通常是最终答案
                answer = matches[-1].upper()
                # 确保答案是A、B、C、D之一
                if answer in ['A', 'B', 'C', 'D']:
                    return answer
        
        # 如果没有明确匹配，尝试从文本中提取
        if 'A' in response.upper()[-50:]:  # 检查最后50个字符
            return 'A'
        elif 'B' in response.upper()[-50:]:
            return 'B'
        elif 'C' in response.upper()[-50:]:
            return 'C'
        elif 'D' in response.upper()[-50:]:
            return 'D'
        
        return None
    
    def evaluate_problem(self, problem: Dict, response: str, time_taken: float, tokens_used: int) -> Dict:
        """Evaluate reasoning result for a single problem"""
        predicted_answer = self.extract_answer(response)
        correct_answer = problem['answer'].upper()
        
        is_correct = predicted_answer == correct_answer if predicted_answer else False
        tps = tokens_used / time_taken if time_taken > 0 else 0
        
        return {
            'question_id': problem['question_id'],
            'question_preview': problem['question'][:100] + "...",
            'predicted': predicted_answer,
            'correct': correct_answer,
            'is_correct': is_correct,
            'time_taken': time_taken,
            'tokens_used': tokens_used,
            'tps': tps,
            'response': response,
            'options': problem['options']
        }
    
    def create_prompt(self, problem: Dict) -> str:
        """Create problem prompt in English for GPQA format"""
        options_text = ""
        for letter, text in problem['options'].items():
            options_text += f"{letter.upper()}) {text}\n"
        
        return f"""Please solve the following multiple-choice question. Show your step-by-step reasoning clearly, and provide your final answer as a single letter (A, B, C, or D).

Question: {problem['question']}

Options:
{options_text}

Please show your complete reasoning process and end with "Answer: X" where X is the correct letter:"""

    def save_single_response(self, question_id: str, response: str, timestamp: str):
        """Save individual problem response to file"""
        filename = f"{self.output_dir}/response_{question_id}_{timestamp}.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(response)
            return filename
        except Exception as e:
            print(f"Failed to save response: {e}")
            return None
    
    def run_benchmark(self, num_problems: int = None, save_responses: bool = True) -> Dict:
        """Run the complete benchmark test"""
        results = []
        total_tokens = 0
        total_time = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not self.problems:
            print("No problems loaded. Please check if gpqa_diamond.jsonl exists.")
            return {}
        
        test_problems = self.problems[:num_problems] if num_problems else self.problems
        
        print(f"Starting GPQA benchmark with {len(test_problems)} problems...")
        print(f"Model: {self.model_name}")
        print("-" * 80)
        
        for i, problem in enumerate(test_problems, 1):
            print(f"Processing problem {i}/{len(test_problems)}: {problem['question_id']}")
            print(f"Question: {problem['question'][:80]}...")
            
            prompt = self.create_prompt(problem)
            response, time_taken, tokens_used = self.call_model(prompt)
            
            # Save individual response if requested
            if save_responses:
                saved_file = self.save_single_response(problem['question_id'], response, timestamp)
                if saved_file:
                    print(f"  Response saved: {saved_file}")
            
            result = self.evaluate_problem(problem, response, time_taken, tokens_used)
            results.append(result)
            
            total_tokens += tokens_used
            total_time += time_taken
            
            status = "✓ CORRECT" if result['is_correct'] else "✗ INCORRECT"
            print(f"  Result: {status} | Time: {time_taken:.2f}s | TPS: {result['tps']:.2f}")
            if result['predicted']:
                print(f"  Predicted: {result['predicted']} | Correct: {result['correct']}")
            else:
                print(f"  No answer extracted | Correct: {result['correct']}")
            print("-" * 60)
            
            # Avoid making requests too frequently
            time.sleep(0.5)
        
        # Calculate statistics
        if results:
            accuracy = sum(1 for r in results if r['is_correct']) / len(results)
            tps_values = [r['tps'] for r in results if r['tps'] > 0]
            avg_tps = statistics.mean(tps_values) if tps_values else 0
            avg_time = statistics.mean([r['time_taken'] for r in results])
            
            # Calculate confusion matrix
            confusion_matrix = self.calculate_confusion_matrix(results)
            
        else:
            accuracy = 0
            avg_tps = 0
            avg_time = 0
            confusion_matrix = {}
        
        summary = {
            'model': self.model_name,
            'timestamp': timestamp,
            'total_problems': len(test_problems),
            'accuracy': accuracy,
            'average_tps': avg_tps,
            'average_time_per_problem': avg_time,
            'total_tokens': total_tokens,
            'total_time': total_time,
            'confusion_matrix': confusion_matrix,
            'detailed_results': results
        }
        
        return summary
    
    def calculate_confusion_matrix(self, results: List[Dict]) -> Dict:
        """Calculate confusion matrix for multiple choice questions"""
        confusion = {}
        all_answers = ['A', 'B', 'C', 'D']
        
        # Initialize confusion matrix
        for true_ans in all_answers:
            confusion[true_ans] = {}
            for pred_ans in all_answers:
                confusion[true_ans][pred_ans] = 0
            confusion[true_ans]['None'] = 0  # For no answer cases
        
        # Fill confusion matrix
        for result in results:
            true_ans = result['correct']
            pred_ans = result['predicted'] if result['predicted'] else 'None'
            
            if true_ans in confusion and pred_ans in confusion[true_ans]:
                confusion[true_ans][pred_ans] += 1
        
        return confusion
    
    def save_results(self, summary: Dict, filename: str = None):
        """Save test results"""
        if filename is None:
            filename = f"{self.output_dir}/gpqa_benchmark_{self.model_name.replace('/', '_')}_{summary['timestamp']}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nTest results saved to: {filename}")
        print(f"Model: {summary['model']}")
        print(f"Accuracy: {summary['accuracy']:.2%} ({sum(1 for r in summary['detailed_results'] if r['is_correct'])}/{summary['total_problems']})")
        print(f"Average TPS: {summary['average_tps']:.2f}")
        print(f"Average time per problem: {summary['average_time_per_problem']:.2f}s")
        print(f"Total tokens used: {summary['total_tokens']}")
        print(f"Total time: {summary['total_time']:.2f}s")
        
        # Print confusion matrix
        if summary['confusion_matrix']:
            print("\nConfusion Matrix:")
            self.print_confusion_matrix(summary['confusion_matrix'])
    
    def print_confusion_matrix(self, confusion_matrix: Dict):
        """Print formatted confusion matrix"""
        all_answers = ['A', 'B', 'C', 'D']
        
        # Header
        print("True\\Pred", end="")
        for pred in all_answers + ['None']:
            print(f"{pred:>8}", end="")
        print()
        
        # Rows
        for true in all_answers:
            print(f"{true:>10}", end="")
            for pred in all_answers + ['None']:
                count = confusion_matrix[true].get(pred, 0)
                print(f"{count:>8}", end="")
            print()
    
    def print_detailed_results(self, summary: Dict):
        """Print detailed results for each problem"""
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)
        
        for result in summary['detailed_results']:
            status = "✓" if result['is_correct'] else "✗"
            predicted = result['predicted'] if result['predicted'] else "No answer"
            print(f"{status} Problem {result['question_id']}:")
            print(f"    Predicted: {predicted} | Correct: {result['correct']}")
            print(f"    Time: {result['time_taken']:.2f}s | TPS: {result['tps']:.2f}")
            print(f"    Question: {result['question_preview']}")
            print()

def main():
    # Configuration for vLLM service
    API_URL = "http://localhost:8099/v1/completions"  # vLLM completions endpoint
    MODEL_NAME = "DeepSeek-R1-Distill-Qwen-32B"
    API_KEY = None  # vLLM typically doesn't require API key
    
    # Create test instance
    benchmark = GPQABenchmark(API_URL, MODEL_NAME)
    
    # Test connection first
    print("Testing connection to vLLM service...")
    test_prompt = "What is 2+2? Answer with a single number."
    try:
        response, time_taken, tokens = benchmark.call_model(test_prompt)
        print(f"Connection test successful. Response: {response.strip()}")
        print(f"Test time: {time_taken:.2f}s, Tokens: {tokens}")
    except Exception as e:
        print(f"Connection test failed: {e}")
        return
    
    # Run test
    # num_problems = 10  # Set to None to test all problems
    # print(f"\nStarting GPQA benchmark with {num_problems if num_problems else 'all'} problems...")
    
    results = benchmark.run_benchmark(num_problems=None, save_responses=False)
    
    # Save and display results
    if results:
        benchmark.save_results(results)
        # benchmark.print_detailed_results(results)
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()