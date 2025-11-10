import json
import time
import requests
import re
import os
import statistics
from typing import Dict, List, Tuple
from datetime import datetime

class AIMEBenchmark:
    def __init__(self, api_url: str, model_name: str, output_dir: str = "aime_results", api_key: str = None):
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        self.output_dir = output_dir
        self.problems = self.load_problems()
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_problems(self) -> List[Dict]:
        """Load AIME problem dataset"""
        problems = []
        try:
            with open('/home/externals/suanfabu/x00806807/dataset/aime_2025.jsonl', 'r') as f:
                for line in f:
                    problem_data = json.loads(line)
                    # 适配AIME 2025格式
                    problem = {
                        'problem_idx': problem_data.get('problem_idx', problem_data.get('ID', 'Unknown')),
                        'problem': problem_data.get('problem', problem_data.get('Problem', '')),
                        'answer': problem_data.get('answer', problem_data.get('Answer', '')),
                        'problem_type': problem_data.get('problem_type', [])
                    }
                    problems.append(problem)
            print(f"Loaded {len(problems)} problems")
        except FileNotFoundError:
            print("Error: aime_2025.jsonl file not found")
        except Exception as e:
            print(f"Error loading problems: {e}")
            
        return problems
    
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
            'max_tokens': 20000,
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
        """Extract answer from model response"""
        # Match various possible answer formats for AIME
        patterns = [
            r'\\boxed\{(\d+)\}',
            r'answer[:]?\s*(\d+)',
            r'final answer[:]?\s*(\d+)',
            r'(\d{1,3})(?=\s*$)',
            r'\[(\d+)\]',
            r'[Aa]nswer:\s*(\d+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                # 返回最后一个匹配，通常是最终答案
                return matches[-1]
        
        return None
    
    def evaluate_problem(self, problem: Dict, response: str, time_taken: float, tokens_used: int) -> Dict:
        """Evaluate reasoning result for a single problem"""
        predicted_answer = self.extract_answer(response)
        correct_answer = str(problem['answer'])
        
        is_correct = predicted_answer == correct_answer if predicted_answer else False
        tps = tokens_used / time_taken if time_taken > 0 else 0
        
        return {
            'problem_idx': problem['problem_idx'],
            'problem': problem['problem'],
            'predicted': predicted_answer,
            'correct': correct_answer,
            'is_correct': is_correct,
            'time_taken': time_taken,
            'tokens_used': tokens_used,
            'tps': tps,
            'response': response,
            'problem_type': problem['problem_type']
        }
    
    def create_prompt(self, problem: Dict) -> str:
        """Create problem prompt in English for AIME format"""
        return f"""Please solve the following AIME math problem. Show your step-by-step reasoning clearly, and provide your final answer in the format \\boxed{{answer}}.

Problem: {problem['problem']}

Please show your complete reasoning process:"""

    def save_single_response(self, problem_idx: str, response: str, timestamp: str):
        """Save individual problem response to file"""
        filename = f"{self.output_dir}/response_{problem_idx}_{timestamp}.txt"
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
            print("No problems loaded. Please check if aime_2024.jsonl exists.")
            return {}
        
        test_problems = self.problems[:num_problems] if num_problems else self.problems
        
        print(f"Starting AIME benchmark with {len(test_problems)} problems...")
        print(f"Model: {self.model_name}")
        print("-" * 60)
        
        for i, problem in enumerate(test_problems, 1):
            print(f"Processing problem {i}/{len(test_problems)}: {problem['problem_idx']}")
            print(f"Problem: {problem['problem'][:100]}...")
            
            prompt = self.create_prompt(problem)
            response, time_taken, tokens_used = self.call_model(prompt)
            
            # Save individual response if requested
            if save_responses:
                saved_file = self.save_single_response(problem['problem_idx'], response, timestamp)
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
            print("-" * 40)
            
            # Avoid making requests too frequently
            time.sleep(0.5)
        
        # Calculate statistics
        if results:
            accuracy = sum(1 for r in results if r['is_correct']) / len(results)
            tps_values = [r['tps'] for r in results if r['tps'] > 0]
            avg_tps = statistics.mean(tps_values) if tps_values else 0
            avg_time = statistics.mean([r['time_taken'] for r in results])
            
            # Calculate accuracy by problem type
            problem_types = {}
            for result in results:
                for p_type in result['problem_type']:
                    if p_type not in problem_types:
                        problem_types[p_type] = {'total': 0, 'correct': 0}
                    problem_types[p_type]['total'] += 1
                    if result['is_correct']:
                        problem_types[p_type]['correct'] += 1
            
            # Calculate accuracy for each problem type
            type_accuracy = {}
            for p_type, counts in problem_types.items():
                type_accuracy[p_type] = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
        else:
            accuracy = 0
            avg_tps = 0
            avg_time = 0
            type_accuracy = {}
        
        summary = {
            'model': self.model_name,
            'timestamp': timestamp,
            'total_problems': len(test_problems),
            'accuracy': accuracy,
            'average_tps': avg_tps,
            'average_time_per_problem': avg_time,
            'total_tokens': total_tokens,
            'total_time': total_time,
            'type_accuracy': type_accuracy,
            'detailed_results': results
        }
        
        return summary
    
    def save_results(self, summary: Dict, filename: str = None):
        """Save test results"""
        if filename is None:
            filename = f"{self.output_dir}/aime_benchmark_{self.model_name.replace('/', '_')}_{summary['timestamp']}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nTest results saved to: {filename}")
        print(f"Model: {summary['model']}")
        print(f"Accuracy: {summary['accuracy']:.2%} ({sum(1 for r in summary['detailed_results'] if r['is_correct'])}/{summary['total_problems']})")
        print(f"Average TPS: {summary['average_tps']:.2f}")
        print(f"Average time per problem: {summary['average_time_per_problem']:.2f}s")
        print(f"Total tokens used: {summary['total_tokens']}")
        print(f"Total time: {summary['total_time']:.2f}s")
        
        if summary['type_accuracy']:
            print("\nAccuracy by problem type:")
            for p_type, acc in summary['type_accuracy'].items():
                print(f"  {p_type}: {acc:.2%}")
    
    def print_detailed_results(self, summary: Dict):
        """Print detailed results for each problem"""
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)
        
        for result in summary['detailed_results']:
            status = "✓" if result['is_correct'] else "✗"
            predicted = result['predicted'] if result['predicted'] else "No answer"
            print(f"{status} Problem {result['problem_idx']}:")
            print(f"    Predicted: {predicted} | Correct: {result['correct']}")
            print(f"    Time: {result['time_taken']:.2f}s | TPS: {result['tps']:.2f}")
            print(f"    Types: {', '.join(result['problem_type'])}")
            print()

def main():
    # Configuration for vLLM service
    API_URL = "http://localhost:8099/v1/completions"  # vLLM completions endpoint
    MODEL_NAME = "DeepSeek-R1-Distill-Qwen-32B"
    API_KEY = None  # vLLM typically doesn't require API key
    
    # Create test instance
    benchmark = AIMEBenchmark(API_URL, MODEL_NAME)
    
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
    # num_problems = 2  # Set to None to test all problems
    # print(f"\nStarting benchmark with {num_problems if num_problems else 'all'} problems...")
    
    results = benchmark.run_benchmark(num_problems=None, save_responses=True)
    
    # Save and display results
    if results:
        benchmark.save_results(results)
        # benchmark.print_detailed_results(results)
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()