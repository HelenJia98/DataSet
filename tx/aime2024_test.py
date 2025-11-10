import json
import time
import requests
import re
from typing import Dict, List, Tuple
import statistics

class AIMEBenchmark:
    def __init__(self, api_url: str, model_name: str, api_key: str = None):
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        self.problems = self.load_problems()
        
    def load_problems(self) -> List[Dict]:
        """Load AIME problem dataset"""
        problems = []
        with open('/home/externals/suanfabu/x00806807/dataset/aime_2024.jsonl', 'r') as f:
            for line in f:
                problems.append(json.loads(line))
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
            
            # vLLM may not provide token usage in standard response
            # You might need to check your vLLM configuration
            total_tokens = result.get('usage', {}).get('total_tokens', 0)
            
            return content, end_time - start_time, total_tokens
            
        except Exception as e:
            print(f"API call error: {e}")
            print(f"Response: {response.text if 'response' in locals() else 'No response'}")
            return "", 0, 0
    
    def extract_answer(self, response: str) -> str:
        """Extract answer from model response"""
        # Match various possible answer formats
        patterns = [
            r'\\boxed\{(\d+)\}',
            r'answer[:]?\s*(\d+)',
            r'final answer[:]?\s*(\d+)',
            r'(\d{2,3})(?=\s*$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                return match.group(1)
        
        return None
    
    def evaluate_problem(self, problem: Dict, response: str, time_taken: float, tokens_used: int) -> Dict:
        """Evaluate reasoning result for a single problem"""
        predicted_answer = self.extract_answer(response)
        correct_answer = str(problem['Answer'])
        
        is_correct = predicted_answer == correct_answer if predicted_answer else False
        tps = tokens_used / time_taken if time_taken > 0 else 0
        
        return {
            'problem_id': problem['ID'],
            'predicted': predicted_answer,
            'correct': correct_answer,
            'is_correct': is_correct,
            'time_taken': time_taken,
            'tokens_used': tokens_used,
            'tps': tps,
        }
    
    def create_prompt(self, problem: Dict) -> str:
        """Create problem prompt in English"""
        return f"""Please solve the following AIME math problem. Show your step-by-step reasoning, and provide your final answer in the format \\boxed{{answer}}.

Problem: {problem['Problem']}

Please show your complete reasoning process:"""

    def run_benchmark(self, num_problems: int = None) -> Dict:
        """Run the complete benchmark test"""
        results = []
        total_tokens = 0
        total_time = 0
        
        test_problems = self.problems[:num_problems] if num_problems else self.problems
        
        print(f"Starting test with {len(test_problems)} AIME problems...")
        
        for i, problem in enumerate(test_problems, 1):
            print(f"Processing problem {i}/{len(test_problems)}: {problem['ID']}")
            
            prompt = self.create_prompt(problem)
            response, time_taken, tokens_used = self.call_model(prompt)
            with open('aime_2024_response.txt', 'a', encoding='utf-8') as f:
                f.write(f"index {i}")
                f.write(response)
                f.write("\n")
        
            result = self.evaluate_problem(problem, response, time_taken, tokens_used)
            results.append(result)
            
            total_tokens += tokens_used
            total_time += time_taken
            
            print(f"  Result: {'Correct' if result['is_correct'] else 'Incorrect'} | "
                  f"Time: {time_taken:.2f}s | TPS: {result['tps']:.2f}")
            if result['predicted']:
                print(f"  Predicted: {result['predicted']} | Correct: {result['correct']}")
            else:
                print(f"  No answer extracted | Correct: {result['correct']}")
            
            # Avoid making requests too frequently
            time.sleep(0.5)
        
        # Calculate statistics
        accuracy = sum(1 for r in results if r['is_correct']) / len(results)
        tps_values = [r['tps'] for r in results if r['tps'] > 0]
        avg_tps = statistics.mean(tps_values) if tps_values else 0
        avg_time = statistics.mean([r['time_taken'] for r in results])
        
        summary = {
            'model': self.model_name,
            'total_problems': len(test_problems),
            'accuracy': accuracy,
            'average_tps': avg_tps,
            'average_time_per_problem': avg_time,
            'total_tokens': total_tokens,
            'total_time': total_time,
            'detailed_results': results
        }
        
        return summary
    
    def save_results(self, summary: Dict, filename: str):
        """Save test results"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nTest results saved to: {filename}")
        print(f"Model: {summary['model']}")
        print(f"Accuracy: {summary['accuracy']:.2%} ({sum(1 for r in summary['detailed_results'] if r['is_correct'])}/{summary['total_problems']})")
        print(f"Average TPS: {summary['average_tps']:.2f}")
        print(f"Average time per problem: {summary['average_time_per_problem']:.2f}s")

def main():
    # Configuration for vLLM service
    API_URL = "http://localhost:8099/v1/completions"  # vLLM completions endpoint
    MODEL_NAME = "DeepSeek-R1-Distill-Qwen-32B"
    API_KEY = None  # vLLM typically doesn't require API key
    
    # Create test instance
    benchmark = AIMEBenchmark(API_URL, MODEL_NAME, API_KEY)
    
    # Test connection first
    '''
    print("Testing connection to vLLM service...")
    test_prompt = "Say 'Hello' in one word."
    try:
        response, time_taken, tokens = benchmark.call_model(test_prompt)
        print(f"Connection test successful. Response: {response.strip()}")
    except Exception as e:
        print(f"Connection test failed: {e}")
        return
    '''
    
    # Run test (you can specify number of problems to test)
    results = benchmark.run_benchmark()
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"aime_benchmark_{MODEL_NAME}_{timestamp}.json"
    benchmark.save_results(results, output_file)
    
    # Print detailed results
    '''
    print("\nDetailed Results:")
    for result in results['detailed_results']:
        status = "✓" if result['is_correct'] else "✗"
        print(f"{status} {result['problem_id']}: Predicted {result['predicted']} | Correct {result['correct']}")
    '''

if __name__ == "__main__":
    main()