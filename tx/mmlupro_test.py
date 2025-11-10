import json
import time
import requests
import re
import os
import statistics
from typing import Dict, List, Tuple
from datetime import datetime

class MMLUProBenchmark:
    def __init__(self, api_url: str, model_name: str, output_dir: str = "mmlu_pro_results", api_key: str = None):
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        self.output_dir = output_dir
        self.problems = self.load_problems()
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_problems(self) -> List[Dict]:
        """Load MMLU-Pro problem dataset"""
        problems = []
        try:
            with open('/home/externals/suanfabu/x00806807/dataset/mmlu_pro.jsonl', 'r') as f:
                for line in f:
                    problem_data = json.loads(line)
                    # 适配MMLU-Pro格式
                    problem = {
                        'question_id': problem_data.get('question_id', ''),
                        'question': problem_data.get('question', ''),
                        'options': problem_data.get('options', []),
                        'answer': problem_data.get('answer', ''),
                        'answer_index': problem_data.get('answer_index', -1),
                        'category': problem_data.get('category', ''),
                        'src': problem_data.get('src', ''),
                        'cot_content': problem_data.get('cot_content', '')
                    }
                    problems.append(problem)
            print(f"Loaded {len(problems)} MMLU-Pro problems")
        except FileNotFoundError:
            print("Error: mmlu_pro.jsonl file not found")
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
            'max_tokens': 1024,
            'stream': False
        }
        
        start_time = time.time()
        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=120)
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
    
    def extract_answer(self, response: str, options_count: int) -> str:
        """Extract answer from model response for multiple choice questions"""
        # 字母选项模式 (A, B, C, D, ... I)
        letter_patterns = [
            r'[Aa]nswer:\s*([A-I])',
            r'[Tt]he correct answer is\s*([A-I])',
            r'[Ff]inal answer:\s*([A-I])',
            r'\\boxed\{([A-I])\}',
            r'[\[\(]\s*([A-I])\s*[\]\)]',
            r'\b([A-I])\b(?=\s*$)'
        ]
        
        # 数字索引模式 (0, 1, 2, ...)
        index_patterns = [
            r'[Aa]nswer:\s*(\d+)',
            r'[Tt]he correct answer is\s*(\d+)',
            r'[Ff]inal answer:\s*(\d+)',
            r'\\boxed\{(\d+)\}',
            r'[\[\(]\s*(\d+)\s*[\]\)]',
            r'\boption\s+(\d+)\b'
        ]
        
        # 首先尝试字母匹配
        for pattern in letter_patterns:
            matches = re.findall(pattern, response)
            if matches:
                answer_letter = matches[-1].upper()
                # 验证字母在有效范围内
                if 'A' <= answer_letter <= 'I':
                    return answer_letter
        
        # 然后尝试数字匹配
        for pattern in index_patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    index = int(matches[-1])
                    # 将数字索引转换为字母 (0->A, 1->B, ...)
                    if 0 <= index < options_count:
                        return chr(ord('A') + index)
                except ValueError:
                    continue
        
        # 如果没有明确匹配，尝试从文本中提取
        response_upper = response.upper()
        for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'][:options_count]:
            if f" {letter} " in f" {response_upper} " or response_upper.endswith(letter):
                return letter
        
        return None
    
    def evaluate_problem(self, problem: Dict, response: str, time_taken: float, tokens_used: int) -> Dict:
        """Evaluate reasoning result for a single problem"""
        predicted_answer = self.extract_answer(response, len(problem['options']))
        correct_answer = problem['answer']
        
        is_correct = predicted_answer == correct_answer if predicted_answer else False
        tps = tokens_used / time_taken if time_taken > 0 else 0
        
        return {
            'question_id': problem['question_id'],
            'question': problem['question'],
            'predicted': predicted_answer,
            'correct': correct_answer,
            'is_correct': is_correct,
            'time_taken': time_taken,
            'tokens_used': tokens_used,
            'tps': tps,
            'response': response,
            'category': problem['category'],
            'src': problem['src'],
            'options_count': len(problem['options'])
        }
    
    def create_prompt(self, problem: Dict) -> str:
        """Create problem prompt for MMLU-Pro format"""
        options_text = ""
        for i, option in enumerate(problem['options']):
            option_letter = chr(ord('A') + i)
            options_text += f"{option_letter}) {option}\n"
        
        return f"""Please solve the following multiple-choice question. Show your step-by-step reasoning clearly, and provide your final answer as a single letter (A, B, C, etc.).

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
            print("No problems loaded. Please check if mmlu_pro.jsonl exists.")
            return {}
        
        test_problems = self.problems[:num_problems] if num_problems else self.problems
        
        print(f"Starting MMLU-Pro benchmark with {len(test_problems)} problems...")
        print(f"Model: {self.model_name}")
        print("-" * 80)
        
        for i, problem in enumerate(test_problems, 1):
            print(f"Processing problem {i}/{len(test_problems)}: {problem['question_id']}")
            print(f"Category: {problem['category']} | Source: {problem['src']}")
            print(f"Question: {problem['question'][:80]}...")
            print(f"Options: {len(problem['options'])} options available")
            
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
            
            # Calculate accuracy by category
            category_stats = self.calculate_category_stats(results)
            
            # Calculate confusion matrix
            confusion_matrix = self.calculate_confusion_matrix(results)
            
            # Calculate options distribution
            options_stats = self.calculate_options_stats(results)
            
        else:
            accuracy = 0
            avg_tps = 0
            avg_time = 0
            category_stats = {}
            confusion_matrix = {}
            options_stats = {}
        
        summary = {
            'model': self.model_name,
            'timestamp': timestamp,
            'total_problems': len(test_problems),
            'accuracy': accuracy,
            'average_tps': avg_tps,
            'average_time_per_problem': avg_time,
            'total_tokens': total_tokens,
            'total_time': total_time,
            'category_statistics': category_stats,
            'confusion_matrix': confusion_matrix,
            'options_statistics': options_stats,
            'detailed_results': results
        }
        
        return summary
    
    def calculate_category_stats(self, results: List[Dict]) -> Dict:
        """Calculate statistics by category"""
        category_stats = {}
        
        for result in results:
            category = result['category']
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'correct': 0}
            
            category_stats[category]['total'] += 1
            if result['is_correct']:
                category_stats[category]['correct'] += 1
        
        # Calculate accuracy for each category
        for category, stats in category_stats.items():
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        return category_stats
    
    def calculate_confusion_matrix(self, results: List[Dict]) -> Dict:
        """Calculate confusion matrix for multiple choice questions"""
        confusion = {}
        all_answers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        
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
    
    def calculate_options_stats(self, results: List[Dict]) -> Dict:
        """Calculate statistics about options distribution"""
        options_stats = {
            'options_count_distribution': {},
            'accuracy_by_options_count': {}
        }
        
        # Count problems by number of options
        for result in results:
            options_count = result['options_count']
            if options_count not in options_stats['options_count_distribution']:
                options_stats['options_count_distribution'][options_count] = {'total': 0, 'correct': 0}
            
            options_stats['options_count_distribution'][options_count]['total'] += 1
            if result['is_correct']:
                options_stats['options_count_distribution'][options_count]['correct'] += 1
        
        # Calculate accuracy for each options count
        for count, stats in options_stats['options_count_distribution'].items():
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        return options_stats
    
    def save_results(self, summary: Dict, filename: str = None):
        """Save test results"""
        if filename is None:
            filename = f"{self.output_dir}/mmlu_pro_benchmark_{self.model_name.replace('/', '_')}_{summary['timestamp']}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nTest results saved to: {filename}")
        print(f"Model: {summary['model']}")
        print(f"Accuracy: {summary['accuracy']:.2%} ({sum(1 for r in summary['detailed_results'] if r['is_correct'])}/{summary['total_problems']})")
        print(f"Average TPS: {summary['average_tps']:.2f}")
        print(f"Average time per problem: {summary['average_time_per_problem']:.2f}s")
        print(f"Total tokens used: {summary['total_tokens']}")
        print(f"Total time: {summary['total_time']:.2f}s")
        
        # Print category statistics
        if summary['category_statistics']:
            print("\nAccuracy by Category:")
            for category, stats in summary['category_statistics'].items():
                print(f"  {category}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        
        # Print options statistics
        if summary['options_statistics']:
            print("\nAccuracy by Number of Options:")
            for count, stats in summary['options_statistics']['options_count_distribution'].items():
                print(f"  {count} options: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    def print_confusion_matrix(self, confusion_matrix: Dict):
        """Print formatted confusion matrix"""
        all_answers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        
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
            print(f"    Category: {result['category']} | Source: {result['src']}")
            print(f"    Predicted: {predicted} | Correct: {result['correct']}")
            print(f"    Time: {result['time_taken']:.2f}s | TPS: {result['tps']:.2f}")
            print(f"    Question: {result['question'][:80]}...")
            print()

def main():
    # Configuration for vLLM service
    API_URL = "http://localhost:8099/v1/completions"  # vLLM completions endpoint
    MODEL_NAME = "DeepSeek-R1-Distill-Qwen-32B"
    API_KEY = None  # vLLM typically doesn't require API key
    
    # Create test instance
    benchmark = MMLUProBenchmark(API_URL, MODEL_NAME)
    
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
    # print(f"\nStarting MMLU-Pro benchmark with {num_problems if num_problems else 'all'} problems...")
    
    results = benchmark.run_benchmark(num_problems=None, save_responses=False)
    
    # Save and display results
    if results:
        benchmark.save_results(results)
        # benchmark.print_detailed_results(results)
        
        # Print confusion matrix if there are results
        '''
        if results['confusion_matrix']:
            print("\n" + "="*80)
            print("CONFUSION MATRIX")
            print("="*80)
            benchmark.print_confusion_matrix(results['confusion_matrix'])
        '''
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()