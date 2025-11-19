import json
import time
import requests
import re
import os
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import shutil
import glob

class LongBenchV2Benchmark:
    def __init__(self, api_url: str, model_name: str, dataset_path: str, 
                 output_dir: str = "longbench_results", api_key: str = None, 
                 max_workers: int = 5, timeout: int = 300,
                 start_index: Optional[int] = None, end_index: Optional[int] = None):
        self.api_url = api_url
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.api_key = api_key
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.timeout = timeout
        self.start_index = start_index
        self.end_index = end_index
        self.samples = self.load_dataset()
        self.results_queue = queue.Queue()
        self.lock = threading.Lock()
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_dataset(self) -> List[Dict]:
        """Load LongBenchV2 dataset from specified path with optional index range"""
        samples = []
        try:
            print(f"Loading dataset from: {self.dataset_path}")
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # 应用索引范围过滤
                    if self.start_index is not None and line_num < self.start_index:
                        continue
                    if self.end_index is not None and line_num > self.end_index:
                        break
                    
                    try:
                        sample = json.loads(line.strip())
                        # 标准化LongBenchV2数据格式
                        standardized_sample = {
                            '_id': sample.get('_id', f'sample_{line_num}'),
                            'domain': sample.get('domain', 'unknown'),
                            'sub_domain': sample.get('sub_domain', ''),
                            'difficulty': sample.get('difficulty', ''),
                            'length': sample.get('length', ''),
                            'question': sample.get('question', ''),
                            'choice_A': sample.get('choice_A', ''),
                            'choice_B': sample.get('choice_B', ''),
                            'choice_C': sample.get('choice_C', ''),
                            'choice_D': sample.get('choice_D', ''),
                            'answer': sample.get('answer', ''),
                            'context': sample.get('context', ''),
                            'line_number': line_num  # 保存原始行号
                        }
                        samples.append(standardized_sample)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue
            
            range_info = ""
            if self.start_index is not None or self.end_index is not None:
                start = self.start_index if self.start_index is not None else 1
                end = self.end_index if self.end_index is not None else "end"
                range_info = f" (lines {start} to {end})"
            
            print(f"Successfully loaded {len(samples)} samples from dataset{range_info}")
            
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {self.dataset_path}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            
        return samples
    
    def call_model(self, prompt: str) -> Tuple[str, float, int]:
        """
        Call the vLLM inference service for LongBenchV2
        Returns: (response content, time taken, number of tokens)
        """
        headers = {
            'Content-Type': 'application/json'
        }
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        # vLLM API payload format
        payload = {
            'model': self.model_name,
            'prompt': prompt,
            'temperature': 0,
            'max_tokens': 2048,
            'stream': False
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                self.api_url, 
                json=payload, 
                headers=headers
            )
            response.raise_for_status()
            result = response.json()
            end_time = time.time()
            
            # Extract response content from vLLM format
            content = result['choices'][0]['text']
            
            # Token usage
            total_tokens = result.get('usage', {}).get('total_tokens', 0)
            
            return content.strip(), end_time - start_time, total_tokens
            
        except requests.exceptions.Timeout:
            print(f"API call timeout after {self.timeout} seconds")
            return "", self.timeout, 0
        except requests.exceptions.RequestException as e:
            print(f"API call error: {e}")
            return "", 0, 0
        except Exception as e:
            print(f"Unexpected error during API call: {e}")
            return "", 0, 0
    
    def extract_answer(self, response: str) -> str:
        """Extract answer from model response for multiple choice questions"""
        if not response:
            return None
            
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
        response_upper = response.upper()
        if 'A' in response_upper[-100:]:  # 检查最后100个字符
            return 'A'
        elif 'B' in response_upper[-100:]:
            return 'B'
        elif 'C' in response_upper[-100:]:
            return 'C'
        elif 'D' in response_upper[-100:]:
            return 'D'
        
        return None
    
    def evaluate_sample(self, sample: Dict, response: str, time_taken: float, tokens_used: int) -> Dict:
        """Evaluate reasoning result for a single sample"""
        predicted_answer = self.extract_answer(response)
        correct_answer = sample['answer'].upper() if sample['answer'] else ''
        
        is_correct = predicted_answer == correct_answer if predicted_answer and correct_answer else False
        tps = tokens_used / time_taken if time_taken > 0 else 0
        
        return {
            'sample_id': sample['_id'],
            'line_number': sample.get('line_number', 0),
            'domain': sample['domain'],
            'sub_domain': sample['sub_domain'],
            'difficulty': sample['difficulty'],
            'length': sample['length'],
            'question_preview': sample['question'][:100] + "..." if len(sample['question']) > 100 else sample['question'],
            'context_length': len(sample['context']),
            'predicted': predicted_answer,
            'correct': correct_answer,
            'is_correct': is_correct,
            'time_taken': time_taken,
            'tokens_used': tokens_used,
            'tps': tps,
            'response': response,
            'options': {
                'A': sample.get('choice_A', ''),
                'B': sample.get('choice_B', ''),
                'C': sample.get('choice_C', ''),
                'D': sample.get('choice_D', '')
            }
        }
    
    def create_prompt(self, sample: Dict) -> str:
        """Create problem prompt for LongBenchV2 format"""
        # 构建选项文本
        options_text = ""
        if sample.get('choice_A'):
            options_text = f"A) {sample.get('choice_A', '')}\n"
            options_text += f"B) {sample.get('choice_B', '')}\n"
            options_text += f"C) {sample.get('choice_C', '')}\n"
            options_text += f"D) {sample.get('choice_D', '')}\n"
        
        # 构建完整提示
        prompt = f"""Please read the following context carefully and answer the question based on it.

Context:
{sample['context']}

Question: {sample['question']}

"""
        
        if options_text:
            prompt += f"Options:\n{options_text}\n"
        
        prompt += """Please provide your final answer as a single letter (A, B, C, or D).

Reasoning:"""
        
        return prompt

    def save_single_response(self, sample_id: str, response: str, timestamp: str):
        """Save individual sample response to file"""
        filename = f"{self.output_dir}/response_{sample_id}_{timestamp}.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(response)
            return filename
        except Exception as e:
            print(f"Failed to save response: {e}")
            return None

    def process_single_sample(self, sample: Dict, index: int, total: int, timestamp: str, save_responses: bool):
        """Process a single sample (worker function for threads)"""
        with self.lock:
            print(f"Processing sample {index}/{total}: {sample['_id']} (line {sample.get('line_number', 'N/A')})")
            print(f"Domain: {sample['domain']} | Difficulty: {sample['difficulty']} | Length: {sample['length']}")
            print(f"Question: {sample['question'][:80]}...")
            print(f"Context length: {len(sample['context'])} characters")
        
        prompt = self.create_prompt(sample)
        response, time_taken, tokens_used = self.call_model(prompt)
        
        # Save individual response if requested
        if save_responses and response:
            saved_file = self.save_single_response(sample['_id'], response, timestamp)
            if saved_file:
                with self.lock:
                    print(f"  Response saved: {saved_file}")
        
        result = self.evaluate_sample(sample, response, time_taken, tokens_used)
        
        with self.lock:
            status = "✓ CORRECT" if result['is_correct'] else "✗ INCORRECT"
            print(f"  Result: {status} | Time: {time_taken:.2f}s | TPS: {result['tps']:.2f}")
            if result['predicted']:
                print(f"  Predicted: {result['predicted']} | Correct: {result['correct']}")
            else:
                print(f"  No answer extracted | Correct: {result['correct']}")
            print("-" * 80)
        
        self.results_queue.put(result)
        return result
    
    def run_benchmark(self, num_samples: int = None, save_responses: bool = False) -> Dict:
        """Run the complete benchmark test using multi-threading"""
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not self.samples:
            print("No samples loaded. Please check if dataset file exists and is accessible.")
            return {}
        
        # 如果指定了num_samples，则进一步限制样本数量
        test_samples = self.samples[:num_samples] if num_samples else self.samples
        
        # 显示测试范围信息
        range_info = ""
        if self.start_index is not None or self.end_index is not None:
            start = self.start_index if self.start_index is not None else 1
            end = self.end_index if self.end_index is not None else "end"
            range_info = f" (lines {start} to {end})"
        
        print(f"Starting LongBenchV2 benchmark with {len(test_samples)} samples{range_info}...")
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset_path}")
        print(f"Concurrent workers: {self.max_workers}")
        print(f"Timeout: {self.timeout}s")
        print("-" * 80)
        
        # 记录基准测试开始时间
        benchmark_start_time = time.time()
        
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(self.process_single_sample, sample, i+1, len(test_samples), timestamp, save_responses): sample
                for i, sample in enumerate(test_samples)
            }
            
            # Collect results as they complete
            completed_count = 0
            for future in as_completed(future_to_sample):
                try:
                    future.result()  # We don't need the return value since we're using queue
                    completed_count += 1
                except Exception as exc:
                    sample = future_to_sample[future]
                    print(f'Sample {sample["_id"]} generated an exception: {exc}')
                    completed_count += 1
        
        # 记录基准测试结束时间
        benchmark_end_time = time.time()
        benchmark_total_time = benchmark_end_time - benchmark_start_time
        
        # Collect all results from queue
        while not self.results_queue.empty():
            results.append(self.results_queue.get())
        
        # Sort results by line_number for consistent ordering
        results.sort(key=lambda x: x.get('line_number', 0))
        
        # Calculate statistics
        if results:
            accuracy = sum(1 for r in results if r['is_correct']) / len(results)
            tps_values = [r['tps'] for r in results if r['tps'] > 0]
            avg_tps = statistics.mean(tps_values) if tps_values else 0
            avg_time = statistics.mean([r['time_taken'] for r in results])
            total_tokens = sum([r['tokens_used'] for r in results])
            total_time = sum([r['time_taken'] for r in results])
            
            # Calculate per-domain statistics
            domain_stats = self.calculate_domain_statistics(results)
            # Calculate confusion matrix
            confusion_matrix = self.calculate_confusion_matrix(results)
            
        else:
            accuracy = 0
            avg_tps = 0
            avg_time = 0
            total_tokens = 0
            total_time = 0
            domain_stats = {}
            confusion_matrix = {}
        
        summary = {
            'model': self.model_name,
            'timestamp': timestamp,
            'dataset_path': self.dataset_path,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'total_samples': len(test_samples),
            'successful_samples': len(results),
            'concurrent_workers': self.max_workers,
            'timeout': self.timeout,
            'accuracy': accuracy,
            'average_tps': avg_tps,
            'average_time_per_sample': avg_time,
            'total_tokens': total_tokens,
            'total_time': total_time,
            'benchmark_execution_time': benchmark_total_time,
            'domain_statistics': domain_stats,
            'confusion_matrix': confusion_matrix,
            'detailed_results': results
        }
        
        return summary
    
    def calculate_domain_statistics(self, results: List[Dict]) -> Dict:
        """Calculate statistics per domain"""
        domain_stats = {}
        
        for result in results:
            domain = result['domain']
            if domain not in domain_stats:
                domain_stats[domain] = {
                    'total': 0,
                    'correct': 0,
                    'accuracy': 0.0,
                    'avg_time': 0.0,
                    'avg_tps': 0.0
                }
            
            domain_stats[domain]['total'] += 1
            if result['is_correct']:
                domain_stats[domain]['correct'] += 1
        
        # Calculate averages
        for domain in domain_stats:
            domain_data = [r for r in results if r['domain'] == domain]
            if domain_data:
                domain_stats[domain]['accuracy'] = domain_stats[domain]['correct'] / domain_stats[domain]['total']
                domain_stats[domain]['avg_time'] = statistics.mean([r['time_taken'] for r in domain_data])
                tps_values = [r['tps'] for r in domain_data if r['tps'] > 0]
                domain_stats[domain]['avg_tps'] = statistics.mean(tps_values) if tps_values else 0
        
        return domain_stats
    
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
            # 在文件名中包含索引范围信息
            range_suffix = ""
            if self.start_index is not None or self.end_index is not None:
                start = self.start_index if self.start_index is not None else 1
                end = self.end_index if self.end_index is not None else "end"
                range_suffix = f"_lines_{start}_to_{end}"
            
            filename = f"{self.output_dir}/longbenchv2_benchmark_{self.model_name.replace('/', '_')}_{summary['timestamp']}{range_suffix}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"\nTest results saved to: {filename}")
            self.print_summary(summary)
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def print_summary(self, summary: Dict):
        """Print test summary"""
        print(f"\n{'='*80}")
        print("LONGBENCH V2 BENCHMARK RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Model: {summary['model']}")
        print(f"Dataset: {summary['dataset_path']}")
        
        # 显示测试范围
        if self.start_index is not None or self.end_index is not None:
            start = self.start_index if self.start_index is not None else 1
            end = self.end_index if self.end_index is not None else "end"
            print(f"Test Range: lines {start} to {end}")
        
        print(f"Concurrent Workers: {summary['concurrent_workers']}")
        print(f"Timeout: {summary['timeout']}s")
        print(f"Total Samples: {summary['total_samples']}")
        print(f"Successful Samples: {summary['successful_samples']}")
        print(f"Accuracy: {summary['accuracy']:.2%} ({sum(1 for r in summary['detailed_results'] if r['is_correct'])}/{summary['successful_samples']})")
        print(f"Average TPS: {summary['average_tps']:.2f}")
        print(f"Average time per sample: {summary['average_time_per_sample']:.2f}s")
        print(f"Total tokens used: {summary['total_tokens']}")
        print(f"Total API time (sum of all requests): {summary['total_time']:.2f}s")
        print(f"Benchmark execution time (wall clock time): {summary['benchmark_execution_time']:.2f}s")
        print(f"Concurrency efficiency: {summary['total_time'] / summary['benchmark_execution_time']:.2f}x")
        
        # Print domain statistics
        if summary['domain_statistics']:
            print(f"\n{'='*50}")
            print("DOMAIN STATISTICS")
            print(f"{'='*50}")
            for domain, stats in summary['domain_statistics'].items():
                print(f"  {domain:20} {stats['accuracy']:>6.2%} accuracy ({stats['correct']:>3}/{stats['total']:>3})")
        
        # Print confusion matrix
        if summary['confusion_matrix']:
            print(f"\n{'='*50}")
            print("CONFUSION MATRIX")
            print(f"{'='*50}")
            self.print_confusion_matrix(summary['confusion_matrix'])
    
    def print_confusion_matrix(self, confusion_matrix: Dict):
        """Print formatted confusion matrix"""
        all_answers = ['A', 'B', 'C', 'D']
        
        # Header
        print("True\\Pred", end="")
        for pred in all_answers + ['None']:
            print(f"{pred:>8}", end="")
        print()
        print("-" * 50)
        
        # Rows
        for true in all_answers:
            print(f"{true:>10}", end="")
            for pred in all_answers + ['None']:
                count = confusion_matrix[true].get(pred, 0)
                print(f"{count:>8}", end="")
            print()


def main():
    # Configuration
    API_URL = "http://localhost:8099/v1/completions"  # vLLM completions endpoint
    MODEL_NAME = "DeepSeek-R1-Distill-Qwen-32B"  # 替换为您的模型名称
    DATASET_PATH = "/home/externals/suanfabu/x00806807/dataset_v2/longbenchV2.jsonl"  # 替换为您的数据集路径
    API_KEY = None  # vLLM typically doesn't require API key
    
    # Configure number of concurrent workers
    MAX_WORKERS = 20  # 根据您的硬件调整
    TIMEOUT = 600  # LongBenchV2可能需要更长的超时时间
    
    # 新增：指定测试范围 (设置为None测试所有样本)
    START_INDEX = 0    # 起始行号 (包含)
    END_INDEX = 99     # 结束行号 (包含)
    
    # Create test instance with index range
    benchmark = LongBenchV2Benchmark(
        api_url=API_URL,
        model_name=MODEL_NAME,
        dataset_path=DATASET_PATH,
        max_workers=MAX_WORKERS,
        timeout=TIMEOUT,
        start_index=START_INDEX,
        end_index=END_INDEX
    )
    
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
    
    # 记录整体开始时间
    overall_start_time = time.time()
    
    # Run test - 注意：num_samples参数现在是在索引范围内的进一步限制
    results = benchmark.run_benchmark(
        num_samples=None,      # 设置为None测试索引范围内的所有样本，或指定数量
        save_responses=False   # 设置为True会保存每个样本的响应到单独文件
    )
    
    # 记录整体结束时间
    overall_end_time = time.time()
    overall_total_time = overall_end_time - overall_start_time
    
    # 将整体执行时间也添加到结果中
    if results:
        results['overall_execution_time'] = overall_total_time
    
    # Save and display results
    if results:
        benchmark.save_results(results)
        
        # 打印额外的时间统计信息
        print(f"\nAdditional Time Statistics:")
        print(f"Overall execution time (including saving): {overall_total_time:.2f}s")
        if 'benchmark_execution_time' in results:
            print(f"Time spent on saving results: {overall_total_time - results['benchmark_execution_time']:.2f}s")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()