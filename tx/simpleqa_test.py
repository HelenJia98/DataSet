import json
import time
import requests
import re
import os
import statistics
from typing import Dict, List, Tuple
from datetime import datetime
import ast
from collections import Counter

class SimpleQABenchmark:
    def __init__(self, api_url: str, model_name: str, output_dir: str = "simpleqa_results", api_key: str = None):
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        self.output_dir = output_dir
        self.problems = self.load_problems()
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_problems(self) -> List[Dict]:
        """Load SimpleQA problem dataset"""
        problems = []
        try:
            with open('/home/externals/suanfabu/x00806807/dataset/simple_qa.jsonl', 'r') as f:
                for line in f:
                    problem_data = json.loads(line)
                    # 解析metadata字段
                    metadata_str = problem_data.get('metadata', '{}')
                    try:
                        # 尝试解析metadata字符串为字典
                        if isinstance(metadata_str, str):
                            metadata = ast.literal_eval(metadata_str)
                        else:
                            metadata = metadata_str
                    except:
                        metadata = {}
                    
                    # 适配SimpleQA格式
                    problem = {
                        'question': problem_data.get('problem', problem_data.get('question', '')),
                        'answer': problem_data.get('answer', ''),
                        'metadata': metadata,
                        'topic': metadata.get('topic', 'Unknown'),
                        'answer_type': metadata.get('answer_type', 'Unknown'),
                        'question_id': f"simpleqa_{len(problems)+1}"
                    }
                    problems.append(problem)
            print(f"Loaded {len(problems)} SimpleQA problems")
        except FileNotFoundError:
            print("Error: simpleQA.jsonl file not found")
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
            'temperature': 0.1,
            'max_tokens': 512,
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
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        
        # 转换为小写，移除多余空格和标点
        normalized = text.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)  # 移除标点
        normalized = re.sub(r'\s+', ' ', normalized)  # 合并多个空格
        
        return normalized
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        normalized = self.normalize_text(text)
        return normalized.split()
    
    def calculate_f1(self, predicted: str, correct: str) -> float:
        """Calculate F1 score between predicted and correct answers"""
        if not predicted or not correct:
            return 0.0
        
        pred_tokens = set(self.tokenize(predicted))
        correct_tokens = set(self.tokenize(correct))
        
        if not pred_tokens or not correct_tokens:
            return 0.0
        
        # 计算交集
        common_tokens = pred_tokens & correct_tokens
        num_common = len(common_tokens)
        
        if num_common == 0:
            return 0.0
        
        # 计算precision和recall
        precision = num_common / len(pred_tokens)
        recall = num_common / len(correct_tokens)
        
        # 计算F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def calculate_exact_match(self, predicted: str, correct: str) -> bool:
        """Calculate exact match"""
        return self.normalize_text(predicted) == self.normalize_text(correct)
    
    def calculate_containment(self, predicted: str, correct: str) -> float:
        """Calculate containment score (how much of correct answer is contained in prediction)"""
        if not predicted or not correct:
            return 0.0
        
        pred_normalized = self.normalize_text(predicted)
        correct_normalized = self.normalize_text(correct)
        
        # 检查正确答案是否包含在预测答案中
        if correct_normalized in pred_normalized:
            return 1.0
        
        # 检查预测答案是否包含在正确答案中
        if pred_normalized in correct_normalized:
            return len(pred_normalized) / len(correct_normalized) if correct_normalized else 0.0
        
        return 0.0
    
    def extract_answer(self, response: str) -> str:
        """Extract answer from model response for factoid questions"""
        # 首先尝试提取明确标注的答案
        patterns = [
            r'[Aa]nswer:\s*([^\n\.]+)',
            r'[Tt]he answer is\s*([^\n\.]+)',
            r'[Ff]inal answer:\s*([^\n\.]+)',
            r'\\boxed\{([^}]+)\}',
            r'[\[\(]\s*([^\]\)]+)\s*[\]\)]',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                # 返回最后一个匹配，通常是最终答案
                answer = matches[-1].strip()
                if answer:
                    return answer
        
        # 如果没有明确匹配，尝试从最后一行提取
        lines = response.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            # 如果最后一行看起来像答案（不是太长，且包含字母）
            if len(last_line) < 100 and re.search(r'[a-zA-Z]', last_line):
                return last_line
        
        # 如果还是没找到，返回整个响应的前50个字符
        return response.strip()[:50]
    
    def evaluate_problem(self, problem: Dict, response: str, time_taken: float, tokens_used: int) -> Dict:
        """Evaluate reasoning result for a single problem using F1 score"""
        predicted_answer = self.extract_answer(response)
        correct_answer = problem['answer']
        
        # 计算多种评分指标
        f1_score = self.calculate_f1(predicted_answer, correct_answer)
        exact_match = self.calculate_exact_match(predicted_answer, correct_answer)
        containment_score = self.calculate_containment(predicted_answer, correct_answer)
        
        # 设置F1阈值来判断是否正确（通常使用0.5-0.7作为阈值）
        f1_threshold = 0.6
        is_correct_f1 = f1_score >= f1_threshold
        is_correct_exact = exact_match
        
        tps = tokens_used / time_taken if time_taken > 0 else 0
        
        return {
            'question_id': problem['question_id'],
            'question': problem['question'],
            'predicted': predicted_answer,
            'correct': correct_answer,
            'f1_score': f1_score,
            'exact_match': exact_match,
            'containment_score': containment_score,
            'is_correct_f1': is_correct_f1,
            'is_correct_exact': is_correct_exact,
            'time_taken': time_taken,
            'tokens_used': tokens_used,
            'tps': tps,
            'response': response,
            'topic': problem['topic'],
            'answer_type': problem['answer_type']
        }
    
    def create_prompt(self, problem: Dict) -> str:
        """Create problem prompt with factual emphasis"""
        return f"""Answer the following factual question based on your knowledge. Provide a precise, accurate, and concise answer. Focus on factual correctness.

Question: {problem['question']}

Provide your factual answer:"""

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
    
    def run_benchmark(self, num_problems: int = None, save_responses: bool = True, f1_threshold: float = 0.6) -> Dict:
        """Run the complete benchmark test with F1 scoring"""
        results = []
        total_tokens = 0
        total_time = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not self.problems:
            print("No problems loaded. Please check if simpleQA.jsonl exists.")
            return {}
        
        test_problems = self.problems[:num_problems] if num_problems else self.problems
        
        print(f"Starting SimpleQA benchmark with {len(test_problems)} problems...")
        print(f"Model: {self.model_name}")
        print(f"Prompt Style: Factual Emphasis")
        print(f"F1 Threshold: {f1_threshold}")
        print("-" * 80)
        
        for i, problem in enumerate(test_problems, 1):
            print(f"Processing problem {i}/{len(test_problems)}: {problem['question_id']}")
            print(f"Question: {problem['question']}")
            print(f"Topic: {problem['topic']} | Answer Type: {problem['answer_type']}")
            
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
            
            status_f1 = "✓" if result['is_correct_f1'] else "✗"
            status_exact = "✓" if result['is_correct_exact'] else "✗"
            print(f"  Result: F1 {status_f1} | Exact {status_exact} | F1 Score: {result['f1_score']:.3f}")
            print(f"  Time: {time_taken:.2f}s | TPS: {result['tps']:.2f}")
            print(f"  Predicted: {result['predicted']}")
            print(f"  Correct: {result['correct']}")
            print("-" * 60)
            
            # Avoid making requests too frequently
            time.sleep(0.5)
        
        # Calculate comprehensive statistics
        if results:
            # F1-based accuracy
            accuracy_f1 = sum(1 for r in results if r['is_correct_f1']) / len(results)
            accuracy_exact = sum(1 for r in results if r['is_correct_exact']) / len(results)
            
            # F1 score statistics
            f1_scores = [r['f1_score'] for r in results]
            avg_f1 = statistics.mean(f1_scores)
            median_f1 = statistics.median(f1_scores)
            
            # Containment statistics
            containment_scores = [r['containment_score'] for r in results]
            avg_containment = statistics.mean(containment_scores)
            
            # Performance statistics
            tps_values = [r['tps'] for r in results if r['tps'] > 0]
            avg_tps = statistics.mean(tps_values) if tps_values else 0
            avg_time = statistics.mean([r['time_taken'] for r in results])
            
            # Calculate accuracy by topic and answer type
            topic_stats_f1 = self.calculate_topic_stats(results, metric='f1')
            topic_stats_exact = self.calculate_topic_stats(results, metric='exact')
            type_stats_f1 = self.calculate_type_stats(results, metric='f1')
            type_stats_exact = self.calculate_type_stats(results, metric='exact')
            
        else:
            accuracy_f1 = accuracy_exact = avg_f1 = median_f1 = avg_containment = 0
            avg_tps = avg_time = 0
            f1_distribution = {}
            topic_stats_f1 = topic_stats_exact = type_stats_f1 = type_stats_exact = {}
        
        summary = {
            'model': self.model_name,
            'timestamp': timestamp,
            'total_problems': len(test_problems),
            'scoring_metrics': {
                'f1_threshold': f1_threshold,
                'accuracy_f1': accuracy_f1,
                'accuracy_exact': accuracy_exact,
                'average_f1': avg_f1,
                'median_f1': median_f1,
                'average_containment': avg_containment,
            },
            'performance_metrics': {
                'average_tps': avg_tps,
                'average_time_per_problem': avg_time,
                'total_tokens': total_tokens,
                'total_time': total_time
            },
            'topic_statistics_f1': topic_stats_f1,
            'topic_statistics_exact': topic_stats_exact,
            'type_statistics_f1': type_stats_f1,
            'type_statistics_exact': type_stats_exact,
            'prompt_style': 'factual_emphasis',
            'detailed_results': results
        }
        
        return summary
    
    def calculate_f1_distribution(self, f1_scores: List[float]) -> Dict:
        """Calculate F1 score distribution"""
        distribution = {
            'perfect_1.0': sum(1 for score in f1_scores if score == 1.0),
            'high_0.8_0.99': sum(1 for score in f1_scores if 0.8 <= score < 1.0),
            'medium_0.6_0.79': sum(1 for score in f1_scores if 0.6 <= score < 0.8),
            'low_0.4_0.59': sum(1 for score in f1_scores if 0.4 <= score < 0.6),
            'poor_0.2_0.39': sum(1 for score in f1_scores if 0.2 <= score < 0.4),
            'very_poor_0.0_0.19': sum(1 for score in f1_scores if score < 0.2),
        }
        
        total = len(f1_scores)
        for key in distribution:
            distribution[key + '_percent'] = distribution[key] / total if total > 0 else 0
        
        return distribution
    
    def calculate_topic_stats(self, results: List[Dict], metric: str = 'f1') -> Dict:
        """Calculate statistics by topic"""
        topic_stats = {}
        
        for result in results:
            topic = result['topic']
            if topic not in topic_stats:
                topic_stats[topic] = {'total': 0, 'correct': 0, 'f1_scores': []}
            
            topic_stats[topic]['total'] += 1
            topic_stats[topic]['f1_scores'].append(result['f1_score'])
            
            if metric == 'f1':
                if result['is_correct_f1']:
                    topic_stats[topic]['correct'] += 1
            else:  # exact
                if result['is_correct_exact']:
                    topic_stats[topic]['correct'] += 1
        
        # Calculate accuracy and average F1 for each topic
        for topic, stats in topic_stats.items():
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            stats['average_f1'] = statistics.mean(stats['f1_scores']) if stats['f1_scores'] else 0
        
        return topic_stats
    
    def calculate_type_stats(self, results: List[Dict], metric: str = 'f1') -> Dict:
        """Calculate statistics by answer type"""
        type_stats = {}
        
        for result in results:
            answer_type = result['answer_type']
            if answer_type not in type_stats:
                type_stats[answer_type] = {'total': 0, 'correct': 0, 'f1_scores': []}
            
            type_stats[answer_type]['total'] += 1
            type_stats[answer_type]['f1_scores'].append(result['f1_score'])
            
            if metric == 'f1':
                if result['is_correct_f1']:
                    type_stats[answer_type]['correct'] += 1
            else:  # exact
                if result['is_correct_exact']:
                    type_stats[answer_type]['correct'] += 1
        
        # Calculate accuracy and average F1 for each answer type
        for answer_type, stats in type_stats.items():
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            stats['average_f1'] = statistics.mean(stats['f1_scores']) if stats['f1_scores'] else 0
        
        return type_stats
    
    def save_results(self, summary: Dict, filename: str = None):
        """Save test results"""
        if filename is None:
            filename = f"{self.output_dir}/simpleqa_benchmark_{self.model_name.replace('/', '_')}_{summary['timestamp']}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nTest results saved to: {filename}")
        print(f"Model: {summary['model']}")
        print(f"Prompt Style: {summary['prompt_style']}")
        print(f"F1 Threshold: {summary['scoring_metrics']['f1_threshold']}")
        
        # Print scoring metrics
        metrics = summary['scoring_metrics']
        print(f"\nSCORING METRICS:")
        print(f"  F1-based Accuracy: {metrics['accuracy_f1']:.2%}")
        print(f"  Exact Match Accuracy: {metrics['accuracy_exact']:.2%}")
        print(f"  Average F1 Score: {metrics['average_f1']:.3f}")
        print(f"  Median F1 Score: {metrics['median_f1']:.3f}")
        print(f"  Average Containment: {metrics['average_containment']:.3f}")
        
        # Print performance metrics
        perf = summary['performance_metrics']
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Average TPS: {perf['average_tps']:.2f}")
        print(f"  Average time per problem: {perf['average_time_per_problem']:.2f}s")
        print(f"  Total tokens used: {perf['total_tokens']}")
        print(f"  Total time: {perf['total_time']:.2f}s")
        
        # Print topic statistics
        if summary['topic_statistics_f1']:
            print(f"\nACCURACY BY TOPIC (F1-based):")
            for topic, stats in sorted(summary['topic_statistics_f1'].items(), 
                                     key=lambda x: x[1]['accuracy'], reverse=True):
                print(f"  {topic}: {stats['accuracy']:.2%} (F1: {stats['average_f1']:.3f})")
    
    def print_detailed_results(self, summary: Dict):
        """Print detailed results for each problem"""
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)
        
        for result in summary['detailed_results']:
            status_f1 = "✓" if result['is_correct_f1'] else "✗"
            status_exact = "✓" if result['is_correct_exact'] else "✗"
            predicted = result['predicted'][:80] + "..." if len(result['predicted']) > 80 else result['predicted']
            
            print(f"F1 {status_f1} | Exact {status_exact} | Problem {result['question_id']}:")
            print(f"    F1 Score: {result['f1_score']:.3f} | Containment: {result['containment_score']:.3f}")
            print(f"    Question: {result['question']}")
            print(f"    Predicted: {predicted}")
            print(f"    Correct: {result['correct']}")
            print(f"    Time: {result['time_taken']:.2f}s | TPS: {result['tps']:.2f}")
            print(f"    Topic: {result['topic']} | Type: {result['answer_type']}")
            print()

def main():
    # Configuration for vLLM service
    API_URL = "http://localhost:8099/v1/completions"
    MODEL_NAME = "DeepSeek-R1-Distill-Qwen-32B"
    API_KEY = None
    
    # Create test instance
    benchmark = SimpleQABenchmark(API_URL, MODEL_NAME)
    
    # Test connection first
    print("Testing connection to vLLM service...")
    test_prompt = "What is the capital of France?"
    try:
        response, time_taken, tokens = benchmark.call_model(test_prompt)
        print(f"Connection test successful. Response: {response.strip()}")
        print(f"Test time: {time_taken:.2f}s, Tokens: {tokens}")
    except Exception as e:
        print(f"Connection test failed: {e}")
        return
    
    # Run test with F1 scoring
    # num_problems = 10
    f1_threshold = 0.6  # 可调整的F1阈值
    
    # print(f"\nStarting SimpleQA benchmark with {num_problems if num_problems else 'all'} problems...")
    # print(f"Using F1 scoring with threshold: {f1_threshold}")
    
    results = benchmark.run_benchmark(
        num_problems=None, 
        save_responses=False, 
        f1_threshold=f1_threshold
    )
    
    # Save and display results
    if results:
        benchmark.save_results(results)
        benchmark.print_detailed_results(results)
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()