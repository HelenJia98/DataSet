import json
import time
import requests
import re
import os
import statistics
import subprocess
import tempfile
from typing import Dict, List, Tuple
from datetime import datetime

class HumanEvalBenchmark:
    def __init__(self, api_url: str, model_name: str, output_dir: str = "humaneval_results", api_key: str = None):
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        self.output_dir = output_dir
        self.problems = self.load_problems()
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_problems(self) -> List[Dict]:
        """Load HumanEval problem dataset"""
        problems = []
        try:
            with open('/home/externals/suanfabu/x00806807/dataset/humaneval.jsonl', 'r') as f:
                for line in f:
                    problem_data = json.loads(line)
                    # 适配HumanEval格式
                    problem = {
                        'task_id': problem_data.get('task_id', ''),
                        'prompt': problem_data.get('prompt', ''),
                        'canonical_solution': problem_data.get('canonical_solution', ''),
                        'test': problem_data.get('test', ''),
                        'entry_point': problem_data.get('entry_point', ''),
                        'function_name': self.extract_function_name(problem_data.get('prompt', ''))
                    }
                    problems.append(problem)
            print(f"Loaded {len(problems)} HumanEval problems")
        except FileNotFoundError:
            print("Error: humaneval.jsonl file not found")
        except Exception as e:
            print(f"Error loading problems: {e}")
            
        return problems
    
    def extract_function_name(self, prompt: str) -> str:
        """Extract function name from prompt"""
        match = re.search(r'def\s+(\w+)', prompt)
        return match.group(1) if match else "unknown_function"
    
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
            'max_tokens': 20000,  # 代码生成通常需要更多tokens
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
    
    def extract_code(self, response: str, function_name: str) -> str:
        """Extract code from model response"""
        # 尝试提取代码块
        code_patterns = [
            r'```python\s*(.*?)\s*```',  # Markdown代码块
            r'```\s*(.*?)\s*```',        # 通用代码块
            r'def\s+' + function_name + r'.*?(?=\n\s*def|\n\s*$|\Z)',  # 函数定义
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                code = matches[0].strip()
                if 'def ' + function_name in code:
                    return code
        
        # 如果没有找到代码块，尝试从响应中提取函数定义
        lines = response.split('\n')
        code_lines = []
        in_function = False
        indent_level = None
        
        for line in lines:
            if line.strip().startswith('def ' + function_name):
                in_function = True
                if indent_level is None:
                    # 检测缩进级别
                    indent_level = len(line) - len(line.lstrip())
                code_lines.append(line)
            elif in_function:
                if line.strip() and not line.startswith(' ' * (indent_level if indent_level else 4)):
                    # 遇到不同缩进级别的非空行，可能函数结束
                    break
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        return response
    
    def execute_test(self, problem: Dict, generated_code: str) -> Tuple[bool, str]:
        """Execute the test cases on generated code"""
        # 创建完整的Python代码
        full_code = problem['prompt'] + generated_code + '\n\n' + problem['test']
        
        # 添加测试运行代码
        test_runner = f"""
if __name__ == "__main__":
    try:
        check({problem['entry_point']})
        print("ALL_TESTS_PASSED")
    except AssertionError as e:
        print(f"TEST_FAILED: {{e}}")
    except Exception as e:
        print(f"ERROR: {{e}}")
"""
        full_code += test_runner
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_file = f.name
        
        try:
            # 执行代码
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=10  # 10秒超时
            )
            
            # 清理临时文件
            os.unlink(temp_file)
            
            if result.returncode == 0:
                if "ALL_TESTS_PASSED" in result.stdout:
                    return True, "All tests passed"
                else:
                    return False, f"Tests failed: {result.stdout}"
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                return False, f"Execution error: {error_msg}"
                
        except subprocess.TimeoutExpired:
            os.unlink(temp_file)
            return False, "Execution timeout"
        except Exception as e:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            return False, f"Test execution failed: {e}"
    
    def evaluate_problem(self, problem: Dict, response: str, time_taken: float, tokens_used: int) -> Dict:
        """Evaluate reasoning result for a single problem"""
        generated_code = self.extract_code(response, problem['function_name'])
        tests_passed, test_output = self.execute_test(problem, generated_code)
        tps = tokens_used / time_taken if time_taken > 0 else 0
        
        return {
            'task_id': problem['task_id'],
            'function_name': problem['function_name'],
            'generated_code': generated_code,
            'tests_passed': tests_passed,
            'test_output': test_output,
            'time_taken': time_taken,
            'tokens_used': tokens_used,
            'tps': tps,
            'response': response
        }
    
    def create_prompt(self, problem: Dict) -> str:
        """Create problem prompt for code generation"""
        return f"""Please complete the following Python function. Provide only the code implementation without any explanation.

{problem['prompt']}

Please complete the function implementation:"""

    def save_single_response(self, task_id: str, response: str, generated_code: str, timestamp: str):
        """Save individual problem response to file"""
        filename = f"{self.output_dir}/response_{task_id.replace('/', '_')}_{timestamp}.py"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("# Generated code for: " + task_id + "\n")
                f.write("# Full response:\n")
                f.write('"""\n' + response + '\n"""\n\n')
                f.write("# Extracted code:\n")
                f.write(generated_code)
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
            print("No problems loaded. Please check if humaneval.jsonl exists.")
            return {}
        
        test_problems = self.problems[:num_problems] if num_problems else self.problems
        
        print(f"Starting HumanEval benchmark with {len(test_problems)} problems...")
        print(f"Model: {self.model_name}")
        print("-" * 80)
        
        for i, problem in enumerate(test_problems, 1):
            print(f"Processing problem {i}/{len(test_problems)}: {problem['task_id']}")
            print(f"Function: {problem['function_name']}")
            
            prompt = self.create_prompt(problem)
            response, time_taken, tokens_used = self.call_model(prompt)
            
            result = self.evaluate_problem(problem, response, time_taken, tokens_used)
            results.append(result)
            
            # Save individual response if requested
            if save_responses:
                saved_file = self.save_single_response(
                    problem['task_id'], response, result['generated_code'], timestamp
                )
                if saved_file:
                    print(f"  Code saved: {saved_file}")
            
            total_tokens += tokens_used
            total_time += time_taken
            
            status = "✓ PASSED" if result['tests_passed'] else "✗ FAILED"
            print(f"  Result: {status} | Time: {time_taken:.2f}s | TPS: {result['tps']:.2f}")
            print(f"  Test output: {result['test_output'][:100]}...")
            print("-" * 60)
            
            # Avoid making requests too frequently
            time.sleep(0.5)
        
        # Calculate statistics
        if results:
            pass_rate = sum(1 for r in results if r['tests_passed']) / len(results)
            tps_values = [r['tps'] for r in results if r['tps'] > 0]
            avg_tps = statistics.mean(tps_values) if tps_values else 0
            avg_time = statistics.mean([r['time_taken'] for r in results])
            
            # 计算编译错误和执行错误的比例
            execution_errors = sum(1 for r in results if not r['tests_passed'] and "Execution error" in r['test_output'])
            compilation_errors = sum(1 for r in results if not r['tests_passed'] and "ERROR:" in r['test_output'])
            test_failures = sum(1 for r in results if not r['tests_passed'] and "Tests failed" in r['test_output'])
            
        else:
            pass_rate = 0
            avg_tps = 0
            avg_time = 0
            execution_errors = 0
            compilation_errors = 0
            test_failures = 0
        
        summary = {
            'model': self.model_name,
            'timestamp': timestamp,
            'total_problems': len(test_problems),
            'pass_rate': pass_rate,
            'average_tps': avg_tps,
            'average_time_per_problem': avg_time,
            'total_tokens': total_tokens,
            'total_time': total_time,
            'error_breakdown': {
                'execution_errors': execution_errors,
                'compilation_errors': compilation_errors,
                'test_failures': test_failures
            },
            'detailed_results': results
        }
        
        return summary
    
    def save_results(self, summary: Dict, filename: str = None):
        """Save test results"""
        if filename is None:
            filename = f"{self.output_dir}/humaneval_benchmark_{self.model_name.replace('/', '_')}_{summary['timestamp']}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nTest results saved to: {filename}")
        print(f"Model: {summary['model']}")
        print(f"Pass Rate: {summary['pass_rate']:.2%} ({sum(1 for r in summary['detailed_results'] if r['tests_passed'])}/{summary['total_problems']})")
        print(f"Average TPS: {summary['average_tps']:.2f}")
        print(f"Average time per problem: {summary['average_time_per_problem']:.2f}s")
        print(f"Total tokens used: {summary['total_tokens']}")
        print(f"Total time: {summary['total_time']:.2f}s")
        
        # Print error breakdown
        if summary['error_breakdown']:
            print("\nError Breakdown:")
            errors = summary['error_breakdown']
            total_errors = sum(errors.values())
            if total_errors > 0:
                print(f"  Execution errors: {errors['execution_errors']} ({errors['execution_errors']/total_errors:.1%})")
                print(f"  Compilation errors: {errors['compilation_errors']} ({errors['compilation_errors']/total_errors:.1%})")
                print(f"  Test failures: {errors['test_failures']} ({errors['test_failures']/total_errors:.1%})")
    
    def print_detailed_results(self, summary: Dict):
        """Print detailed results for each problem"""
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)
        
        for result in summary['detailed_results']:
            status = "✓" if result['tests_passed'] else "✗"
            print(f"{status} {result['task_id']}:")
            print(f"    Function: {result['function_name']}")
            print(f"    Time: {result['time_taken']:.2f}s | TPS: {result['tps']:.2f}")
            print(f"    Test Result: {result['test_output']}")
            if not result['tests_passed']:
                print(f"    Generated Code Preview: {result['generated_code'][:100]}...")
            print()

def main():
    # Configuration for vLLM service
    API_URL = "http://localhost:8099/v1/completions"  # vLLM completions endpoint
    MODEL_NAME = "DeepSeek-R1-Distill-Qwen-32B"
    API_KEY = None  # vLLM typically doesn't require API key
    
    # Create test instance
    benchmark = HumanEvalBenchmark(API_URL, MODEL_NAME)
    
    # Test connection first
    print("Testing connection to vLLM service...")
    test_prompt = "Write a Python function that adds two numbers."
    try:
        response, time_taken, tokens = benchmark.call_model(test_prompt)
        print(f"Connection test successful. Response: {response.strip()[:100]}...")
        print(f"Test time: {time_taken:.2f}s, Tokens: {tokens}")
    except Exception as e:
        print(f"Connection test failed: {e}")
        return
    
    # Run test
    # num_problems = 2  # Set to None to test all problems
    # print(f"\nStarting HumanEval benchmark with {num_problems if num_problems else 'all'} problems...")
    
    results = benchmark.run_benchmark(num_problems=None, save_responses=False)
    
    # Save and display results
    if results:
        benchmark.save_results(results)
        # benchmark.print_detailed_results(results)
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()