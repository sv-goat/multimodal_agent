#!/usr/bin/env python3
"""
Extract and analyze metrics from DocVQA experiment results.
"""

import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Any
import sys

sys.path.insert(0, str(Path(__file__).parent))
from docvqa_eval import evaluate_docvqa


def parse_prometheus_metrics(log_file: str) -> Dict[str, Any]:
    """Parse Prometheus metrics from the log file."""
    metrics = {}
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract memory metrics
    memory_match = re.search(r'process_resident_memory_bytes\s+(\d+\.?\d*e?[+-]?\d*)', content)
    if memory_match:
        memory_bytes = float(memory_match.group(1))
        metrics['peak_memory_mb'] = memory_bytes / (1024 * 1024)
        metrics['peak_memory_gb'] = memory_bytes / (1024 * 1024 * 1024)
    
    virtual_memory_match = re.search(r'process_virtual_memory_bytes\s+(\d+\.?\d*e?[+-]?\d*)', content)
    if virtual_memory_match:
        virtual_bytes = float(virtual_memory_match.group(1))
        metrics['virtual_memory_gb'] = virtual_bytes / (1024 * 1024 * 1024)
    
    # Extract CPU time
    cpu_match = re.search(r'process_cpu_seconds_total\s+(\d+\.?\d*)', content)
    if cpu_match:
        metrics['total_cpu_seconds'] = float(cpu_match.group(1))
    
    # Extract request counts
    request_count_match = re.search(r'http_requests_total\{[^}]*status="2xx"[^}]*\}\s+(\d+\.?\d*)', content)
    if request_count_match:
        metrics['total_requests'] = int(float(request_count_match.group(1)))
    
    # Extract inference time statistics
    inference_sum_match = re.search(r'vllm:request_inference_time_seconds_sum\{[^}]*\}\s+(\d+\.?\d*)', content)
    inference_count_match = re.search(r'vllm:request_inference_time_seconds_count\{[^}]*\}\s+(\d+\.?\d*)', content)
    if inference_sum_match and inference_count_match:
        total_inference_time = float(inference_sum_match.group(1))
        inference_count = float(inference_count_match.group(1))
        metrics['avg_inference_time_seconds'] = total_inference_time / inference_count if inference_count > 0 else 0
        metrics['total_inference_time_seconds'] = total_inference_time
    
    # Extract decode time statistics
    decode_sum_match = re.search(r'vllm:request_decode_time_seconds_sum\{[^}]*\}\s+(\d+\.?\d*)', content)
    decode_count_match = re.search(r'vllm:request_decode_time_seconds_count\{[^}]*\}\s+(\d+\.?\d*)', content)
    if decode_sum_match and decode_count_match:
        total_decode_time = float(decode_sum_match.group(1))
        decode_count = float(decode_count_match.group(1))
        metrics['avg_decode_time_seconds'] = total_decode_time / decode_count if decode_count > 0 else 0
    
    # Extract prefill time statistics
    prefill_sum_match = re.search(r'vllm:request_prefill_time_seconds_sum\{[^}]*\}\s+(\d+\.?\d*)', content)
    prefill_count_match = re.search(r'vllm:request_prefill_time_seconds_count\{[^}]*\}\s+(\d+\.?\d*)', content)
    if prefill_sum_match and prefill_count_match:
        total_prefill_time = float(prefill_sum_match.group(1))
        prefill_count = float(prefill_count_match.group(1))
        metrics['avg_prefill_time_seconds'] = total_prefill_time / prefill_count if prefill_count > 0 else 0
    
    # Extract HTTP request duration
    http_duration_sum_match = re.search(r'http_request_duration_highr_seconds_sum\s+(\d+\.?\d*)', content)
    http_duration_count_match = re.search(r'http_request_duration_highr_seconds_count\s+(\d+\.?\d*)', content)
    if http_duration_sum_match and http_duration_count_match:
        total_http_time = float(http_duration_sum_match.group(1))
        http_count = float(http_duration_count_match.group(1))
        metrics['avg_http_request_duration_seconds'] = total_http_time / http_count if http_count > 0 else 0
    
    return metrics


def analyze_latencies(latencies: List[float]) -> Dict[str, Any]:
    """Analyze latency statistics."""
    if not latencies:
        return {}
    
    return {
        'mean_latency_seconds': statistics.mean(latencies),
        'median_latency_seconds': statistics.median(latencies),
        'min_latency_seconds': min(latencies),
        'max_latency_seconds': max(latencies),
        'std_latency_seconds': statistics.stdev(latencies) if len(latencies) > 1 else 0,
        'p95_latency_seconds': sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        'p99_latency_seconds': sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
        'total_samples': len(latencies),
        'total_time_seconds': sum(latencies)
    }


def analyze_tool_metrics(tool_metrics: List[Dict]) -> Dict[str, Any]:
    """Analyze tool usage metrics."""
    if not tool_metrics:
        return {}
    
    total_tool_calls = sum(m.get('tool_total_count', 0) for m in tool_metrics)
    successful_tool_calls = sum(m.get('tool_success_count', 0) for m in tool_metrics)
    tool_latencies = [m.get('tool_latency', 0) for m in tool_metrics if m.get('tool_latency', 0) > 0]
    samples_with_tools = sum(1 for m in tool_metrics if m.get('tool_total_count', 0) > 0)
    
    return {
        'total_tool_calls': total_tool_calls,
        'successful_tool_calls': successful_tool_calls,
        'tool_success_rate': successful_tool_calls / total_tool_calls if total_tool_calls > 0 else 0,
        'avg_tool_calls_per_sample': total_tool_calls / len(tool_metrics) if tool_metrics else 0,
        'samples_with_tool_usage': samples_with_tools,
        'samples_without_tool_usage': len(tool_metrics) - samples_with_tools,
        'tool_usage_rate': samples_with_tools / len(tool_metrics) if tool_metrics else 0,
        'avg_tool_latency_seconds': statistics.mean(tool_latencies) if tool_latencies else 0,
        'total_tool_latency_seconds': sum(tool_latencies),
        'max_tool_latency_seconds': max(tool_latencies) if tool_latencies else 0
    }


def analyze_token_usage(usage_metrics: List[Dict]) -> Dict[str, Any]:
    """Analyze token usage metrics."""
    if not usage_metrics:
        return {}
    
    prompt_tokens = [m.get('prompt_tokens', 0) for m in usage_metrics]
    completion_tokens = [m.get('completion_tokens', 0) for m in usage_metrics]
    
    return {
        'total_prompt_tokens': sum(prompt_tokens),
        'total_completion_tokens': sum(completion_tokens),
        'total_tokens': sum(prompt_tokens) + sum(completion_tokens),
        'avg_prompt_tokens': statistics.mean(prompt_tokens) if prompt_tokens else 0,
        'avg_completion_tokens': statistics.mean(completion_tokens) if completion_tokens else 0,
        'avg_total_tokens_per_sample': statistics.mean([p + c for p, c in zip(prompt_tokens, completion_tokens)]) if prompt_tokens else 0,
        'max_prompt_tokens': max(prompt_tokens) if prompt_tokens else 0,
        'max_completion_tokens': max(completion_tokens) if completion_tokens else 0,
        'min_prompt_tokens': min(prompt_tokens) if prompt_tokens else 0,
        'min_completion_tokens': min(completion_tokens) if completion_tokens else 0
    }


def extract_all_metrics(experiment_dir: str) -> Dict[str, Any]:
    """Extract all metrics from an experiment directory."""
    exp_path = Path(experiment_dir)
    
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    results = {
        'experiment_directory': str(exp_path),
        'experiment_name': exp_path.name
    }
    
    # 1. Evaluate accuracy using DocVQA evaluation
    predictions_file = exp_path / "predictions.json"
    ground_truth_file = exp_path / "ground_truth.json"
    
    if predictions_file.exists() and ground_truth_file.exists():
        print("Evaluating DocVQA accuracy...")
        eval_results = evaluate_docvqa(
            str(predictions_file),
            str(ground_truth_file),
            threshold=0.5,
            prediction_key="answer",
            apply_extraction=True
        )
        results['accuracy_metrics'] = {
            'mean_anls': eval_results['mean_anls'],
            'accuracy': eval_results['accuracy'],
            'accuracy_percentage': eval_results['accuracy'] * 100,
            'threshold': eval_results['threshold'],
            'n_questions': eval_results['n_questions'],
            'correct_predictions': int(eval_results['accuracy'] * eval_results['n_questions']),
            'incorrect_predictions': int((1 - eval_results['accuracy']) * eval_results['n_questions'])
        }
    
    # 2. Analyze sample latencies
    latencies_file = exp_path / "sample_latencies.json"
    if latencies_file.exists():
        print("Analyzing sample latencies...")
        with open(latencies_file, 'r') as f:
            latencies = json.load(f)
        results['latency_metrics'] = analyze_latencies(latencies)
    
    # 3. Analyze tool metrics
    tool_metrics_file = exp_path / "tool_metrics.json"
    if tool_metrics_file.exists():
        print("Analyzing tool usage...")
        with open(tool_metrics_file, 'r') as f:
            tool_metrics = json.load(f)
        results['tool_metrics'] = analyze_tool_metrics(tool_metrics)
    
    # 4. Analyze token usage
    usage_metrics_file = exp_path / "usage_metrics.json"
    if usage_metrics_file.exists():
        print("Analyzing token usage...")
        with open(usage_metrics_file, 'r') as f:
            usage_metrics = json.load(f)
        results['token_usage'] = analyze_token_usage(usage_metrics)
    
    # 5. Parse Prometheus metrics log
    metrics_log_files = list(exp_path.glob("metrics_*.log"))
    if metrics_log_files:
        print("Parsing Prometheus metrics...")
        # Use the most recent log file
        latest_log = max(metrics_log_files, key=lambda p: p.stat().st_mtime)
        prometheus_metrics = parse_prometheus_metrics(str(latest_log))
        results['system_metrics'] = prometheus_metrics
        results['metrics_log_file'] = str(latest_log)
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print a formatted summary of all metrics."""
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(f"\nExperiment: {results.get('experiment_name', 'Unknown')}")
    print(f"Directory: {results.get('experiment_directory', 'Unknown')}")
    
    # Accuracy metrics
    if 'accuracy_metrics' in results:
        acc = results['accuracy_metrics']
        print("\n" + "-"*80)
        print("ACCURACY METRICS")
        print("-"*80)
        print(f"  Mean ANLS Score:        {acc['mean_anls']:.4f}")
        print(f"  Accuracy (ANLS > 0.5):  {acc['accuracy_percentage']:.2f}%")
        print(f"  Correct Predictions:    {acc['correct_predictions']}/{acc['n_questions']}")
        print(f"  Incorrect Predictions:  {acc['incorrect_predictions']}/{acc['n_questions']}")
        print(f"  Total Questions:        {acc['n_questions']}")
    
    # Latency metrics
    if 'latency_metrics' in results:
        lat = results['latency_metrics']
        print("\n" + "-"*80)
        print("LATENCY METRICS (End-to-End Sample Processing)")
        print("-"*80)
        print(f"  Mean Latency:            {lat['mean_latency_seconds']:.2f} seconds")
        print(f"  Median Latency:         {lat['median_latency_seconds']:.2f} seconds")
        print(f"  Min Latency:             {lat['min_latency_seconds']:.2f} seconds")
        print(f"  Max Latency:             {lat['max_latency_seconds']:.2f} seconds")
        print(f"  P95 Latency:             {lat['p95_latency_seconds']:.2f} seconds")
        print(f"  P99 Latency:             {lat['p99_latency_seconds']:.2f} seconds")
        print(f"  Std Deviation:           {lat['std_latency_seconds']:.2f} seconds")
        print(f"  Total Samples:           {lat['total_samples']}")
        print(f"  Total Time:              {lat['total_time_seconds']:.2f} seconds ({lat['total_time_seconds']/60:.2f} minutes)")
    
    # Tool metrics
    if 'tool_metrics' in results:
        tool = results['tool_metrics']
        print("\n" + "-"*80)
        print("TOOL USAGE METRICS")
        print("-"*80)
        print(f"  Total Tool Calls:        {tool['total_tool_calls']}")
        print(f"  Successful Tool Calls:   {tool['successful_tool_calls']}")
        print(f"  Tool Success Rate:       {tool['tool_success_rate']*100:.2f}%")
        print(f"  Avg Tool Calls/Sample:   {tool['avg_tool_calls_per_sample']:.2f}")
        print(f"  Samples Using Tools:     {tool['samples_with_tool_usage']}")
        print(f"  Samples Without Tools:   {tool['samples_without_tool_usage']}")
        print(f"  Tool Usage Rate:         {tool['tool_usage_rate']*100:.2f}%")
        print(f"  Avg Tool Latency:        {tool['avg_tool_latency_seconds']:.3f} seconds")
        print(f"  Max Tool Latency:        {tool['max_tool_latency_seconds']:.3f} seconds")
        print(f"  Total Tool Latency:      {tool['total_tool_latency_seconds']:.2f} seconds")
    
    # Token usage
    if 'token_usage' in results:
        tokens = results['token_usage']
        print("\n" + "-"*80)
        print("TOKEN USAGE METRICS")
        print("-"*80)
        print(f"  Total Prompt Tokens:      {tokens['total_prompt_tokens']:,}")
        print(f"  Total Completion Tokens:  {tokens['total_completion_tokens']:,}")
        print(f"  Total Tokens:             {tokens['total_tokens']:,}")
        print(f"  Avg Prompt Tokens:        {tokens['avg_prompt_tokens']:.1f}")
        print(f"  Avg Completion Tokens:    {tokens['avg_completion_tokens']:.1f}")
        print(f"  Avg Total Tokens/Sample:  {tokens['avg_total_tokens_per_sample']:.1f}")
        print(f"  Max Prompt Tokens:        {tokens['max_prompt_tokens']}")
        print(f"  Max Completion Tokens:    {tokens['max_completion_tokens']}")
    
    # System metrics
    if 'system_metrics' in results:
        sys_metrics = results['system_metrics']
        print("\n" + "-"*80)
        print("SYSTEM METRICS (from Prometheus)")
        print("-"*80)
        if 'peak_memory_gb' in sys_metrics:
            print(f"  Peak Memory (Resident):   {sys_metrics['peak_memory_gb']:.2f} GB ({sys_metrics.get('peak_memory_mb', 0):.0f} MB)")
        if 'virtual_memory_gb' in sys_metrics:
            print(f"  Virtual Memory:           {sys_metrics['virtual_memory_gb']:.2f} GB")
        if 'total_cpu_seconds' in sys_metrics:
            print(f"  Total CPU Time:           {sys_metrics['total_cpu_seconds']:.2f} seconds ({sys_metrics['total_cpu_seconds']/60:.2f} minutes)")
        if 'total_requests' in sys_metrics:
            print(f"  Total HTTP Requests:      {sys_metrics['total_requests']}")
        if 'avg_inference_time_seconds' in sys_metrics:
            print(f"  Avg Inference Time:       {sys_metrics['avg_inference_time_seconds']:.3f} seconds")
        if 'avg_decode_time_seconds' in sys_metrics:
            print(f"  Avg Decode Time:          {sys_metrics['avg_decode_time_seconds']:.3f} seconds")
        if 'avg_prefill_time_seconds' in sys_metrics:
            print(f"  Avg Prefill Time:         {sys_metrics['avg_prefill_time_seconds']:.3f} seconds")
        if 'avg_http_request_duration_seconds' in sys_metrics:
            print(f"  Avg HTTP Request Duration: {sys_metrics['avg_http_request_duration_seconds']:.3f} seconds")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract and analyze metrics from DocVQA experiment results")
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment directory containing results files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    try:
        results = extract_all_metrics(args.experiment_dir)
        print_summary(results)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed results saved to: {args.output}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

