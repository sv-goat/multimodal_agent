#!/usr/bin/env python3
"""
Generate comprehensive visualizations from experiment results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import argparse

sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def load_experiment_data(exp_dir: str) -> Dict[str, Any]:
    """Load all data from an experiment directory."""
    exp_path = Path(exp_dir)
    
    data = {}
    
    # Load metrics summary
    metrics_file = exp_path / "metrics_summary.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            data['metrics'] = json.load(f)
    
    # Load raw data files
    files = {
        'latencies': 'sample_latencies.json',
        'tool_metrics': 'tool_metrics.json',
        'usage_metrics': 'usage_metrics.json',
        'predictions': 'predictions.json',
        'ground_truth': 'ground_truth.json'
    }
    
    for key, filename in files.items():
        filepath = exp_path / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data[key] = json.load(f)
    
    return data

def plot_latency_distribution(data: Dict, output_dir: Path):
    """Plot latency distribution and statistics."""
    latencies = data.get('latencies', [])
    if not latencies:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram
    ax = axes[0, 0]
    ax.hist(latencies, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(latencies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(latencies):.2f}s')
    ax.axvline(np.median(latencies), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(latencies):.2f}s')
    ax.set_xlabel('Latency (seconds)')
    ax.set_ylabel('Frequency')
    ax.set_title('End-to-End Latency Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Box plot
    ax = axes[0, 1]
    bp = ax.boxplot(latencies, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Latency Box Plot')
    ax.grid(True, alpha=0.3)
    
    # 3. Cumulative distribution
    ax = axes[1, 0]
    sorted_latencies = np.sort(latencies)
    cumulative = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
    ax.plot(sorted_latencies, cumulative * 100, linewidth=2)
    ax.axvline(np.percentile(latencies, 95), color='red', linestyle='--', 
               label=f'P95: {np.percentile(latencies, 95):.2f}s')
    ax.axvline(np.percentile(latencies, 99), color='orange', linestyle='--', 
               label=f'P99: {np.percentile(latencies, 99):.2f}s')
    ax.set_xlabel('Latency (seconds)')
    ax.set_ylabel('Cumulative Percentage (%)')
    ax.set_title('Cumulative Latency Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Statistics summary
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
    Latency Statistics
    
    Mean:     {np.mean(latencies):.2f} s
    Median:   {np.median(latencies):.2f} s
    Min:      {np.min(latencies):.2f} s
    Max:      {np.max(latencies):.2f} s
    Std Dev:  {np.std(latencies):.2f} s
    
    Percentiles:
    P50:      {np.percentile(latencies, 50):.2f} s
    P75:      {np.percentile(latencies, 75):.2f} s
    P95:      {np.percentile(latencies, 95):.2f} s
    P99:      {np.percentile(latencies, 99):.2f} s
    
    Total Time: {np.sum(latencies)/60:.2f} minutes
    """
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_analysis.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: latency_analysis.png")

def plot_tool_usage_analysis(data: Dict, output_dir: Path):
    """Plot tool usage statistics and patterns."""
    tool_metrics = data.get('tool_metrics', [])
    if not tool_metrics:
        return
    
    # Extract data
    tool_latencies = [m.get('tool_latency', 0) for m in tool_metrics]
    tool_calls = [m.get('tool_total_count', 0) for m in tool_metrics]
    tool_success = [m.get('tool_success_count', 0) for m in tool_metrics]
    
    # Filter non-zero latencies
    non_zero_latencies = [l for l in tool_latencies if l > 0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Tool usage distribution
    ax = axes[0, 0]
    usage_counts = [0, 0, 0, 0]  # 0, 1, 2, 3+ calls
    for calls in tool_calls:
        if calls == 0:
            usage_counts[0] += 1
        elif calls == 1:
            usage_counts[1] += 1
        elif calls == 2:
            usage_counts[2] += 1
        else:
            usage_counts[3] += 1
    
    labels = ['0 calls', '1 call', '2 calls', '3+ calls']
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
    ax.bar(labels, usage_counts, color=colors, edgecolor='black', alpha=0.7)
    ax.set_ylabel('Number of Samples')
    ax.set_title('Tool Call Distribution per Sample')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(usage_counts):
        ax.text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 2. Tool latency distribution
    ax = axes[0, 1]
    if non_zero_latencies:
        ax.hist(non_zero_latencies, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(np.mean(non_zero_latencies), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(non_zero_latencies):.2f}s')
        ax.set_xlabel('Tool Latency (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title('Tool Latency Distribution (Non-zero)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Tool usage rate pie chart
    ax = axes[1, 0]
    samples_with_tools = sum(1 for calls in tool_calls if calls > 0)
    samples_without_tools = len(tool_calls) - samples_with_tools
    sizes = [samples_with_tools, samples_without_tools]
    labels_pie = [f'With Tools\n({samples_with_tools})', f'Without Tools\n({samples_without_tools})']
    colors_pie = ['#66b3ff', '#ff9999']
    ax.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%', 
           startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title('Tool Usage Rate')
    
    # 4. Tool success rate
    ax = axes[1, 1]
    total_calls = sum(tool_calls)
    total_success = sum(tool_success)
    success_rate = (total_success / total_calls * 100) if total_calls > 0 else 0
    
    categories = ['Successful', 'Failed']
    values = [total_success, total_calls - total_success]
    colors_bar = ['#4CAF50', '#f44336']
    
    bars = ax.bar(categories, values, color=colors_bar, edgecolor='black', alpha=0.7)
    ax.set_ylabel('Number of Tool Calls')
    ax.set_title(f'Tool Success Rate: {success_rate:.1f}%')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}\n({val/total_calls*100:.1f}%)' if total_calls > 0 else '0',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tool_usage_analysis.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: tool_usage_analysis.png")

def plot_token_usage_analysis(data: Dict, output_dir: Path):
    """Plot token usage statistics."""
    usage_metrics = data.get('usage_metrics', [])
    if not usage_metrics:
        return
    
    prompt_tokens = [m.get('prompt_tokens', 0) for m in usage_metrics]
    completion_tokens = [m.get('completion_tokens', 0) for m in usage_metrics]
    total_tokens = [p + c for p, c in zip(prompt_tokens, completion_tokens)]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Token distribution
    ax = axes[0, 0]
    ax.hist(total_tokens, bins=30, edgecolor='black', alpha=0.7, color='mediumpurple')
    ax.axvline(np.mean(total_tokens), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(total_tokens):.0f}')
    ax.set_xlabel('Total Tokens per Sample')
    ax.set_ylabel('Frequency')
    ax.set_title('Token Usage Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Prompt vs Completion tokens scatter
    ax = axes[0, 1]
    ax.scatter(prompt_tokens, completion_tokens, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Prompt Tokens')
    ax.set_ylabel('Completion Tokens')
    ax.set_title('Prompt vs Completion Tokens')
    ax.grid(True, alpha=0.3)
    
    # Add correlation line
    if len(prompt_tokens) > 1:
        z = np.polyfit(prompt_tokens, completion_tokens, 1)
        p = np.poly1d(z)
        ax.plot(prompt_tokens, p(prompt_tokens), "r--", alpha=0.8, linewidth=2,
               label=f'Correlation: {np.corrcoef(prompt_tokens, completion_tokens)[0,1]:.2f}')
        ax.legend()
    
    # 3. Token breakdown
    ax = axes[1, 0]
    categories = ['Prompt', 'Completion', 'Total']
    totals = [np.sum(prompt_tokens), np.sum(completion_tokens), np.sum(total_tokens)]
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    bars = ax.bar(categories, totals, color=colors, edgecolor='black', alpha=0.7)
    ax.set_ylabel('Total Tokens')
    ax.set_title('Token Usage Breakdown')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, totals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Average tokens per sample
    ax = axes[1, 1]
    avg_prompt = np.mean(prompt_tokens)
    avg_completion = np.mean(completion_tokens)
    avg_total = np.mean(total_tokens)
    
    categories = ['Avg Prompt', 'Avg Completion', 'Avg Total']
    values = [avg_prompt, avg_completion, avg_total]
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    bars = ax.bar(categories, values, color=colors, edgecolor='black', alpha=0.7)
    ax.set_ylabel('Average Tokens')
    ax.set_title('Average Token Usage per Sample')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'token_usage_analysis.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: token_usage_analysis.png")

def plot_performance_summary(data: Dict, output_dir: Path):
    """Plot overall performance summary."""
    metrics = data.get('metrics', {})
    if not metrics:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Accuracy metrics
    ax = axes[0, 0]
    acc_metrics = metrics.get('accuracy_metrics', {})
    if acc_metrics:
        categories = ['Mean ANLS', 'Accuracy\n(ANLS>0.5)']
        values = [acc_metrics.get('mean_anls', 0), acc_metrics.get('accuracy', 0)]
        colors = ['#FF6B6B', '#4ECDC4']
        bars = ax.bar(categories, values, color=colors, edgecolor='black', alpha=0.7)
        ax.set_ylabel('Score')
        ax.set_title('Accuracy Metrics')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Correct vs Incorrect
    ax = axes[0, 1]
    if acc_metrics:
        correct = acc_metrics.get('correct_predictions', 0)
        incorrect = acc_metrics.get('incorrect_predictions', 0)
        sizes = [correct, incorrect]
        labels = [f'Correct\n({correct})', f'Incorrect\n({incorrect})']
        colors = ['#4CAF50', '#f44336']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax.set_title('Prediction Accuracy')
    
    # 3. Performance metrics comparison
    ax = axes[1, 0]
    lat_metrics = metrics.get('latency_metrics', {})
    tool_metrics = metrics.get('tool_metrics', {})
    
    if lat_metrics and tool_metrics:
        categories = ['Mean\nLatency', 'Tool\nLatency', 'Inference\nTime']
        values = [
            lat_metrics.get('mean_latency_seconds', 0),
            tool_metrics.get('avg_tool_latency_seconds', 0),
            metrics.get('system_metrics', {}).get('avg_inference_time_seconds', 0)
        ]
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        bars = ax.bar(categories, values, color=colors, edgecolor='black', alpha=0.7)
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Performance Metrics')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    # 4. Resource usage
    ax = axes[1, 1]
    sys_metrics = metrics.get('system_metrics', {})
    if sys_metrics:
        categories = ['Peak Memory\n(MB)', 'Virtual Memory\n(GB)', 'CPU Time\n(seconds)']
        values = [
            sys_metrics.get('peak_memory_mb', 0),
            sys_metrics.get('virtual_memory_gb', 0),
            sys_metrics.get('total_cpu_seconds', 0)
        ]
        # Normalize for visualization (use different scales)
        normalized = [
            values[0] / 1000,  # MB to relative scale
            values[1] * 100,   # GB to relative scale
            values[2] * 10     # seconds to relative scale
        ]
        
        bars = ax.bar(categories, normalized, color=['#FF6B6B', '#4ECDC4', '#95E1D3'],
                     edgecolor='black', alpha=0.7)
        ax.set_ylabel('Normalized Value')
        ax.set_title('System Resource Usage (Normalized)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add actual value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            if i == 0:
                label = f'{val:.0f} MB'
            elif i == 1:
                label = f'{val:.2f} GB'
            else:
                label = f'{val:.1f} s'
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                    label, ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_summary.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: performance_summary.png")

def plot_latency_vs_tool_usage(data: Dict, output_dir: Path):
    """Plot relationship between latency and tool usage."""
    latencies = data.get('latencies', [])
    tool_metrics = data.get('tool_metrics', [])
    
    if not latencies or not tool_metrics or len(latencies) != len(tool_metrics):
        return
    
    tool_calls = [m.get('tool_total_count', 0) for m in tool_metrics]
    tool_latencies = [m.get('tool_latency', 0) for m in tool_metrics]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Latency vs Tool calls
    ax = axes[0]
    ax.scatter(tool_calls, latencies, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Number of Tool Calls')
    ax.set_ylabel('End-to-End Latency (seconds)')
    ax.set_title('Latency vs Tool Usage')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    if len(set(tool_calls)) > 1:
        z = np.polyfit(tool_calls, latencies, 1)
        p = np.poly1d(z)
        x_line = np.array([min(tool_calls), max(tool_calls)])
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2,
               label=f'Correlation: {np.corrcoef(tool_calls, latencies)[0,1]:.2f}')
        ax.legend()
    
    # 2. Tool latency vs Total latency
    ax = axes[1]
    ax.scatter(tool_latencies, latencies, alpha=0.6, s=60, edgecolors='black', linewidth=0.5, color='green')
    ax.set_xlabel('Tool Latency (seconds)')
    ax.set_ylabel('End-to-End Latency (seconds)')
    ax.set_title('Tool Latency vs Total Latency')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    non_zero_indices = [i for i, tl in enumerate(tool_latencies) if tl > 0]
    if len(non_zero_indices) > 1:
        tool_lat_nonzero = [tool_latencies[i] for i in non_zero_indices]
        lat_nonzero = [latencies[i] for i in non_zero_indices]
        z = np.polyfit(tool_lat_nonzero, lat_nonzero, 1)
        p = np.poly1d(z)
        x_line = np.array([min(tool_lat_nonzero), max(tool_lat_nonzero)])
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2,
               label=f'Correlation: {np.corrcoef(tool_lat_nonzero, lat_nonzero)[0,1]:.2f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_tool_correlation.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: latency_tool_correlation.png")

def plot_token_vs_latency(data: Dict, output_dir: Path):
    """Plot relationship between token usage and latency."""
    latencies = data.get('latencies', [])
    usage_metrics = data.get('usage_metrics', [])
    
    if not latencies or not usage_metrics or len(latencies) != len(usage_metrics):
        return
    
    total_tokens = [m.get('prompt_tokens', 0) + m.get('completion_tokens', 0) 
                   for m in usage_metrics]
    prompt_tokens = [m.get('prompt_tokens', 0) for m in usage_metrics]
    completion_tokens = [m.get('completion_tokens', 0) for m in usage_metrics]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Total tokens vs Latency
    ax = axes[0]
    ax.scatter(total_tokens, latencies, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Total Tokens')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Total Tokens vs Latency')
    ax.grid(True, alpha=0.3)
    
    if len(set(total_tokens)) > 1:
        z = np.polyfit(total_tokens, latencies, 1)
        p = np.poly1d(z)
        x_line = np.array([min(total_tokens), max(total_tokens)])
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2,
               label=f'Correlation: {np.corrcoef(total_tokens, latencies)[0,1]:.2f}')
        ax.legend()
    
    # 2. Prompt tokens vs Latency
    ax = axes[1]
    ax.scatter(prompt_tokens, latencies, alpha=0.6, s=60, edgecolors='black', linewidth=0.5, color='orange')
    ax.set_xlabel('Prompt Tokens')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Prompt Tokens vs Latency')
    ax.grid(True, alpha=0.3)
    
    if len(set(prompt_tokens)) > 1:
        z = np.polyfit(prompt_tokens, latencies, 1)
        p = np.poly1d(z)
        x_line = np.array([min(prompt_tokens), max(prompt_tokens)])
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2,
               label=f'Correlation: {np.corrcoef(prompt_tokens, latencies)[0,1]:.2f}')
        ax.legend()
    
    # 3. Completion tokens vs Latency
    ax = axes[2]
    ax.scatter(completion_tokens, latencies, alpha=0.6, s=60, edgecolors='black', linewidth=0.5, color='purple')
    ax.set_xlabel('Completion Tokens')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Completion Tokens vs Latency')
    ax.grid(True, alpha=0.3)
    
    if len(set(completion_tokens)) > 1:
        z = np.polyfit(completion_tokens, latencies, 1)
        p = np.poly1d(z)
        x_line = np.array([min(completion_tokens), max(completion_tokens)])
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2,
               label=f'Correlation: {np.corrcoef(completion_tokens, latencies)[0,1]:.2f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'token_latency_correlation.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: token_latency_correlation.png")

def generate_all_plots(exp_dir: str, output_dir: str = None):
    """Generate all visualization plots."""
    exp_path = Path(exp_dir)
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = exp_path / "plots"
    
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {exp_path}")
    data = load_experiment_data(exp_dir)
    
    if not data:
        print("Error: No data found in experiment directory")
        return
    
    print(f"\nGenerating visualizations...")
    print(f"Output directory: {out_path}\n")
    
    # Generate all plots
    plot_latency_distribution(data, out_path)
    plot_tool_usage_analysis(data, out_path)
    plot_token_usage_analysis(data, out_path)
    plot_performance_summary(data, out_path)
    plot_latency_vs_tool_usage(data, out_path)
    plot_token_vs_latency(data, out_path)
    
    print(f"\n✓ All visualizations saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations from experiment results")
    parser.add_argument("experiment_dir", type=str, help="Path to experiment directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory for plots (default: experiment_dir/plots)")
    
    args = parser.parse_args()
    
    generate_all_plots(args.experiment_dir, args.output)

