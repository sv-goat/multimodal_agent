#!/usr/bin/env python3
"""
Generate CSV tables from experimental data
No markdown output, English only
"""
import json
import csv
from pathlib import Path

def load_json_data():
    """Load experimental data from JSON files"""
    all_data = []
    
    # Try to load data with success counts first, fall back to regular data
    if Path('final_report_data_with_success.json').exists():
        print("Loading data with success counts...")
        with open('final_report_data_with_success.json') as f:
            original_data = json.load(f)
        all_data.extend(original_data)
    elif Path('final_report_data.json').exists():
        print("Loading regular data (no success counts)...")
        with open('final_report_data.json') as f:
            original_data = json.load(f)
        # Add default values for missing fields
        for exp in original_data:
            exp['total_tool_success'] = exp.get('total_tool_success', 0)
            exp['tool_success_rate'] = exp.get('tool_success_rate', 0)
        all_data.extend(original_data)
    
    # Load React variants data
    if Path('react_variants_data_with_success.json').exists():
        print("Loading React variants with success counts...")
        with open('react_variants_data_with_success.json') as f:
            react_data = json.load(f)
        
        # Convert tools_config to method names
        tools_to_method = {
            'OCR': 'ReactOCR',
            'OCR+Calc': 'ReactOCR+Calc',
            'OCR+Calc+Web': 'ReactOCR+Calc+Web'
        }
        
        for exp in react_data:
            exp['mode'] = tools_to_method.get(exp['tools_config'], exp['tools_config'])
            all_data.append(exp)
    elif Path('react_variants_data.json').exists():
        print("Loading React variants (no success counts)...")
        with open('react_variants_data.json') as f:
            react_data = json.load(f)
        
        # Convert tools_config to method names
        tools_to_method = {
            'OCR': 'ReactOCR',
            'OCR+Calc': 'ReactOCR+Calc',
            'OCR+Calc+Web': 'ReactOCR+Calc+Web'
        }
        
        for exp in react_data:
            exp['mode'] = tools_to_method.get(exp['tools_config'], exp['tools_config'])
            # Add default values for missing fields
            exp['total_tool_success'] = exp.get('total_tool_success', 0)
            exp['tool_success_rate'] = exp.get('tool_success_rate', 0)
            all_data.append(exp)
    
    return all_data

def generate_detailed_csv(all_data):
    """Generate experimental_results_detailed.csv"""
    rows = []
    
    for exp in sorted(all_data, key=lambda x: (x['mode'], x['model_size'], x['shots'])):
        row = {
            'Method': exp['mode'].upper() if exp['mode'] in ['direct', 'cot', 'react'] else exp['mode'],
            'Model': f"Qwen3-VL-{exp['model_size']}",
            'Shots': exp['shots'],
            'ANLS': f"{exp['mean_anls']:.4f}",
            'Accuracy_%': f"{exp['accuracy']*100:.1f}",
            'Tool_Usage_%': f"{exp['tool_usage_rate']*100:.1f}",
            'Avg_Tool_Calls': f"{exp['avg_tool_calls']:.2f}",
            'Mean_Latency_s': f"{exp['mean_latency']:.2f}",
            'Median_Latency_s': f"{exp['median_latency']:.2f}",
            'Avg_Tokens': f"{exp['avg_tokens']:.0f}",
            'Total_Tool_Calls': exp['total_tool_calls'],
            'Total_Tool_Success': exp.get('total_tool_success', 0),
            'Tool_Success_Rate_%': f"{exp.get('tool_success_rate', 0)*100:.1f}"
        }
        rows.append(row)
    
    # Write CSV
    fieldnames = ['Method', 'Model', 'Shots', 'ANLS', 'Accuracy_%', 
                  'Tool_Usage_%', 'Avg_Tool_Calls', 'Mean_Latency_s', 
                  'Median_Latency_s', 'Avg_Tokens', 'Total_Tool_Calls',
                  'Total_Tool_Success', 'Tool_Success_Rate_%']
    
    with open('experimental_results_detailed_new.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Generated: experimental_results_detailed.csv ({len(rows)} rows)")
    return rows

def generate_method_summary(all_data):
    """Generate method summary by averaging across models and shots"""
    from collections import defaultdict
    
    method_stats = defaultdict(lambda: {
        'anls': [], 'accuracy': [], 'tokens': [], 'tool_calls': [],
        'tool_usage': [], 'latency': []
    })
    
    for exp in all_data:
        method = exp['mode'].upper() if exp['mode'] in ['direct', 'cot', 'react'] else exp['mode']
        method_stats[method]['anls'].append(exp['mean_anls'])
        method_stats[method]['accuracy'].append(exp['accuracy'] * 100)
        method_stats[method]['tokens'].append(exp['avg_tokens'])
        method_stats[method]['tool_calls'].append(exp['total_tool_calls'])
        method_stats[method]['tool_usage'].append(exp['tool_usage_rate'] * 100)
        method_stats[method]['latency'].append(exp['mean_latency'])
    
    rows = []
    for method in sorted(method_stats.keys()):
        stats = method_stats[method]
        rows.append({
            'Method': method,
            'Avg_ANLS': f"{sum(stats['anls'])/len(stats['anls']):.4f}",
            'Avg_Accuracy_%': f"{sum(stats['accuracy'])/len(stats['accuracy']):.1f}",
            'Avg_Tokens': f"{sum(stats['tokens'])/len(stats['tokens']):.0f}",
            'Total_Tool_Calls': sum(stats['tool_calls']),
            'Avg_Tool_Usage_%': f"{sum(stats['tool_usage'])/len(stats['tool_usage']):.1f}",
            'Avg_Latency_s': f"{sum(stats['latency'])/len(stats['latency']):.2f}"
        })
    
    with open('method_summary.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Method', 'Avg_ANLS', 'Avg_Accuracy_%', 'Avg_Tokens', 
                     'Total_Tool_Calls', 'Avg_Tool_Usage_%', 'Avg_Latency_s']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Generated: method_summary.csv ({len(rows)} rows)")

def generate_shot_comparison(all_data):
    """Generate comparison by shot configuration"""
    rows = []
    
    # Group by method and shots
    from collections import defaultdict
    shot_stats = defaultdict(lambda: defaultdict(list))
    
    for exp in all_data:
        method = exp['mode'].upper() if exp['mode'] in ['direct', 'cot', 'react'] else exp['mode']
        shot_stats[method][exp['shots']].append(exp['mean_anls'])
    
    # Create rows
    for method in sorted(shot_stats.keys()):
        row = {'Method': method}
        for shots in [0, 4, 8]:
            if shots in shot_stats[method]:
                avg_anls = sum(shot_stats[method][shots]) / len(shot_stats[method][shots])
                row[f'{shots}shot_ANLS'] = f"{avg_anls:.4f}"
            else:
                row[f'{shots}shot_ANLS'] = 'N/A'
        rows.append(row)
    
    with open('shot_comparison.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Method', '0shot_ANLS', '4shot_ANLS', '8shot_ANLS']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Generated: shot_comparison.csv ({len(rows)} rows)")

def generate_model_comparison(all_data):
    """Generate comparison by model size"""
    rows = []
    
    # Group by method and model
    from collections import defaultdict
    model_stats = defaultdict(lambda: defaultdict(list))
    
    for exp in all_data:
        method = exp['mode'].upper() if exp['mode'] in ['direct', 'cot', 'react'] else exp['mode']
        model_stats[method][exp['model_size']].append(exp['mean_anls'])
    
    # Create rows
    for method in sorted(model_stats.keys()):
        row = {'Method': method}
        for model in ['2B', '4B', '8B']:
            if model in model_stats[method]:
                avg_anls = sum(model_stats[method][model]) / len(model_stats[method][model])
                row[f'{model}_ANLS'] = f"{avg_anls:.4f}"
            else:
                row[f'{model}_ANLS'] = 'N/A'
        rows.append(row)
    
    with open('model_comparison.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Method', '2B_ANLS', '4B_ANLS', '8B_ANLS']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Generated: model_comparison.csv ({len(rows)} rows)")

def generate_best_configs(all_data, top_n=10):
    """Generate top N configurations by ANLS"""
    sorted_data = sorted(all_data, key=lambda x: x['mean_anls'], reverse=True)[:top_n]
    
    rows = []
    for exp in sorted_data:
        rows.append({
            'Rank': len(rows) + 1,
            'Method': exp['mode'].upper() if exp['mode'] in ['direct', 'cot', 'react'] else exp['mode'],
            'Model': f"Qwen3-VL-{exp['model_size']}",
            'Shots': exp['shots'],
            'ANLS': f"{exp['mean_anls']:.4f}",
            'Accuracy_%': f"{exp['accuracy']*100:.1f}"
        })
    
    with open('top_configurations.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Rank', 'Method', 'Model', 'Shots', 'ANLS', 'Accuracy_%']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Generated: top_configurations.csv ({len(rows)} rows)")

def main():
    """Main function to generate all CSV tables"""
    print("="*70)
    print("Generating CSV Tables from Experimental Data")
    print("="*70)
    print()
    
    # Load data
    print("Loading data...")
    all_data = load_json_data()
    
    if not all_data:
        print("Error: No data found!")
        print("Please ensure final_report_data.json exists")
        return
    
    print(f"Loaded {len(all_data)} experimental results")
    print()
    
    # Generate CSV files
    print("Generating CSV files...")
    print()
    
    generate_detailed_csv(all_data)
    generate_method_summary(all_data)
    generate_shot_comparison(all_data)
    generate_model_comparison(all_data)
    generate_best_configs(all_data)
    
    print()
    print("="*70)
    print("Complete! Generated 5 CSV files:")
    print("="*70)
    print("  1. experimental_results_detailed.csv  - All configurations")
    print("  2. method_summary.csv                 - Summary by method")
    print("  3. shot_comparison.csv                - Comparison by shots")
    print("  4. model_comparison.csv               - Comparison by model size")
    print("  5. top_configurations.csv             - Top 10 configurations")
    print("="*70)

if __name__ == "__main__":
    main()

