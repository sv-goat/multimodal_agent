"""
Batch experiment runner for parameter sweeps across models, modes, and configurations.

This module generates and executes all experiment combinations for ablation studies,
iterating over different models, prompting modes, shot counts, and tool combinations.

Usage:
    python master_runner.py  # Run with defaults
    python master_runner.py --models Qwen/Qwen3-VL-4B-Instruct --num_shots 0 4 8
"""

import argparse
import itertools
import subprocess


def main(sequence_lengths, modes, models, all_tool_combinations, num_shots):
    """Generate and run all experiment combinations."""
    commands = []
    for sequence_length, mode, model in itertools.product(sequence_lengths, modes, models):
        
        for num_shot in num_shots:
            if mode != 'react':
                command = f"CACHE_DIR=/tmp/vllm_cache_$$; export TORCHINDUCTOR_CACHE_DIR=$CACHE_DIR; python run_experiment.py --max_tokens {sequence_length} --mode {mode} --model {model} --shots {num_shot}; rm -rf $CACHE_DIR"
                commands.append(command)
            else:
                for tool_combo in all_tool_combinations:
                    tools_arg = f"--tools {','.join(tool_combo)}" if tool_combo else ""
                    command = f"CACHE_DIR=/tmp/vllm_cache_$$; export TORCHINDUCTOR_CACHE_DIR=$CACHE_DIR; python run_experiment.py --max_tokens {sequence_length} --mode {mode} --model {model} --shots {num_shot} {tools_arg}; rm -rf $CACHE_DIR"
                    commands.append(command)
        

    for command in commands:
        subprocess.run(command, shell=True)

        # Clear pyc cache from root after every loop
        subprocess.run("find / -name '*.pyc' -delete", shell=True)

if __name__ == '__main__':
    # Accept the arguments in main from argparse with set defaults
    parser = argparse.ArgumentParser(description="Run experiments with various configurations")
    parser.add_argument("--sequence_lengths", type=int, nargs='+', default=[512, 1024],
                        help="List of sequence lengths to use")
    parser.add_argument("--modes", type=str, nargs='+', default=['react'],
                        help="Modes to run experiments in (e.g., react, cot, direct)")
    parser.add_argument("--models", type=str, nargs='+', default=['Qwen/Qwen3-VL-8B-Instruct', 'Qwen/Qwen3-VL-4B-Instruct', 'Qwen/Qwen3-VL-2B-Instruct'],
                        help="Models to run the experiments with")
    parser.add_argument("--tool_combinations", type=str, nargs='+', action='append',
                        default=None,
                        help="Tool combinations for ReAct mode. Use multiple times for multiple combos, e.g., --tool_combinations extract_text_from_image --tool_combinations extract_text_from_image calculator")
    parser.add_argument("--num_shots", type=int, nargs='+', default=[0, 4, 8],
                        help="Number of shots to use in experiments")
    args = parser.parse_args()

    # Set default tool combinations if not provided
    if args.tool_combinations is None:
        args.tool_combinations = [
            ["extract_text_from_image"],
            ["extract_text_from_image", "calculator"],
            ["extract_text_from_image", "calculator", "web_search"]
        ]

    main(args.sequence_lengths, args.modes, args.models, args.tool_combinations, args.num_shots)
