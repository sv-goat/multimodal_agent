import itertools
import subprocess

def main():
    sequence_lengths = [1024]
    modes = ['direct', 'cot', 'react']
    models = ['Qwen/Qwen3-VL-8B-Instruct', 'Qwen/Qwen3-VL-4B-Instruct', 'Qwen/Qwen3-VL-2B-Instruct']
    tools = ["calculator", "extract_text_from_image"]
    all_tool_combinations = []
    for r in range(1, len(tools) + 1):
        combinations = itertools.combinations(tools, r)
        for combo in combinations:
            all_tool_combinations.append(list(combo))
    num_shots = [0, 4, 8]

    commands = []
    for sequence_length, mode, model, num_shot, tool_combo in itertools.product(sequence_lengths, modes, models, num_shots, all_tool_combinations):
        command = f"python run_experiment.py --max_tokens {sequence_length} --mode {mode} --model {model} --shots {num_shot} --tools {','.join(tool_combo)}"
        commands.append(command)

    for command in commands:
        subprocess.run(command, shell=True)

        # Clear pyc cache after every loop
        subprocess.run("find . -name '*.pyc' -delete", shell=True)

if __name__ == '__main__':
    main()
