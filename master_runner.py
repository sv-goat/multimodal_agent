import itertools
import subprocess

def main():
    sequence_lengths = [1024]
    modes = ['react']
    models = ['Qwen/Qwen3-VL-8B-Instruct', 'Qwen/Qwen3-VL-4B-Instruct', 'Qwen/Qwen3-VL-2B-Instruct']
    all_tool_combinations = [["extract_text_from_image"], ["extract_text_from_image", "calculator"], ["extract_text_from_image", "calculator", "web_search"]]

    num_shots = [0, 4, 8]

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
    main()
