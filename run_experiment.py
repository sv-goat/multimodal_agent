import subprocess
import argparse
import time
import os
from pathlib import Path

# start a server
def start_server(description, cmd):
    print(f"\n=== Starting {description} ===")
    print(" ".join(cmd))
    proc = subprocess.Popen(cmd)
    time.sleep(5)
    return proc

# verify the servers are up 
def ensure_servers(args):
    if args.skip_server_setup:
        print("Skipping server startup.")
        return []

    processes = []

    # controller server
    controller_cmd = [
        "vllm", "serve", args.controller_model,
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
        "--gpu-memory-utilization", "0.5",
        "--port", "8000"
    ]

    processes.append(start_server("Controller Model Server", controller_cmd))

    # vlm server
    vlm_cmd = [
        "vllm", "serve", args.vlm_model,
        "--gpu-memory-utilization", "0.5",
        "--max-model-len", "40000",
        "--port", "6006",
        "--allowed-local-media-path", os.getcwd(),
    ]
    processes.append(start_server("Vision-Language Model Server", vlm_cmd))

    return processes

def run_single_experiment(mode, args):
    print(f"\n=== Running Experiment Mode: {mode} ===")

    out_name = f"{args.experiment_prefix}_{mode}"
    cmd = [
        "python",
        "main_tool_model.py",
        "--dataset", args.dataset,
        "--split", args.split,
        "--controller_base_url", args.controller_base_url,
        "--controller_model", args.controller_model,
        "--vlm_base_url", args.vlm_base_url,
        "--shots", str(args.shots),
        "--num_samples", str(args.num_samples),
        "--start_index", str(args.start_index),
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--max_tokens", str(args.max_tokens),
        "--experiment_name", out_name,
    ]

    if mode == "baseline":
        pass
    elif mode == "fewshot":
        cmd.append("--use_fewshot")
    elif mode == "cot":
        cmd.append("--use_fewshot")
        cmd.append("--use_cot")
    elif mode == "tool_calling":
        cmd.append("--use_fewshot")
        cmd.append("--use_tools")
    elif mode == "react":
        cmd.append("--use_fewshot")
        cmd.append("--use_tools")
        cmd.append("--use_react")

    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="docvqa")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--controller_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--controller_base_url", default="http://0.0.0.0:8000/v1")
    parser.add_argument("--vlm_model", default="stabilityai/stablelm-2-12b")
    parser.add_argument("--vlm_base_url", default="http://0.0.0.0:6006/v1")

    parser.add_argument("--mode", default="baseline",
                        choices=["all", "baseline", "fewshot", "tool_calling", "cot", "react"])

    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--start_index", type=int, default=0)

    parser.add_argument("--experiment_prefix", type=str, default="ablations_qwen_docvqa_ocr_calc_vlm_1024")

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=512)

    # for large scale experiments, avoid setup every time
    parser.add_argument("--skip_server_setup", action="store_true")

    args = parser.parse_args()

    procs = ensure_servers(args)

    try:
        if args.mode == "all":
            for mode in ["baseline", "fewshot", "tool_calling", "cot", "react"]:
                run_single_experiment(mode, args)
        else:
            run_single_experiment(args.mode, args)

    finally:
        for p in procs:
            p.terminate()

if __name__ == "__main__":
    main()
