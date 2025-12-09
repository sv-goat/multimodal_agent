import subprocess
import argparse
import time
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import socket
import time

def plot_ablation_results(exp_dir_prefix, modes=["baseline","fewshot","cot","tool_calling","react"]):
    results = {}
    for mode in modes:
        exp_dir = Path(f"{exp_dir_prefix}_{mode}")
        if not exp_dir.exists():
            continue

        # Load predictions and ground truth
        with open(exp_dir / "predictions.json") as f:
            preds = json.load(f)
        with open(exp_dir / "ground_truth.json") as f:
            gts = json.load(f)
        with open(exp_dir / "tool_metrics.json") as f:
            tools = json.load(f)

        # Compute simple accuracy
        correct = sum(
            preds[qid]["answer"].strip().lower() in [a.lower() for a in gts[qid]["answers"]]
            for qid in preds
        )
        total = len(preds)
        accuracy = correct / total

        avg_tool_calls = sum(t.get("tool_total_count",0) for t in tools) / max(len(tools),1)
        avg_latency = sum(t.get("tool_latency",0) for t in tools) / max(len(tools),1)

        results[mode] = {
            "accuracy": accuracy,
            "avg_tool_calls": avg_tool_calls,
            "avg_tool_latency": avg_latency
        }

    if not results:
        print("No results found to plot.")
        return

    # Accuracy plot
    plt.figure(figsize=(8,5))
    sns.barplot(x=list(results.keys()), y=[r["accuracy"] for r in results.values()])
    plt.ylabel("Accuracy")
    plt.title("Ablation: Accuracy Across Methods")
    plt.tight_layout()
    plt.savefig(f"{exp_dir_prefix}_accuracy.png")
    plt.close()

    # Tool usage
    plt.figure(figsize=(8,5))
    sns.barplot(x=list(results.keys()), y=[r["avg_tool_calls"] for r in results.values()])
    plt.ylabel("Avg Tool Calls per Sample")
    plt.title("Ablation: Tool Usage Across Methods")
    plt.tight_layout()
    plt.savefig(f"{exp_dir_prefix}_tool_usage.png")
    plt.close()

    # Tool latency
    plt.figure(figsize=(8,5))
    sns.barplot(x=list(results.keys()), y=[r["avg_tool_latency"] for r in results.values()])
    plt.ylabel("Avg Tool Latency (s)")
    plt.title("Ablation: Tool Latency Across Methods")
    plt.tight_layout()
    plt.savefig(f"{exp_dir_prefix}_tool_latency.png")
    plt.close()

# start a server
def start_server(description, cmd, port):
    print(f"\n=== Starting {description} ===")
    print(" ".join(cmd))
    proc = subprocess.Popen(cmd)
    wait_for_port(port)
    return proc

def kill_process(proc):
    if proc.poll() is not None:
        # already exited
        return
    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass

# wait for port to be up with exponential backoff
def wait_for_port(port, host="0.0.0.0", timeout=120):
    start = time.time()
    backoff = 1
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            if s.connect_ex((host, port)) == 0:
                print(f"Port {port} is ready.")
                return True
        print(f"Waiting for port {port}...")
        time.sleep(backoff)
        backoff *= 2
    raise TimeoutError(f"Timed out waiting for port {port}.")

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

    processes.append(start_server("Controller Model Server", controller_cmd, 8000))

    # vlm server
    vlm_cmd = [
        "vllm", "serve", args.vlm_model,
        "--gpu-memory-utilization", "0.4",
        "--port", "6006",
        "--allowed-local-media-path", os.getcwd(),
    ]
    processes.append(start_server("Vision-Language Model Server", vlm_cmd, 6006))

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
        print("\n=== Generating Ablation Plots ===")
        plot_ablation_results(args.experiment_prefix)

    finally:
        for p in procs:
            kill_process(p)

if __name__ == "__main__":
    main()
