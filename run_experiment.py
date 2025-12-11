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
import threading
from urllib import request as urlrequest
import wandb

def plot_ablation_results(exp_dir):
    results = {}

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

    results = {
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
    plt.savefig(f"{exp_dir}_accuracy.png")
    plt.close()

    # Tool usage
    plt.figure(figsize=(8,5))
    sns.barplot(x=list(results.keys()), y=[r["avg_tool_calls"] for r in results.values()])
    plt.ylabel("Avg Tool Calls per Sample")
    plt.title("Ablation: Tool Usage Across Methods")
    plt.tight_layout()
    plt.savefig(f"{exp_dir}_tool_usage.png")
    plt.close()

    # Tool latency
    plt.figure(figsize=(8,5))
    sns.barplot(x=list(results.keys()), y=[r["avg_tool_latency"] for r in results.values()])
    plt.ylabel("Avg Tool Latency (s)")
    plt.title("Ablation: Tool Latency Across Methods")
    plt.tight_layout()
    plt.savefig(f"{exp_dir}_tool_latency.png")
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
def wait_for_port(port, host="0.0.0.0", timeout=240):
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
        "--gpu-memory-utilization", "0.6",
        "--port", "8000"
    ]

    processes.append(start_server("Controller Model Server", controller_cmd, 8000))

    # vlm server
    vlm_cmd = [
        "vllm", "serve", args.vlm_model,
        "--max-model-len", "40000", 
        "--gpu-memory-utilization", "0.35",
        "--port", "6006",
        "--allowed-local-media-path", os.getcwd(),
    ]
    processes.append(start_server("Vision-Language Model Server", vlm_cmd, 6006))

    return processes

def run_single_experiment(mode, args, out_name, wandb_instance=None):
    print(f"\n=== Running Experiment Mode: {mode} ===")

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
        "--tools" + args.tools
    ]

    if mode == "cot":
        cmd.append("--use_fewshot")
        cmd.append("--use_tools")
        cmd.append("--use_cot")
    elif mode == "tool_calling":
        cmd.append("--use_fewshot")
        cmd.append("--use_tools")
    elif mode == "react":
        cmd.append("--use_fewshot")
        cmd.append("--use_tools")
        cmd.append("--use_react")

    print(" ".join(cmd))
    # Set WandB environment variables so main_tool_model can call wandb.init() with sensible defaults
    env = os.environ.copy()
    # Use dataset as project and out_name as run name. WANDB_API_KEY should be set externally.
    env["WANDB_PROJECT"] = "sllm_multimodal_agent"
    env["WANDB_RUN_NAME"] = out_name
    # Store local wandb files inside the experiment folder
    env["WANDB_DIR"] = out_name
    subprocess.run(cmd, check=True, env=env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="docvqa")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--controller_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--controller_base_url", default="http://0.0.0.0:8000/v1")
    parser.add_argument("--vlm_model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--vlm_base_url", default="http://0.0.0.0:6006/v1")

    parser.add_argument("--mode", default="cot", choices=["direct", "cot", "react"])
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--start_index", type=int, default=0)

    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--tools", nargs='*', default=["get_image_description"], help="List of tool names to enable")

    parser.add_argument("--metrics_interval", type=int, default=60, help="Seconds between metrics polls")
    parser.add_argument("--no_metrics_logging", action="store_true", help="Disable metrics logging to file")

    args = parser.parse_args()

    procs = ensure_servers(args)

    # base prefix (without mode) so each mode run gets its own folder suffix
    out_name = f"exp_{args.dataset}_{args.mode}_shots{args.shots}_samples{args.num_samples}_start{args.start_index}"
    os.makedirs(out_name, exist_ok=True)

    # Start metrics logger (poll controller /metrics) to track vllm server stats
    metrics_threads = []
    metrics_stop_events = []
    metrics_logfile = None
    if not args.no_metrics_logging:
        def metrics_url_from_base(base):
            u = base.rstrip('/')
            if u.endswith('/v1'):
                u = u[:-3]
            return u.rstrip('/') + '/metrics'

        metrics_controller_url = metrics_url_from_base(args.controller_base_url)
        metrics_vlm_url = metrics_url_from_base(args.vlm_base_url)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        # save metrics inside the per-mode experiment folder so they live with results
        metrics_logfile = os.path.join(out_name, f"vllm_metrics_{timestamp}.log")

        def metrics_logger(url, outpath, interval, stop_event):
            with open(outpath, 'a', encoding='utf-8') as fh:
                fh.write(f"Log started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                while not stop_event.is_set():
                    ts = time.strftime('%Y-%m-%d %H:%M:%S')
                    try:
                        resp = urlrequest.urlopen(url, timeout=10)
                        body = resp.read().decode('utf-8')
                        fh.write(f"=== {ts} ({url}) ===\n")
                        fh.write(body + '\n')
                    except Exception as e:
                        fh.write(f"=== {ts} ({url}) (ERROR fetching metrics): {e} ===\n")
                    fh.flush()
                    stop_event.wait(interval)
                fh.write(f"Log ended at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        for url in [metrics_controller_url, metrics_vlm_url]:
            stop_event = threading.Event()
            t = threading.Thread(target=metrics_logger, args=(url, metrics_logfile, args.metrics_interval, stop_event), daemon=True)
            metrics_threads.append(t)
            metrics_stop_events.append(stop_event)
            t.start()

    try:
        # run the requested mode; create per-mode out_name and pass to main

        # Init wandb config for this experiment
        wandb_config = {
            "dataset": args.dataset,
            "controller_model": args.controller_model,
            "shots": args.shots,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "avail_tools": args.tools,
            "mode": args.mode,
        }
        wandb.init(config=wandb_config)
        
        run_single_experiment(args.mode, args, out_name, wandb_instance=wandb)
        print("\n=== Generating Ablation Plots ===")
        plot_ablation_results(out_name)

    finally:
        # stop metrics threads
        for ev in metrics_stop_events:
            ev.set()
        for t in metrics_threads:
            t.join(timeout=5)
        for p in procs:
            kill_process(p)

        # stop wandb
        wandb.finish()

if __name__ == "__main__":
    main()
