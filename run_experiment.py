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
import signal

def plot_ablation_results(exp_dir):
    results = {}

    # Load predictions and ground truth
    with open(os.path.join(exp_dir, "predictions.json")) as f:
        preds = json.load(f)
    with open(os.path.join(exp_dir, "ground_truth.json")) as f:
        gts = json.load(f)
    with open(os.path.join(exp_dir, "tool_metrics.json")) as f:
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
    plt.figure(figsize=(6,4))
    sns.barplot(x=["accuracy"], y=[results["accuracy"]])
    plt.ylabel("Accuracy")
    plt.title("Ablation: Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "accuracy.png"))
    plt.close()

    # Tool usage
    plt.figure(figsize=(6,4))
    sns.barplot(x=["avg_tool_calls"], y=[results["avg_tool_calls"]])
    plt.ylabel("Avg Tool Calls per Sample")
    plt.title("Ablation: Tool Usage")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "tool_usage.png"))
    plt.close()

    # Tool latency
    plt.figure(figsize=(6,4))
    sns.barplot(x=["avg_tool_latency"], y=[results["avg_tool_latency"]])
    plt.ylabel("Avg Tool Latency (s)")
    plt.title("Ablation: Tool Latency")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "tool_latency.png"))
    plt.close()

# start a server
def start_server(description, cmd, port):
    print(f"\n=== Starting {description} ===")
    print(" ".join(cmd))
    # Start in new process group so we can kill entire tree later
    proc = subprocess.Popen(cmd, start_new_session=True)
    wait_for_port(port)
    return proc

def kill_process(proc):
    """Kill process and all children, then wait for termination."""
    if proc.poll() is not None:
        return  # already exited
    try:
        # Kill the entire process group if possible
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        # Fallback: kill just the process
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass

def kill_vllm_on_port(port):
    """Kill any process listening on the given port."""
    # Use fuser to find and kill processes on the port
    subprocess.run(f"fuser -k {port}/tcp", shell=True, stderr=subprocess.DEVNULL)
    # Also kill any remaining vllm processes owned by this user
    subprocess.run("pkill -9 -f 'vllm serve'", shell=True, stderr=subprocess.DEVNULL)

# wait for port to be up with exponential backoff
def wait_for_port(port, host="0.0.0.0", timeout=1200):
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

    processes = []

    # vlm server
    vlm_cmd = [
        "vllm", "serve", args.model,
        "--max-model-len", "40000", 
        "--gpu-memory-utilization", "0.95",
        "--port", "8000",
        "--allowed-local-media-path", os.getcwd(),
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes"
    ]
    processes.append(start_server("Vision-Language Model Server", vlm_cmd, 8000))

    return processes

def run_single_experiment(mode, args, out_name):
    print(f"\n=== Running Experiment Mode: {mode} ===")

    cmd = [
        "python",
        "main_tool_model.py",
        "--dataset", args.dataset,
        "--split", args.split,
        "--model", args.model,
        "--model_url", args.model_url,
        "--shots", str(args.shots),
        "--num_samples", str(args.num_samples),
        "--max_tokens", str(args.max_tokens),
        "--experiment_name", out_name,
        "--mode", mode,
    ]

    if args.tools:
        cmd.append("--tools")
        cmd.extend(args.tools)

    if mode == "cot":
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
    # Set WandB environment variables so main_tool_model can call wandb.init() with sensible defaults
    env = os.environ.copy()
    env["WANDB_ENTITY"] = "sllm_project"  # Team name
    env["WANDB_PROJECT"] = "sllm_multimodal_agent"
    env["WANDB_RUN_NAME"] = out_name
    # Store local wandb files inside the experiment folder
    env["WANDB_DIR"] = out_name
    subprocess.run(cmd, check=True, env=env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="docvqa")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--model_url", default="http://0.0.0.0:8000/v1")

    parser.add_argument("--mode", default="cot", choices=["direct", "cot", "react"])
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=100)

    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--tools", nargs='*', default=None, help="List of tool names to enable (or a single comma-separated string)")

    parser.add_argument("--metrics_interval", type=int, default=60, help="Seconds between metrics polls")
    parser.add_argument("--no_metrics_logging", action="store_true", help="Disable metrics logging to file")

    args = parser.parse_args()

    # Normalize args.tools into a clean list of tool names.
    # Support these input styles:
    #  - --tools calculator extract_text_from_image
    #  - --tools calculator,extract_text_from_image
    #  - default string value from the parser
    if isinstance(args.tools, str):
        # single string, possibly comma-separated
        args.tools = [t.strip() for t in args.tools.split(',') if t.strip()]
    elif args.tools is None:
        args.tools = []
    else:
        # list: flatten any comma-separated entries inside the list
        normalized = []
        for t in args.tools:
            if isinstance(t, str) and ',' in t:
                normalized.extend([x.strip() for x in t.split(',') if x.strip()])
            elif t:
                normalized.append(t)
        args.tools = normalized

    procs = ensure_servers(args)

    # base prefix (without mode) so each mode run gets its own folder suffix
    tools_str = "_".join(args.tools) if args.tools else "none"
    out_name = f"exp_{args.dataset}_{args.mode}_shots{args.shots}_samples{args.num_samples}_model{args.model.replace('/', '_')}_maxtokens{args.max_tokens}_tools{tools_str}"
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

        metrics_urls = [metrics_url_from_base(args.model_url)]

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

        for url in metrics_urls:
            stop_event = threading.Event()
            t = threading.Thread(target=metrics_logger, args=(url, metrics_logfile, args.metrics_interval, stop_event), daemon=True)
            metrics_threads.append(t)
            metrics_stop_events.append(stop_event)
            t.start()

    try:
        # run the requested mode; create per-mode out_name and pass to main
        
        run_single_experiment(args.mode, args, out_name)
        print("\n=== Generating Ablation Plots ===")
        plot_ablation_results(out_name)

    finally:
        # stop metrics threads
        for ev in metrics_stop_events:
            ev.set()
        for t in metrics_threads:
            t.join(timeout=5)
        # kill vllm server(s)
        for p in procs:
            kill_process(p)
        # Force kill anything still on port 8000
        kill_vllm_on_port(8000)

        # close metrics log
        if metrics_logfile:
            with open(metrics_logfile, 'a', encoding='utf-8') as fh:
                fh.write(f"Log ended at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    main()
