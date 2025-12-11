import json
import os
import re
import time
import argparse
from typing import Any, Optional
from openai import OpenAI
from datasets import load_dataset
from vlm_as_a_tool import call_vl_model
from tqdm import tqdm
import wandb
from docvqa_eval import compute_anls
from pathlib import Path
try:
    from PIL import Image  # type: ignore
    import pytesseract  # type: ignore
    _OCR_AVAILABLE = True
    print("Tesseract OCR is available for use.")
except Exception:
    _OCR_AVAILABLE = False
    print("Tesseract OCR is not available for use.")


def get_function_by_name(name):
    if name == "get_image_description":
        return get_image_description
    if name == "extract_text_from_image":
        return extract_text_from_image
    if name == "calculator":
        return calculator
    raise ValueError(f"Unknown tool function requested: {name}")

def get_image_description(image_path: str, prompt: str = "Describe the image in detail."):
    """Get a description of an image.

    Args:
        image_path: The path to the image file.
        prompt: The prompt to describe the image. Defaults to "Describe the image in detail."

    Returns:
        The description of the image as a string.
    """
    return call_vl_model(prompt, image_path)

def extract_text_from_image(image_path: str, lang: str = "eng") -> str:
    """OCR: Extract raw text from an image using Tesseract if available.

    Falls back to the VLM description pathway if pytesseract/PIL are unavailable.
    """
    if _OCR_AVAILABLE:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image, lang=lang).strip()
    # Fallback: ask the VLM to extract text as best as possible
    return call_vl_model("Extract all visible text verbatim from the image.", image_path)

def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely and return the result as string.

    Supports +, -, *, /, **, %, parentheses, and floats.
    """
    # Very conservative safe eval: only numbers, operators, dots, spaces, and parentheses
    if not re.fullmatch(r"[0-9\.\s\+\-\*\/\%\(\)]+", expression):
        raise ValueError("Expression contains unsupported characters.")
    # Evaluate in a restricted namespace
    result = eval(expression, {"__builtins__": {}}, {})
    return str(result)

def extract_answer(text: Optional[str]) -> str:
    """Try to extract a concise final answer from a verbose model response.

    Heuristics:
    - Look for 'Final Answer: <answer>'
    - Look for lines starting with 'Answer:' or 'A:'
    - Fallback to first non-empty line
    """
    if not text:
        return ""
    patterns = [
        r"Final Answer\s*:\s*(.+)",
        r"Answer\s*:\s*(.+)",
        r"\bA\s*:\s*(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    # Fallback: first non-empty line
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return text.strip()

def define_tools():
    avail_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_image_description",
                "description": "Describe image content or answer a question about it using a VLM.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Absolute path to an image file accessible to the server.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Instruction/question about the image.",
                            "default": "Describe the image in detail.",
                        },
                    },
                    "required": ["image_path", "prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_text_from_image",
                "description": "OCR: Extract all visible text verbatim from an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Absolute path to an image file.",
                        },
                        "lang": {
                            "type": "string",
                            "description": "Tesseract language code (e.g., eng).",
                            "default": "eng",
                        },
                    },
                    "required": ["image_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Compute a basic arithmetic expression. Use for numeric reasoning.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Arithmetic expression with numbers and operators (+-*/%() ).",
                        }
                    },
                    "required": ["expression"],
                },
            },
        },
    ]
    allowed_tools = []
    for tool in args.tools:
        for available_tool in avail_tools:
            if tool == available_tool["function"]["name"]:
                allowed_tools.append(available_tool)
    return allowed_tools


def get_fields_function(dataset, base_path):
    if dataset == "docvqa":
        ds = load_dataset("lmms-lab/DocVQA", "DocVQA")
        def get_fields(sample_index: int) -> dict[str, Any]:
            s = ds[args.split][sample_index]
            image_path = f"images/sample_{sample_index}.png"
            full_image_path = os.path.join(base_path, image_path)
            return {
                "image_path": full_image_path,
                "question": s["question"],
                "answers": s["answers"],
                "question_types": s.get("question_types", []),
                "qid": s["questionId"],
            }
    else:
        # Minimal ChartQA wiring; adjust if field names differ in your local dataset
        ds = load_dataset("lmms-lab/ChartQA", split="test")
        def get_fields(sample_index: int) -> dict[str, Any]:
            s = ds[sample_index]
            image_path = f"images_chartqa/sample_{sample_index}.png"
            full_image_path = os.path.join(base_path, image_path)
            answers = s.get("answers", [s.get("answer", "")])
            return {
                "image_path": full_image_path,
                "question": s.get("query", s.get("question", "")),
                "answers": answers,
                "question_types": s.get("type", []),
                "qid": str(s.get("id", sample_index)),
            }
    return get_fields

def run_experiment(args):
    tools = define_tools()
    
    base_path = os.getcwd()
    get_fields = get_fields_function(args.dataset, base_path)

    openai_api_key = "EMPTY"
    openai_api_base = args.controller_base_url
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    model_name = args.controller_model

    # Build few-shot prompt
    few_shot_prompt = "You are answering questions about documents and charts. "
    # add tools if enabled
    few_shot_prompt += "Use tools when needed." if args.use_tools else ""
    few_shot_prompt += "Always reply with a single line: 'Final Answer: <answer>'.\n\n"

    # react prompt
    if args.use_react:
        few_shot_prompt = (
            "You are a reasoning agent that answers questions about documents and charts. "
            "You should think step by step and use tools when needed. "
            "For each step:\n"
            "1. Thought: your reasoning\n"
            "2. Action: call a tool if needed\n"
            "3. Observation: record the tool output\n"
            "Finally, always return a single line as 'Final Answer: <answer>'.\n\n"
        )

    # add few shot examples if enabled
    if args.use_fewshot:
        for k in range(args.start_index, args.start_index + args.shots):
            s = get_fields(k)
            few_shot_prompt += (
                f"Image path: {s['image_path']}\n"
                f"Question Types: {s['question_types']}\n"
                f"Q: {s['question']}\n"
                f"A: {s['answers']}\n\n"
            )

    predictions = {}
    ground_truths = {}
    sample_latencies = []
    tool_metrics = []
    usage_metrics = []

    # Prepare a W&B table to record per-sample results if W&B is initialized
    results_table = wandb.Table(columns=["image", "user_prompt", "model_trace", "final_extracted_answer", "actual_answer", "anls"])

    # Use existing ANLS computation from docvqa_eval
    def best_similarity(pred: str, gts: list[str]) -> float:
        if not pred:
            return 0.0
        best = 0.0
        for gt in (gts or []):
            try:
                sim = compute_anls(pred, str(gt))
                if sim > best:
                    best = sim
            except Exception:
                continue
        return float(best)

    eval_start = args.start_index + args.shots
    eval_end = eval_start + args.num_samples
    for i in tqdm(range(eval_start, eval_end)):
        s = get_fields(i)

        prompt = f"{few_shot_prompt}"

        # add cot if enabled
        if args.use_cot and not args.use_react:
            prompt += "Let's think step by step.\n"

        prompt += (
            "Now answer the following. "
            + ("Use tools if helpful. " if args.use_tools else "")
            + "Return only one line as 'Final Answer: <answer>'.\n"
            + "Image path: {s['image_path']}\n"
            + "Question Types: {s['question_types']}\n"
            + "Q: {s['question']}\nA: "
        )

        messages = [
            {"role": "user", "content": f"{prompt}"},
        ]

        sample_start_time = time.time()
        prompt_tokens = 0
        completion_tokens = 0

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools if args.use_tools else None, # add tools if enabled
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            extra_body={
                "repetition_penalty": 1.05,
#                "chat_template_kwargs": {"enable_thinking": False},
            }
        )

        if hasattr(response, 'usage'):
            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens

        messages.append(response.choices[0].message.model_dump())

        # Loop through in case there are a series of tool calls. 
        stop_reason = response.choices[0].finish_reason
        tool_latency = 0
        success_count = 0
        total_count = 0
        while args.use_tools and (stop_reason == "tool_calls" or stop_reason == "tool_call"):
            tool_start_time = time.time()
            if tool_calls := messages[-1].get("tool_calls", None):
                for tool_call in tool_calls:
                    call_id: str = tool_call["id"]
                    if fn_call := tool_call.get("function"):
                        try:
                            total_count += 1
                            fn_name: str = fn_call["name"]
                            fn_args: dict = json.loads(fn_call["arguments"])
                            fn_res: str = json.dumps(get_function_by_name(fn_name)(**fn_args))
                            messages.append({
                                "role": "tool",
                                "content": fn_res,
                                "tool_call_id": call_id,
                            })
                            success_count += 1
                        except Exception as e:
                            print(f"Tool call failed: {e}")
            tool_end_time = time.time()
            tool_latency += tool_end_time - tool_start_time

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                extra_body={
                    "repetition_penalty": 1.05,
                },
            )

            messages.append(response.choices[0].message.model_dump())
            stop_reason = response.choices[0].finish_reason

            if hasattr(response, 'usage'):
                prompt_tokens += response.usage.prompt_tokens
                completion_tokens += response.usage.completion_tokens

        sample_end_time = time.time()
        sample_latencies.append(sample_end_time - sample_start_time)

        tool_metrics.append({
            "tool_latency": tool_latency,
            "tool_success_count": success_count,
            "tool_total_count": total_count,
        })

        # Record token usage for final response
        if hasattr(response, 'usage'):
            usage_metrics.append({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            })

        raw_answer = messages[-1]['content']
        final_answer = extract_answer(raw_answer)

        print(f"Sample {i}:")
        print(f"Question: {s['question']}")
        print(f"Predicted Answer (raw): {raw_answer}")
        print(f"Predicted Answer (extracted): {final_answer}")
        print(f"Ground Truth Answer: {s['answers']}\n")

        predictions[s['qid']] = {
            "answer": final_answer
        }
        ground_truths[s['qid']] = {
            "answers": s['answers']
        }
        # Append a row to the W&B table (if enabled)
        trace_parts = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get('role', '')
                content = m.get('content', '')
                trace_parts.append(f"{role}: {content}")
        model_trace = "\n".join(trace_parts)

        image_field = None
        image_field = wandb.Image(s.get('image_path'))

        anls_score = best_similarity(final_answer, s.get('answers', []))

        results_table.add_data(
            image_field,
            prompt,
            model_trace,
            final_answer,
            s.get('answers'),
            anls_score,
        )
    
    experiment_prefix = args.experiment_name
    exp_dir = Path(experiment_prefix)
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "predictions.json", "w", encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    with open(exp_dir / "ground_truth.json", "w", encoding='utf-8') as f:
        json.dump(ground_truths, f, indent=2)

    with open(exp_dir / "sample_latencies.json", "w", encoding='utf-8') as f:
        json.dump(sample_latencies, f, indent=2)
    with open(exp_dir / "tool_metrics.json", "w", encoding='utf-8') as f:
        json.dump(tool_metrics, f, indent=2)
    with open(exp_dir / "usage_metrics.json", "w", encoding='utf-8') as f:
        json.dump(usage_metrics, f, indent=2)

    # Log the W&B table once at the end of the run (if enabled)
    wandb.log({"results_table": results_table})
    # Compute run-level metrics (mean ANLS and accuracy) and log to W&B
    anls_list = []
    correct = 0
    total_q = 0
    # Compute mean ANLS from the individual ANLS scores from the table
    for row in results_table.data:
        anls = row[-1]  # ANLS is the last column
        anls_list.append(anls)
        total_q += 1
        if anls > 0.0:
            correct += 1

    mean_anls = (sum(anls_list) / len(anls_list)) if anls_list else 0.0
    accuracy = (correct / total_q) if total_q > 0 else 0.0
    # Log run-level metrics to W&B if available
    if getattr(wandb, 'run', None) is not None:
        wandb.log({"mean_anls": mean_anls, "accuracy": accuracy, "n_questions": total_q})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tool-augmented DocVQA/ChartQA pipeline.")
    parser.add_argument("--dataset", type=str, default="docvqa", choices=["docvqa", "chartqa"], help="Dataset to use.")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split.")
    parser.add_argument("--controller_base_url", type=str, default="http://0.0.0.0:8000/v1")
    parser.add_argument("--controller_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--vlm_base_url", type=str, default="http://0.0.0.0:6006/v1", help="For documentation only; vlm_as_a_tool uses this.")
    parser.add_argument("--shots", type=int, default=2, help="Number of in-context examples.")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of evaluation samples.")
    parser.add_argument("--experiment_name", type=str, default="fewshot_qwen_docvqa_ocr_calc_vlm_1024", help="Experiment name used to create results folder.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index in the split.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=512)

    # List of tool names to enable (will filter available tools to this list)
    parser.add_argument("--tools", nargs='*', default=["get_image_description"], help="List of tool names to enable")

    # add flags for experiments
    parser.add_argument("--use_tools", action="store_true", help="Enable tool calling-based prompting")
    parser.add_argument("--use_fewshot", action="store_true", help="Enable few-shot prompting")
    parser.add_argument("--use_cot", action="store_true", help="Enable chain-of-thought prompting")
    parser.add_argument("--use_react", action="store_true", help="Enable ReAct-style prompting")


    args = parser.parse_args()
    # Initialize Weights & Biases if available and configured via environment (run_experiment sets WANDB_ env vars)
    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_name = os.environ.get("WANDB_RUN_NAME")
    wandb_dir = os.environ.get("WANDB_DIR")
    wandb_mode = os.environ.get("WANDB_MODE")  # optional: 'offline' or 'online'

    init_kwargs = {}
    if wandb_project:
        init_kwargs["project"] = wandb_project
    if wandb_name:
        init_kwargs["name"] = wandb_name
    if wandb_dir:
        init_kwargs["dir"] = wandb_dir
    if wandb_mode:
        init_kwargs["mode"] = wandb_mode

    if init_kwargs:
        wandb.init(reinit=True, **init_kwargs)
        print(f"W&B initialized with: {init_kwargs}")
    else:
        # fallback: attempt a default init (harmless if WANDB_API_KEY not set)
        wandb.init(reinit=True)
        print("W&B initialized with default settings")

    run_experiment(args)
