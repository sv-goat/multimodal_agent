"""
Generate few-shot traces (ReAct or Chain-of-Thought) for DocVQA using GPT-4.

This module uses OpenAI's GPT-4o to generate reasoning traces that can be used
as few-shot examples for training or prompting smaller models.

Usage:
    python generate_traces.py --trace_type react --num_samples 10
    python generate_traces.py --trace_type cot --num_samples 10 --output cot_traces.json
"""

import json
import os
import argparse
from typing import Any, Dict, List
from datasets import load_dataset
import openai
import base64
import re
from tqdm import tqdm

BASE_PATH = "."
ALLOWED_TOOL_NAMES = {"extract_text_from_image", "calculator"}


def encode_image(path) -> str:
    """Encode an image file to base64 data URL."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def get_fields(ds, sample_index):
    """Extract fields from a dataset sample."""
    s = ds[sample_index]
    image_path = f"images/sample_{sample_index}.png"
    full_image_path = os.path.join(BASE_PATH, image_path)

    return {
        "image_path": full_image_path,
        "question": s["question"],
        "answers": s["answers"],
        "question_types": s.get("question_types", []),
        "qid": s["questionId"],
    }

def validate_react_output(text):
    # Minimal checks: ensure Final Answer exists and any Action uses allowed tools
    if "Final Answer:" not in text:
        raise ValueError("Missing 'Final Answer:' line.")

    # Check Action lines reference allowed tools (very small sanity check)
    for m in re.findall(r"Action\s*:\s*([a-zA-Z_0-9]+)", text):
        if m not in ALLOWED_TOOL_NAMES:
            raise ValueError(f"Illegal tool used in Action: {m}")

def build_react_trace(client, model_name, fields, max_retries = 1):

    image_path = fields["image_path"]
    question = fields["question"]

    system_prompt = (
        "You are a vision-language reasoning agent.\n"
        "Generate an OpenAI API compatible ReAct reasoning trace. STRICT RULES:\n"
        "- Allowed tools ONLY:\n"
        "    extract_text_from_image{\"image_path\":..., \"lang\":...}\n"
        "    calculator{\"expression\":...}\n"
        "- Each cycle must be:\n"
        "    Thought:\n"
        "    Action:\n"
        "    Observation:\n"
        "- End with: Final Answer: <answer>\n"
        "- Respond ONLY with the ReAct trace.\n"
    )

    image_data_url = encode_image(image_path)

    # need to submit this way to keep token limits
    user_content = [
        {
            "type": "text",
            "text": (
                f"Question: {question}\n\n"
                "Produce the ReAct reasoning trace following the required format."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": image_data_url},
        },
    ]

    last_err = None

    for _ in range(max_retries + 1):
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=1024,
        )

        trace_text = response.choices[0].message.content.strip()

        try:
            validate_react_output(trace_text)
            return {
                "qid": fields.get("qid"),
                "image_path": image_path,
                "question": question,
                "react_trace": trace_text,
            }
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to produce a valid ReAct trace: {last_err}")


def build_cot_trace(client, model_name, fields, max_retries=1):
    """Build a Chain-of-Thought trace for a given sample."""
    image_path = fields["image_path"]
    question = fields["question"]
    answers = fields["answers"]

    system_prompt = (
        "You are a vision-language reasoning agent.\n"
        "Generate a Chain-of-Thought reasoning trace. STRICT RULES:\n"
        "- Think step by step about the image and question.\n"
        "- Show your reasoning process clearly.\n"
        "- End with: Final Answer: <answer>\n"
        "- Respond ONLY with the CoT trace.\n"
    )

    image_data_url = encode_image(image_path)

    user_content = [
        {
            "type": "text",
            "text": (
                f"Question: {question}\n"
                f"Ground truth answers: {answers}\n\n"
                "Produce the Chain-of-Thought reasoning trace ending with 'Final Answer: <answer>'."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": image_data_url},
        },
    ]

    last_err = None

    for _ in range(max_retries + 1):
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=1024,
        )

        trace_text = response.choices[0].message.content.strip()

        # Minimal validation: ensure Final Answer exists
        if "Final Answer:" in trace_text:
            return {
                "qid": fields.get("qid"),
                "image_path": image_path,
                "question": question,
                "cot_trace": trace_text,
            }
        else:
            last_err = ValueError("Missing 'Final Answer:' line.")

    raise RuntimeError(f"Failed to produce a valid CoT trace: {last_err}")


def generate_cot_trace_dataset(client, model_name="gpt-4o", split="validation", output_path="cot_traces.json", num_samples=10):
    """Generate CoT traces for a dataset."""
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split)

    samples = []

    for i in tqdm(range(min(num_samples, len(ds))), desc="Generating CoT traces"):
        fields = get_fields(ds, i)
        trace = build_cot_trace(client, model_name, fields)

        samples.append({
            "index": i,
            "fields": fields,
            "cot_trace": trace,
        })

    output = {"samples": samples}

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(samples)} CoT traces to {output_path}")
    return output


def generate_react_trace_dataset(client, model_name="gpt-4o", split="validation", output_path="react_traces.json", num_samples=10):
    """Generate ReAct traces for a dataset."""
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split)

    samples = []

    for i in tqdm(range(min(num_samples, len(ds))), desc="Generating ReAct traces"):
        fields = get_fields(ds, i)
        trace = build_react_trace(client, model_name, fields)

        samples.append({
            "index": i,
            "fields": fields,
            "react_trace": trace,
        })

    output = {"samples": samples}

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(samples)} ReAct traces to {output_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ReAct or CoT traces for DocVQA")
    parser.add_argument("--trace_type", type=str, default="react", choices=["react", "cot"],
                        help="Type of trace to generate: 'react' or 'cot'")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to generate traces for")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: {trace_type}_traces.json)")
    parser.add_argument("--split", type=str, default="validation",
                        help="Dataset split to use")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="OpenAI model to use for generation")
    args = parser.parse_args()

    # Set default output path based on trace type
    if args.output is None:
        args.output = f"{args.trace_type}_traces.json"

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    client = openai.OpenAI(api_key=openai_api_key)

    print(f"Generating {args.num_samples} {args.trace_type.upper()} traces...")

    if args.trace_type == "react":
        data = generate_react_trace_dataset(
            client,
            model_name=args.model,
            split=args.split,
            output_path=args.output,
            num_samples=args.num_samples,
        )
    else:  # cot
        data = generate_cot_trace_dataset(
            client,
            model_name=args.model,
            split=args.split,
            output_path=args.output,
            num_samples=args.num_samples,
        )

    # Print summary of first few examples
    print(f"\n{'='*80}")
    print(f"Generated {len(data['samples'])} traces")
    print(f"{'='*80}\n")

    for sample in data["samples"][:3]:
        print(f"Index: {sample['index']}")
        print(f"Question: {sample['fields']['question']}")
        trace_key = "react_trace" if args.trace_type == "react" else "cot_trace"
        print(f"\n{args.trace_type.upper()} Trace:\n")
        print(sample[trace_key])
        print("\n" + "-"*80 + "\n")
