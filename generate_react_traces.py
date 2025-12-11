import json
import os
from typing import Any, Dict, List
from datasets import load_dataset
import openai
import base64
import re
from IPython.display import Image, display
from tqdm import tqdm

BASE_PATH = "."
ALLOWED_TOOL_NAMES = {"get_image_description", "extract_text_from_image", "calculator"}

def encode_image(path) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def get_fields(ds, sample_index):
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
    for action_line in re.findall(r"Action\s*:\s*(.*)", text):
        m = re.match(r"\s*([a-zA-Z_0-9]+)", action_line)
        if not m:
            raise ValueError(f"Malformed Action line: {action_line}")

        tool_name = m.group(1)
        if tool_name not in ALLOWED_TOOL_NAMES:
            raise ValueError(f"Illegal tool '{tool_name}' used.")

    if "Final Answer:" not in text:
        raise ValueError("Missing 'Final Answer:' line.")

    if not ("Thought:" in text and "Action:" in text and "Observation:" in text):
        raise ValueError("ReAct structure incomplete.")

def build_react_trace(client, model_name, fields, max_retries = 1):

    image_path = fields["image_path"]
    question = fields["question"]

    system_prompt = (
        "You are a vision-language reasoning agent.\n"
        "Generate a ReAct reasoning trace. STRICT RULES:\n"
        "- Allowed tools ONLY:\n"
        "    get_image_description{\"image_path\":..., \"prompt\":...}\n"
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
            max_tokens=800,
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

def generate_react_trace_dataset(client, split = "validation", output_path = "react_traces.json", num_samples = 100):
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split)

    samples = []

    for i in tqdm(range(min(num_samples, len(ds)))):
        fields = get_fields(ds, i)
        trace = build_react_trace(client, "gpt-4o", fields)

        samples.append({
            "index": i,
            "fields": fields,
            "react_trace": trace,
        })

    output = {"samples": samples}

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output

if __name__ == "__main__":
    openai_api_key = os.environ["OPENAI_API_KEY"]
    client = openai.OpenAI(api_key=openai_api_key)

    data = generate_react_trace_dataset(
        client,
        split="validation",
        output_path="react_traces.json",
        num_samples = 100
    )

    # show first 5 examples
    for sample in data["samples"][:5]:
        img_path = sample["fields"]["image_path"]

        print(f"Index: {sample['index']}")
        print(f"Question: {sample['fields']['question']}")

        # Display the image
        display(Image(filename=img_path))

        print("\nReAct Trace:\n")
        print(sample["react_trace"])
        print("\n" + "-"*80 + "\n")
