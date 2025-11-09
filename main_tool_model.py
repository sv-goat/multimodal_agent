import json
import os
from openai import OpenAI
from vlm_as_a_tool import call_vl_model
from datasets import load_dataset
import time


def get_function_by_name(name):
    if name == "get_image_description":
        return get_image_description

def get_image_description(image_path: str, prompt: str = "Describe the image in detail."):
    """Get a description of an image.

    Args:
        image_path: The path to the image file.
        prompt: The prompt to describe the image. Defaults to "Describe the image in detail."

    Returns:
        The description of the image as a string.
    """
    return call_vl_model(prompt, image_path)

if __name__ == "__main__":
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_image_description",
                "description": "Get a description of an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": 'The path to the image file.',
                        },
                        "prompt": {
                            "type": "string",
                            "description": 'The prompt to describe the image. Defaults to "Describe the image in detail."',
                            "default": "Describe the image in detail.",
                        },
                    },
                    "required": ["image_path", "prompt"],
                },
            },
        },
    ]


    base_path = os.getcwd()
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA")
    i = 0

    openai_api_key = "EMPTY"
    openai_api_base = "http://0.0.0.0:8000/v1"
    
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    model_name = "Qwen/Qwen3-8B"

    # Create the 2-shot prompt for the controller model
    two_shot_prompt = ""
    for i in range(i, i+2):
        image_path = f"images/sample_{i}.png"
        full_image_path = os.path.join(base_path, image_path)
        sample = ds["validation"][i]
        two_shot_prompt += f"Image path: {full_image_path}\nQuestion Types: {sample['question_types']}\nQ: {sample['question']}\nA: {sample['answers']}\n"

    predictions = {}
    ground_truths = {}
    sample_latencies = []
    tool_metrics = []
    usage_metrics = []

    for i in range(i+2, i+2+5):
        sample = ds["validation"][i]
        image_path = f"images/sample_{i}.png"
        full_image_path = os.path.join(base_path, image_path)
        prompt = f"{two_shot_prompt}Image path: {full_image_path}\nQuestion Types: {sample['question_types']}\nQ: {sample['question']}\nA: "
        messages = [
            {"role": "user",  "content": f"{prompt}"},
        ]

        sample_start_time = time.time()
        prompt_tokens = 0
        completion_tokens = 0

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            temperature=0.7,
            top_p=0.8,
            max_tokens=512,
            extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        )

        if hasattr(response, 'usage'):
            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens

        messages.append(response.choices[0].message.model_dump())
        
        tool_start_time = time.time()
        success_count = 0
        total_count = 0
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
        tool_latency = tool_end_time - tool_start_time
        tool_metrics.append({
            "latency": tool_latency,
            "success": success_count,
            "total": total_count,
        })
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            temperature=0.7,
            top_p=0.8,
            max_tokens=512,
            extra_body={
                "repetition_penalty": 1.05,
            },
        )
        sample_end_time = time.time()
        sample_latencies.append(sample_end_time - sample_start_time)
        
        # Record token usage for final response
        if hasattr(response, 'usage'):
            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens
            usage_metrics.append({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            })

        messages.append(response.choices[0].message.model_dump())

        # Print the answer. Comment out after testing so that it doesn't affect latency. 
        print(f"Sample {i}:")
        print(f"Question: {sample['question']}")
        print(f"Predicted Answer: {messages[-1]['content']}")
        print(f"Ground Truth Answer: {sample['answers']}\n")

        # Save the answers in a json file
        predictions[sample['questionId']] = {
            "answer": messages[-1]['content']
        }
        ground_truths[sample['questionId']] = {
            "answers": sample['answers']
        }
    
    
    # Save all results
    with open("predictions.json", "w", encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    with open("ground_truth.json", "w", encoding='utf-8') as f:
        json.dump(ground_truths, f, indent=2)
    
    # Save metrics and statistics
    with open("sample_latencies.json", "w", encoding='utf-8') as f:
        json.dump(sample_latencies, f, indent=2)
    with open("tool_metrics.json", "w", encoding='utf-8') as f:
        json.dump(tool_metrics, f, indent=2)
    with open("usage_metrics.json", "w", encoding='utf-8') as f:
        json.dump(usage_metrics, f, indent=2)
