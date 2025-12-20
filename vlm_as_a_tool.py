"""
VLM (Vision-Language Model) wrapper for tool-based image understanding.

This module provides a simple interface to call a vLLM-served VLM for
image description and visual question answering tasks.
"""

from openai import OpenAI


def call_vl_model(prompt: str, image_path: str):
    """
    Call Qwen/Qwen3-VL-4B-Instruct with a prompt and image using vllm.

    Args:
        prompt: The user prompt/question about the image.
        image_path: The path to the image file.

    Returns:
        The VL model's output as a string (e.g., answer or caption).
    """
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:6006/v1",
    )
    
    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-4B-Instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                {"type": "text", "text": prompt}
            ]
        }],
    )
    
    return response.choices[0].message.content
