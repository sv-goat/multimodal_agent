import json
import os
from openai import OpenAI
from vlm_as_a_tool import call_vl_model

def get_current_temperature(location: str, unit: str = "celsius"):
    """Get current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, and the unit in a dict
    """
    return {
        "temperature": 26.1,
        "location": location,
        "unit": unit,
    }


def get_temperature_date(location: str, date: str, unit: str = "celsius"):
    """Get temperature at a location and date.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        date: The date to get the temperature for, in the format "Year-Month-Day".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, the date and the unit in a dict
    """
    return {
        "temperature": 25.9,
        "location": location,
        "date": date,
        "unit": unit,
    }


def get_function_by_name(name):
    if name == "get_current_temperature":
        return get_current_temperature
    if name == "get_temperature_date":
        return get_temperature_date
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
    image_path = "docvqa_example.jpg"
    full_image_path = os.path.join(base_path, image_path)
    prompt = "Describe the image in detail."
    messages = [
        {"role": "user",  "content": f"Describe the image in the file '{full_image_path}' with the prompt '{prompt}'"},
    ]
    
    openai_api_key = "EMPTY"
    openai_api_base = "http://0.0.0.0:8000/v1"
    
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    model_name = "Qwen/Qwen3-8B"

    # Loop to automatically handle multiple tool calls until we get a final response
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

    messages.append(response.choices[0].message.model_dump())
    
    if tool_calls := messages[-1].get("tool_calls", None):
        for tool_call in tool_calls:
            call_id: str = tool_call["id"]
            if fn_call := tool_call.get("function"):
                fn_name: str = fn_call["name"]
                fn_args: dict = json.loads(fn_call["arguments"])
            
                fn_res: str = json.dumps(get_function_by_name(fn_name)(**fn_args))
    
                messages.append({
                    "role": "tool",
                    "content": fn_res,
                    "tool_call_id": call_id,
                })
        # Continue loop to get the next response after tool execution
    
    # Get the final response
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

    messages.append(response.choices[0].message.model_dump())
    print("final response", response.choices[0].message.content)
