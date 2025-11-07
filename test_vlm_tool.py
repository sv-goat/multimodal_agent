from vlm_as_a_tool import call_vl_model
import os

prompt = "What is the date mentioned in the document?"
base_path = os.getcwd()
image_path = "docvqa_example.jpg"
image_path = os.path.join(base_path, image_path)
result = call_vl_model(prompt, image_path)
print("VL Model Output:", result)
