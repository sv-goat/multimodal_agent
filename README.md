# Scaling LLMs Project: Multimodal Reasoning with Agentic Small LLMs

## Team Information
- **Members**:
  - Xinchen Zhang ( xz3052 )
  - Siddarth Ijju ( si2462 )
  - Sai Vignesh ( sv2795 )

---

## 1. Problem Statement
How does agentic reasoning outperform prompt-only few-shot or Chain of Thought (CoT) reasoning, especially on multimodal tasks in small open source LLMs?

---

## 2. Model Description
The models used are `Qwen/Qwen3-VL-2B-Instruct`, `Qwen/Qwen3-VL-4B-Instruct`, and `Qwen/Qwen3-VL-8B-Instruct`. 
The models were evaluated on the DocVQA dataset. 
Further, three different modes were used: Direct, Chain-of-Thought and ReAct. 
The models were run on A6000 GPUs with 48GB RAM, using vLLM as their backend. 

---

## 3. Final Results Summary

### 8-shot Results

| Model | Direct | Best ReAct | Absolute Gain | Relative Gain |
|-------|--------|------------|---------------|---------------|
| 2B    | 0.1595 | 0.4780     | +0.3185       | +200%         |
| 4B    | 0.2405 | 0.5537     | +0.3132       | +130%         |
| 8B    | 0.2269 | 0.4642     | +0.2373       | +105%         |

### Latency and Accuracy Tradeoff

| Method            | Avg Tokens | Mean Latency (s) | Accuracy (%) |
|-------------------|------------|------------------|--------------|
| Direct            | 627        | 0.22             | 8.7          |
| CoT               | 1824       | 0.24             | 11.3         |
| ReAct (OCR)       | 4739       | 5.44             | **41.0**     |
| ReAct (OCR+Calc)  | 4800       | 5.19             | 40.0         |
| ReAct (OCR+Calc+Web) | 5295    | 5.18             | 34.7         |

### Key Observations

1. **ReAct significantly outperforms Direct and CoT**: Across all model sizes, ReAct with OCR achieves higher accuracy than direct prompting, demonstrating that tool-augmented reasoning is crucial for document understanding tasks.

2. **OCR is the most impactful tool**: The extract_text_from_image (OCR) tool provides the largest accuracy boost. Adding calculator and web search tools provides marginal or negative gains, suggesting that accurate text extraction is the primary bottleneck.

3. **Smaller models benefit more from agentic reasoning**: The 2B model shows a 200% relative improvement with ReAct, compared to 105% for the 8B model, indicating that tool augmentation can help close the gap between small and large models.

4. **Latency tradeoff is significant**: ReAct methods are slower than direct prompting due to multiple tool calls and reasoning steps, but the accuracy gains justify this tradeoff for document QA tasks.

---

## 4. Reproducibility Instructions

### A. Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

---

### B. Wandb Dashboard
View training and evaluation metrics here: https://wandb.ai/sllm_project/sllm_multimodal_agent/workspace?nw=nwusersv2795

---

### C. Running Experiments
To run the experiments, run:
```bash
python experiment/master_runner.py --modes react --models Qwen/Qwen3-VL-4B-Instruct --num_shots 0 4 8
```

### D. Quickstart: Minimum Reproducible Result
To reproduce our results, run:
```bash
# Step 1: Set up environment
pip install -r requirements.txt

# Step 2: Download dataset into the images folder
python setup/save_docvqa_images.py --output_dir images --split validation --num_images 100

# Step 3: Generate few-shot examples
python setup/generate_traces.py --trace_type react --num_samples 10 --output fewshot_library/react.json

# Step 4: Run experiments
python experiment/master_runner.py

# Step 5: Extract results to a CSV format for further analysis
python eval/generate_csv_tables.py
```

---

## 5. Project Structure
```
├── experiment/
│   ├── main_tool_model.py   # Main experiment runner
│   ├── run_experiment.py    # Single experiment orchestrator (starts vLLM server)
│   └── master_runner.py     # Batch experiment runner (parameter sweeps)
├── setup/
│   ├── generate_traces.py   # Generate few-shot traces (ReAct or CoT)
│   └── save_docvqa_images.py # Download DocVQA images from HuggingFace
├── eval/
│   ├── docvqa_eval.py       # ANLS evaluation metrics
│   └── generate_csv_tables.py # Extract and visualize metrics
├── scripts/
│   └── run_exps.sh          # SLURM batch job script
├── images/                  # DocVQA images (sample_0.png, sample_1.png, ...)
├── fewshot_library/         # Few-shot prompts (react_2.json, cot_docvqa.json)
└── requirements.txt         # Python dependencies
```

---

## 6. Notes
- Images are expected in `images/` and few-shot traces in `fewshot_library/`.
- Contact information:
  - Xinchen Zhang ( xz3052@columbia.edu )
  - Siddarth Ijju ( si2462@columbia.edu )
  - Sai Vignesh ( sv2795@columbia.edu )