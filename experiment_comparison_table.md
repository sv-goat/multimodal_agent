# Experiment Comparison Table

## Experiment 2 Performs Better

| Metric | Experiment 1 | Experiment 2 | Improvement |
|--------|---------------|--------------|-------------|
| **Accuracy** | 40% | **61%** | **+21%** ✅ |
| **Mean ANLS** | 0.431 | **0.639** | **+48%** ✅ |
| **Avg Latency** | 16.03s | **13.66s** | **-15%** ✅ |
| **Tool Usage Rate** | 63% | **96%** | **+33%** ✅ |
| **Tool Latency** | 2.49s | **0.865s** | **-65%** ✅ |
| **Tokens/Sample** | 2,037 | **1,925** | **-5%** ✅ |

---

## Detailed Comparison Table

### Accuracy Metrics

| Metric | Experiment 1 | Experiment 2 | Difference | Improvement |
|--------|---------------|--------------|------------|-------------|
| Mean ANLS | 0.4311 | **0.6386** | +0.2075 | **+48%** ✅ |
| Accuracy (ANLS>0.5) | 40.00% | **61.00%** | +21% | **+52.5%** ✅ |
| Correct Predictions | 40/100 | **61/100** | +21 | - |
| Incorrect Predictions | 60/100 | 39/100 | -21 | - |

### Latency Performance

| Metric | Experiment 1 | Experiment 2 | Difference | Improvement |
|--------|---------------|--------------|------------|-------------|
| Mean Latency | 16.03s | **13.66s** | -2.37s | **-15%** ✅ |
| Median Latency | 13.62s | **13.22s** | -0.40s | **-3%** ✅ |
| Min Latency | 8.35s | 8.27s | -0.08s | -1% |
| Max Latency | 29.82s | **23.96s** | -5.86s | **-20%** ✅ |
| P95 Latency | 27.72s | **19.03s** | -8.69s | **-31%** ✅ |
| Std Deviation | 5.00s | **2.63s** | -2.37s | **-47%** ✅ |
| Total Time | 26.71 min | **22.77 min** | -3.94 min | **-15%** ✅ |

### Tool Usage

| Metric | Experiment 1 | Experiment 2 | Difference | Improvement |
|--------|---------------|--------------|------------|-------------|
| Tool Usage Rate | 63.00% | **96.00%** | +33% | **+52%** ✅ |
| Total Tool Calls | 64 | **99** | +35 | **+55%** ✅ |
| Avg Tool Calls/Sample | 0.64 | **0.99** | +0.35 | **+55%** ✅ |
| Tool Success Rate | 100% | 100% | 0% | - |
| Avg Tool Latency | 2.49s | **0.865s** | -1.625s | **-65%** ✅ |
| Max Tool Latency | 5.81s | **3.84s** | -1.97s | **-34%** ✅ |
| Total Tool Latency | 156.83s | **83.02s** | -73.81s | **-47%** ✅ |

### Token Usage

| Metric | Experiment 1 | Experiment 2 | Difference | Improvement |
|--------|---------------|--------------|------------|-------------|
| Total Prompt Tokens | 145,973 | **141,810** | -4,163 | **-3%** ✅ |
| Total Completion Tokens | 57,707 | **50,704** | -7,003 | **-12%** ✅ |
| Total Tokens | 203,680 | **192,514** | -11,166 | **-5%** ✅ |
| Avg Prompt Tokens | 1,459.7 | **1,418.1** | -41.6 | **-3%** ✅ |
| Avg Completion Tokens | 577.1 | **507.0** | -70.1 | **-12%** ✅ |
| Avg Total Tokens/Sample | 2,036.8 | **1,925.1** | -111.7 | **-5%** ✅ |
| Max Prompt Tokens | 2,705 | 2,434 | -271 | -10% |
| Max Completion Tokens | 1,020 | 919 | -101 | -10% |

### System Resources

| Metric | Experiment 1 | Experiment 2 | Difference | Improvement |
|--------|---------------|--------------|------------|-------------|
| Peak Memory (MB) | 817 | **801** | -16 MB | **-2%** ✅ |
| Virtual Memory (GB) | 6.89 | 7.42 | +0.53 GB | +8% |
| CPU Time (s) | 16.10 | 19.85 | +3.75s | +23% |
| HTTP Requests | 16 | 83 | +67 | +419% |
| Avg Inference Time | 7.24s | **3.01s** | -4.23s | **-58%** ✅ |
| Avg Decode Time | 7.18s | **2.98s** | -4.20s | **-58%** ✅ |
| Avg Prefill Time | 0.060s | **0.035s** | -0.025s | **-42%** ✅ |
| Avg HTTP Request Duration | 6.66s | **3.11s** | -3.55s | **-53%** ✅ |

---

## Experiment Configuration

### Experiment 1: `fewshot_qwen_docvqa_ocr_calc_vlm_512`
- **Configuration**: Few-shot + OCR + Calculator + VLM
- **Controller Model**: Qwen3-8B
- **VLM Tool**: Qwen3-VL-4B-Instruct
- **Max Tokens**: 512

### Experiment 2: `qwen3_8b_controller_qwen3_4B_VL_DocVQA_+Calculator`
- **Configuration**: Qwen3-8B Controller + Qwen3-VL-4B + Calculator
- **Controller Model**: Qwen3-8B
- **VLM Tool**: Qwen3-VL-4B-Instruct
- **Tools**: Calculator + VLM

---

## Key Findings Summary

### ✅ Experiment 2 outperforms in all key metrics:

1. **Accuracy improved by 21%** (40% → 61%)
2. **Mean ANLS improved by 48%** (0.431 → 0.639)
3. **Latency reduced by 15%** (16.03s → 13.66s)
4. **Tool usage rate increased by 33%** (63% → 96%)
5. **Tool latency reduced by 65%** (2.49s → 0.865s)
6. **Token usage reduced by 5%** (2,037 → 1,925 tokens/sample)

### 🎯 Most Significant Improvements:

- **Tool Latency**: Reduced by 65% (largest improvement)
- **Mean ANLS**: Improved by 48%
- **Inference Speed**: Improved by 58% (7.24s → 3.01s)
- **Latency Stability**: Std deviation reduced by 47% (5.00s → 2.63s)

---

**Generated**: 2025-11-30  
**Data Sources**: 
- Experiment 1: `fewshot_qwen_docvqa_ocr_calc_vlm_512`
- Experiment 2: `qwen3_8b_controller_qwen3_4B_VL_DocVQA_+Calculator`

