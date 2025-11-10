import json
import string
import re
from typing import Dict, Union
import numpy as np
import sys

def normalize_answer(answer: str) -> str:
    """
    Performs answer normalization according to DocVQA standards:
    - Convert to lowercase
    - Remove articles
    - Remove extra whitespace
    """
    answer = answer.lower()
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
    answer = ' '.join(answer.split())
    return answer

def extract_answer(text: str) -> str:
    """
    Extract a concise final answer from a model response.
    Heuristics:
    - Match 'Final Answer: <answer>'
    - Match 'Answer: <answer>' or 'A: <answer>'
    - Fallback to first non-empty line
    """
    if not isinstance(text, str):
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
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return text.strip()

def compute_anls(pred: str, gt: str) -> float:
    """
    Compute Normalized Levenshtein Similarity between prediction and ground truth.
    Returns a score between 0 and 1.
    """
    
    # Normalize both strings
    pred = normalize_answer(pred)
    gt = normalize_answer(gt)
    
    # Simple dynamic programming implementation of Levenshtein distance
    m, n = len(pred), len(gt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred[i-1] == gt[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,      # deletion
                          dp[i][j-1] + 1,        # insertion
                          dp[i-1][j-1] + cost)   # substitution
    
    # Normalize by the length of the longer string
    lev_dist = dp[m][n]
    max_len = max(m, n)
    
    # Convert distance to similarity
    nls = 1.0 - (lev_dist / max_len) if max_len > 0 else 1.0
    return max(0.0, nls)  # Ensure non-negative

def evaluate_docvqa(predictions: Union[str, Dict], 
                   ground_truth: Union[str, Dict],
                   threshold: float = 0.5,
                   prediction_key: str = "answer",
                   apply_extraction: bool = True) -> Dict:
    """
    Evaluate DocVQA predictions using ANLS metric.
    
    Args:
        predictions: Either a path to predictions JSON or a dict with format:
                   {qid: {prediction_key: predicted_answer or raw response}}
        ground_truth: Either a path to ground truth JSON or a dict with format:
                     {qid: {"answers": [answer1, answer2, ...]}}
        threshold: ANLS threshold (default: 0.5 as per DocVQA)
        prediction_key: Key inside prediction dict that contains the answer or raw response.
        apply_extraction: If True, extract a concise answer from raw responses.
    
    Returns:
        Dict containing:
        - ANLS score
        - Accuracy (% of answers with ANLS > threshold)
        - Per-question scores
    """
    # Load files if paths provided
    if isinstance(predictions, str):
        with open(predictions, 'r') as f:
            predictions = json.load(f)
    if isinstance(ground_truth, str):
        with open(ground_truth, 'r') as f:
            ground_truth = json.load(f)
            
    scores = {}
    anls_scores = []
    correct = 0
    total = 0
    
    for qid, pred_info in predictions.items():
        if qid not in ground_truth:
            continue
            
        # Flexible retrieval of predicted answer
        if isinstance(pred_info, dict):
            if prediction_key in pred_info:
                raw_pred = pred_info[prediction_key]
            elif "content" in pred_info:
                raw_pred = pred_info["content"]
            elif "prediction" in pred_info:
                raw_pred = pred_info["prediction"]
            else:
                # take any first string value
                raw_pred = next((v for v in pred_info.values() if isinstance(v, str)), "")
        else:
            raw_pred = str(pred_info)

        pred_answer = extract_answer(raw_pred) if apply_extraction else str(raw_pred)
        gt_answers = ground_truth[qid]["answers"]
        
        # Take max similarity across all ground truth answers
        max_sim = max(compute_anls(pred_answer, gt) for gt in gt_answers)
        anls_scores.append(max_sim)
        
        # Count as correct if similarity > threshold
        if max_sim > threshold:
            correct += 1
        total += 1
        
        scores[qid] = {
            "prediction": pred_answer,
            "ground_truth": gt_answers,
            "anls": max_sim,
            "is_correct": max_sim > threshold
        }
    
    # Compute final metrics
    mean_anls = np.mean(anls_scores) if anls_scores else 0.0
    accuracy = (correct / total) if total > 0 else 0.0
    
    return {
        "mean_anls": mean_anls,
        "accuracy": accuracy,
        "threshold": threshold,
        "n_questions": total,
        "per_question": scores
    }

if __name__ == "__main__":
    
    # Default params file path
    params_file = "params.json"
    
    # Load parameters
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    # Validate required parameters
    required_params = ["predictions", "ground_truth"]
    missing_params = [p for p in required_params if p not in params]
    if missing_params:
        print(f"Error: Missing required parameters: {', '.join(missing_params)}")
        sys.exit(1)
    
    # Set defaults for optional parameters
    params.setdefault("threshold", 0.5)
    params.setdefault("prediction_key", "answer")
    params.setdefault("apply_extraction", True)
    
    # Run evaluation
    results = evaluate_docvqa(
        params["predictions"],
        params["ground_truth"],
        params.get("threshold", 0.5),
        params.get("prediction_key", "answer"),
        params.get("apply_extraction", True)
    )
    
    # Print summary
    print(f"\nDocVQA Evaluation Results:")
    print(f"Mean ANLS: {results['mean_anls']:.4f}")
    print(f"Accuracy (ANLS > {params['threshold']}): {results['accuracy']:.4f}")
    print(f"Number of questions evaluated: {results['n_questions']}")
    
    # Save detailed results if output path provided
    if "output" in params:
        with open(params["output"], 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {params['output']}")