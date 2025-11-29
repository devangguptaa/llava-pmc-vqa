import json
import re
from collections import Counter

def normalize_text(s):
    if s is None:
        return ""
    s = s.strip().lower()

    # remove leading "answer:" if present
    if s.startswith("answer:"):
        s = s.split(":", 1)[1].strip()

    # remove punctuation
    s = re.sub(r"[^\w\s]", " ", s)

    # collapse multiple spaces
    s = re.sub(r"\s+", " ", s)

    return s.strip()


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gt_tokens   = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)



def evaluate(pred_json_path):
    with open(pred_json_path, "r") as f:
        data = json.load(f)

    total = 0
    em_count = 0
    f1_sum = 0.0

    for ex in data:
        gt  = ex.get("gt_answer", "")
        pred = ex.get("prediction", "")

        if gt is None or str(gt).strip() == "":
            continue

        total += 1

        norm_gt   = normalize_text(str(gt))
        norm_pred = normalize_text(str(pred))


        if norm_pred == norm_gt:
            em_count += 1


        f1 = f1_score(str(pred), str(gt))
        f1_sum += f1

    if total == 0:
        print("No valid examples found.")
        return

    em  = em_count / total
    f1m = f1_sum / total

    with open ("baseline_eval_results.txt", "w") as f:
        f.write(f"Total examples evaluated: {total}\n")
        f.write(f"Exact Match (EM): {em_count} / {total} = {em:.4f}\n")
        f.write(f"Token-level F1 (mean): {f1m:.4f}\n")
    print(f"# Examples evaluated: {total}")
    print(f"Exact Match (EM): {em_count} / {total} = {em:.4f}")
    print(f"Token-level F1 (mean): {f1m:.4f}")


if __name__ == "__main__":
    pred_json = "pmc_vqa_llava_baseline_predictions.json"
    evaluate(pred_json)
