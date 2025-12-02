import json
import requests
import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

DEFAULT_MODEL = "phi3:mini"
MAX_WORKERS = 2  # tune this, 4 - 8 is usually fine

PROMPT_TEMPLATE = """
You are a strict medical answer evaluator.

Your task is to decide if a model's prediction is SCIENTIFICALLY and
CLINICALLY equivalent to the ground truth answer.

There are only two labels:

- correct:
  The prediction expresses the same clinical fact as the ground truth.
  Minor rephrasing, extra filler words, or grammatical changes are allowed
  as long as the medical meaning is the same.
  For very short answers (for example "yes", "no", "normal", "abnormal",
  a single structure name, or a single diagnosis), treat answers as correct
  if they clearly state the same truth value or concept, even if they are
  in a longer sentence.
  Example:
    GT: "no"
    Pred: "no it is not"   → correct
    GT: "yes"
    Pred: "yes, it is"     → correct

- incorrect:
  The prediction does not match the scientific meaning of the ground truth,
  is clinically different, or conveys the opposite truth value.
  Example:
    GT: "no"
    Pred: "yes"            → incorrect
    GT: "no pneumothorax"
    Pred: "pneumothorax"   → incorrect

Be strict about medical meaning, but do NOT mark answers incorrect
just because they are longer sentences if they clearly say the same thing.

Return exactly one word: correct or incorrect.

Ground truth: {gt}
Prediction: {pred}
"""


def normalize_text(s):
    if s is None:
        return ""
    s = s.strip().lower()

    if s.startswith("answer:"):
        s = s.split(":", 1)[1].strip()

    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)

    return s.strip()


def run_ollama_http(model, prompt):
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, 
                "options": {
                "num_predict": 4,   # we only need "correct" or "incorrect"
                "temperature": 0.0, # deterministic, faster
            },
            },
        timeout=60,
    )
    if resp.status_code != 200:
        sys.stderr.write(f"Ollama HTTP error {resp.status_code}: {resp.text}\n")
        return ""
    data = resp.json()
    return data.get("response", "").strip()


def parse_verdict(text):
    text = text.strip().lower()
    if not text:
        return "incorrect"

    tokens = text.split()
    first = tokens[0]

    if first == "correct":
        return "correct"
    if first == "incorrect":
        return "incorrect"

    for tok in tokens:
        if tok == "incorrect":
            return "incorrect"
        if tok == "correct":
            return "correct"

    return "incorrect"


def judge_one(item_idx, item, model):
    gt = normalize_text(item.get("gt_answer", ""))
    pred = normalize_text(item.get("prediction", ""))

    if not gt or not pred:
        return None  # skip

    prompt = PROMPT_TEMPLATE.format(gt=gt, pred=pred)
    raw = run_ollama_http(model, prompt)
    verdict = parse_verdict(raw)

    return {
        "id": item.get("id", item_idx),
        "image": item.get("image", ""),
        "question": item.get("question", ""),
        "prediction": pred,
        "gt_answer": gt,
        "verdict": verdict,
        "judge_raw": raw,
    }


def evaluate_with_judge(pred_json_path, model, out_path):
    data = json.load(open(pred_json_path))

    results = []
    total = 0
    correct = 0
    incorrect = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = []
        # submit all jobs
        for idx, item in enumerate(data):
            futures.append(ex.submit(judge_one, idx, item, model))

        # track progress with tqdm
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
            res = fut.result()
            if res is None:
                continue

            # print(
            #     f"\nID: {res['id']}"
            #     f"GT: {res['gt_answer']}"
            #     f"Pred: {res['prediction']}"
            #     f"Verdict: {res['verdict']}"
            # )

            results.append(res)
            total += 1
            if res["verdict"] == "correct":
                correct += 1
            else:
                incorrect += 1
            
            if total % 150 == 0:
                tqdm.write(f"accuracy so far: {correct}/{total} = {correct/total:.4f}")


    # write summary to file
    with open("llm_judge_epoch1_eval_results.txt", "w") as f:
        f.write(f"Total evaluated: {total}\n")
        f.write(f"correct:   {correct}\n")
        f.write(f"incorrect: {incorrect}\n")

        correct_rate = correct / total if total else 0.0
        f.write(f"\nCorrect rate: {correct_rate:.4f}\n")

    print("\n===== LLM Judge Summary =====")
    print(f"Total evaluated: {total}")
    print(f"correct:   {correct}")
    print(f"incorrect: {incorrect}")
    if total:
        print(f"\nCorrect rate: {correct/total:.4f}")

    # write per example results
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved per example results to {out_path}")


def main():
    evaluate_with_judge(
        # pred_json_path="pmc_vqa_llava_baseline_predictions.json",
        pred_json_path="pmc_vqa_llava_epoch1_predictions.json",
        model=DEFAULT_MODEL,
        # out_path="llm_judge_per_example_results.json",
        out_path="llm_judge_epoch1_per_example_results.json",
    )

if __name__ == "__main__":
    main()
