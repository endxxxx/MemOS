import argparse
import json

import numpy as np


def save_to_excel(results, output_path):
    combined_data = []
    overall_row = {"category": "overall"}
    overall_row["llm_judge_score"] = results["metrics"]["llm_judge_score"]
    overall_row["llm_judge_std"] = results["metrics"]["llm_judge_std"]
    for metric, value in results["metrics"]["lexical"].items():
        overall_row[metric] = value
    for metric, value in results["metrics"]["semantic"].items():
        overall_row[metric] = value
    overall_row["context_tokens"] = results["metrics"]["context_tokens"]
    for metric, value in results["metrics"]["duration"].items():
        overall_row[metric] = value
    combined_data.append(overall_row)

    print(f"Excel file saved to: {output_path}")


def calculate_scores(data, grade_path, output_path):
    overall_metrics = {
        "lexical": {
            m: []
            for m in [
                "f1",
                "rouge1_f",
                "rouge2_f",
                "rougeL_f",
                "bleu1",
                "bleu2",
                "bleu3",
                "bleu4",
                "meteor",
            ]
        },
        "semantic": {m: [] for m in ["bert_f1", "similarity"]},
        "context_tokens": [],
        "duration": {
            m: [] for m in ["response_duration_ms", "search_duration_ms", "total_duration_ms"]
        },
    }
    all_judgment_keys = set()
    judgment_run_scores = {}

    for _user_id, user_responses in data.items():
        for q in user_responses:
            if "llm_judgments" in q:
                all_judgment_keys.update(q["llm_judgments"].keys())
    for k in all_judgment_keys:
        judgment_run_scores[k] = []

    for _user_id, user_responses in data.items():
        for q in user_responses:
            if "llm_judgments" in q:
                for k, v in q["llm_judgments"].items():
                    score = 1 if v else 0
                    judgment_run_scores[k].append(score)
            nlp = q.get("nlp_metrics", {})
            for m in overall_metrics["lexical"]:
                v = nlp.get("lexical", {}).get(m)
                if v is not None:
                    overall_metrics["lexical"][m].append(v)
            for m in overall_metrics["semantic"]:
                v = nlp.get("semantic", {}).get(m)
                if v is not None:
                    overall_metrics["semantic"][m].append(v)
            ct = nlp.get("context_tokens")
            if ct is not None:
                overall_metrics["context_tokens"].append(ct)
            for m in overall_metrics["duration"]:
                v = q.get(m)
                if v is not None:
                    overall_metrics["duration"][m].append(v)

    judgment_run_averages = [np.mean(scores) for scores in judgment_run_scores.values() if scores]
    llm_judge_score = np.mean(judgment_run_averages) if judgment_run_averages else 0.0
    llm_judge_std = np.std(judgment_run_averages) if len(judgment_run_averages) > 1 else 0.0

    overall_metric_averages = {
        "llm_judge_score": llm_judge_score,
        "llm_judge_std": llm_judge_std,
        "lexical": {},
        "semantic": {},
        "context_tokens": 0.0,
        "duration": {},
    }
    for group in ["lexical", "semantic"]:
        for m in overall_metrics[group]:
            vals = overall_metrics[group][m]
            overall_metric_averages[group][m] = np.mean(vals) if vals else 0.0
    overall_metric_averages["context_tokens"] = (
        np.mean(overall_metrics["context_tokens"]) if overall_metrics["context_tokens"] else 0.0
    )
    for m in list(overall_metrics["duration"].keys()):
        vals = overall_metrics["duration"][m]
        if vals:
            overall_metric_averages["duration"][m] = np.mean(vals)
            overall_metric_averages["duration"][f"{m}_p50"] = np.percentile(vals, 50)
            overall_metric_averages["duration"][f"{m}_p95"] = np.percentile(vals, 95)
        else:
            overall_metric_averages["duration"][m] = 0.0
            overall_metric_averages["duration"][f"{m}_p50"] = 0.0
            overall_metric_averages["duration"][f"{m}_p95"] = 0.0

    results = {
        "metrics": overall_metric_averages,
    }
    with open(grade_path, "w") as outfile:
        json.dump(results, outfile, indent=4)
    save_to_excel(results, output_path)

    print("\n" + "=" * 80)
    print("📊 \033[1;36mMETRIC CALCULATION SUMMARY\033[0m".center(80))
    print("=" * 80)
    total = sum(len(responses) for responses in data.values())
    print(
        f"🤖 \033[1mLLM-as-a-Judge score:\033[0m \033[92m{results['metrics']['llm_judge_score']:.4f}\033[0m ± \033[93m{results['metrics']['llm_judge_std']:.4f}\033[0m"
    )
    print(f"📋 \033[1mTotal questions evaluated:\033[0m \033[93m{total}\033[0m")
    print("-" * 80)
    print("⏱️  \033[1mDuration Metrics (ms):\033[0m")
    for m in ["response_duration_ms", "search_duration_ms", "total_duration_ms"]:
        print(
            f"   \033[94m{m:<22}\033[0m (avg): \033[92m{results['metrics']['duration'][m]:.2f}\033[0m"
            f" | (P50): \033[96m{results['metrics']['duration'][f'{m}_p50']:.2f}\033[0m"
            f" | (P95): \033[91m{results['metrics']['duration'][f'{m}_p95']:.2f}\033[0m"
        )
    print("-" * 80)
    print(f"📁 \033[1mResults written to:\033[0m \033[1;94m{grade_path}\033[0m")
    print(f"📊 \033[1mExcel report saved to:\033[0m \033[1;94m{output_path}\033[0m")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LongMemeval Analysis Eval Metric Script for Openclaw")
    parser.add_argument(
        "--client",
        type=str,
        choices=["openclaw", "openclaw_memos"],
        default="openclaw_memos",
    )
    args = parser.parse_args()
    client = args.client
    judged_path = f"results/lme/{client}/{client}_lme_judged.json"
    grade_path = f"results/lme/{client}/{client}_lme_grades.json"
    output_path = f"results/lme/{client}/{client}_lme_results.xlsx"
    with open(judged_path) as file:
        data = json.load(file)
    calculate_scores(data, grade_path, output_path)
