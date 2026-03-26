import argparse
import json
import os


def main(file_path):
    # read the json file
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    # category mapping for locomo
    category_mapping = {1: "multi hop", 2: "temporal reasoning", 3: "open domain", 4: "single hop"}

    # initialize variables for statistics
    total_questions = 0
    correct_predictions = 0
    total_f1 = 0

    # statistics for each category
    category_stats = {
        1: {"total": 0, "correct": 0},
        2: {"total": 0, "correct": 0},
        3: {"total": 0, "correct": 0},
        4: {"total": 0, "correct": 0},
    }

    # iterate over all experiment results
    for _exp_name, questions in data.items():
        for question in questions:
            total_questions += 1

            # get category
            category = question["category"]

            # calculate accuracy (true count > false count)
            judgments = question["llm_judgments"]
            true_count = sum(1 for j in judgments.values() if j)
            false_count = sum(1 for j in judgments.values() if not j)

            is_correct = true_count > false_count
            if is_correct:
                correct_predictions += 1

            # update category statistics
            if category in category_stats:
                category_stats[category]["total"] += 1
                if is_correct:
                    category_stats[category]["correct"] += 1

            # accumulate F1 score
            f1_score = question["nlp_metrics"]["lexical"]["f1"]
            total_f1 += f1_score

    # calculate accuracy and average F1 score
    accuracy = correct_predictions / total_questions if total_questions > 0 else 0
    average_f1 = total_f1 / total_questions if total_questions > 0 else 0

    # print statistics for each category and overall
    print("Statistics:")
    print(f"total questions: {total_questions}")
    print(f"correct predictions: {correct_predictions}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"average F1 score: {average_f1:.4f}")
    print()
    print("Statistics by category:")
    print(
        f"{'Category':<10} {'Category Name':<20} {'Questions':<10} {'Correct':<10} {'Accuracy':<10}"
    )
    print("-" * 70)

    for category_code, stats in category_stats.items():
        category_name = category_mapping.get(category_code, f"Unknown({category_code})")
        total = stats["total"]
        correct = stats["correct"]
        category_accuracy = correct / total if total > 0 else 0
        print(
            f"{category_code:<10} {category_name:<20} {total:<10} {correct:<10} {category_accuracy:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OpenCoMo results on LoCoMo dataset")
    parser.add_argument("client_type", type=str, help="Client type, e.g., 'openclaw' or 'mem0'")
    parser.add_argument("version", type=str, help="Version of the evaluation")

    args = parser.parse_args()
    file_path = os.path.join(
        "results", "locomo", f"{args.client_type}-{args.version}", "locomo_judged.json"
    )
    main(file_path)
