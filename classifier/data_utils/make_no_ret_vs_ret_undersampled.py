import argparse
import json
import random
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a balanced A/R Clf1 training file by undersampling the majority class."
    )
    parser.add_argument("--model", required=True, choices=["flan_t5_xl", "flan_t5_xxl", "gpt"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input_path",
        type=Path,
        default=None,
        help="Optional explicit input JSON path. Defaults to classifier/data/.../{model}/silver/no_retrieval_vs_retrieval/train.json",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Optional explicit output JSON path. Defaults to classifier/data/.../{model}/silver/no_retrieval_vs_retrieval/train_undersampled.json",
    )
    return parser.parse_args()


def default_paths(model: str) -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    base = repo_root / "classifier" / "data" / "musique_hotpot_wiki2_nq_tqa_sqd" / model / "silver" / "no_retrieval_vs_retrieval"
    return base / "train.json", base / "train_undersampled.json"


def main() -> None:
    args = parse_args()
    input_path, output_path = default_paths(args.model)
    if args.input_path is not None:
        input_path = args.input_path
    if args.output_path is not None:
        output_path = args.output_path

    with input_path.open() as f:
        data = json.load(f)

    by_label: dict[str, list[dict]] = {"A": [], "R": []}
    for item in data:
        label = item.get("answer")
        if label not in by_label:
            raise ValueError(f"Unexpected label {label!r} in {input_path}")
        by_label[label].append(item)

    minority_size = min(len(by_label["A"]), len(by_label["R"]))
    rng = random.Random(args.seed)

    balanced = list(by_label["R"])
    balanced.extend(rng.sample(by_label["A"], minority_size))
    rng.shuffle(balanced)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(balanced, f, indent=4)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Original counts: {dict(Counter(item['answer'] for item in data))}")
    print(f"Balanced counts: {dict(Counter(item['answer'] for item in balanced))}")


if __name__ == "__main__":
    main()