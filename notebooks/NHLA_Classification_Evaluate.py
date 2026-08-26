from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def load_metrics(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r") as fh:
        return json.load(fh)


def compare_models(face_metrics_path: str, board_metrics_path: str):
    face = load_metrics(face_metrics_path)
    board = load_metrics(board_metrics_path)

    summary = {
        "face": {
            "accuracy": face.get("accuracy"),
            "balanced_accuracy": face.get("balanced_accuracy"),
            "macro_f1": face.get("macro_f1"),
            "weighted_f1": face.get("weighted_f1"),
        },
        "board": {
            "accuracy": board.get("accuracy"),
            "balanced_accuracy": board.get("balanced_accuracy"),
            "macro_f1": board.get("macro_f1"),
            "weighted_f1": board.get("weighted_f1"),
        },
    }

    winner = "board" if summary["board"]["macro_f1"] >= summary["face"]["macro_f1"] else "face"
    summary["winner"] = winner
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare the final 1-side and 2-side NHLA model metrics.")
    parser.add_argument("--face-metrics", type=str, required=True, help="Path to the 1-side metrics.json")
    parser.add_argument("--board-metrics", type=str, required=True, help="Path to the 2-side metrics.json")
    args = parser.parse_args()

    summary = compare_models(args.face_metrics, args.board_metrics)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
