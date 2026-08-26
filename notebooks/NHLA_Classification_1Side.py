from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notebooks.NHLA_Classification import run_optuna_tuning, train_model


def train_1side(
    csv_path: str,
    output_dir: str,
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.2,
    seed: int = 42,
    optuna_trials: int = 5,
    tune_epochs: int = 4,
    encoder_name: str = "resnet18",
    pretrained: bool = True,
    search_space: dict | None = None,
    study_name: str | None = None,
    db_path: str | None = None,
    checkpoint_name: str | None = None,
    log_name: str | None = None,
    metrics_run_name: str | None = None,
    tensorboard_dir: str | None = None,
    metrics_dir: str | None = None,
    checkpoint_interval: int = 5,
):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    best_params = {
        "lr": float(lr),
        "batch_size": int(batch_size),
        "weight_decay": float(weight_decay),
    }

    if optuna_trials > 0:
        trial_root = Path(db_path).parent if db_path else ROOT / "logs" / "optuna"
        trial_root.mkdir(parents=True, exist_ok=True)
        study, best_params = run_optuna_tuning(
            csv_path=csv_path,
            mode="face",
            output_root=str(trial_root),
            n_trials=optuna_trials,
            epochs=tune_epochs,
            seed=seed,
            val_split=val_split,
            pretrained=pretrained,
            search_space=search_space,
            study_name=study_name,
            db_path=db_path,
        )
        print(f"Best 1-side Optuna params: {best_params}")

    result = train_model(
        csv_path=csv_path,
        output_dir=str(output_root),
        mode="face",
        epochs=epochs,
        batch_size=int(best_params["batch_size"]),
        learning_rate=float(best_params["lr"]),
        weight_decay=float(best_params["weight_decay"]),
        seed=seed,
        val_split=val_split,
        use_optuna=optuna_trials > 0,
        optuna_trials=optuna_trials,
        encoder_name=best_params.get("encoder_name", encoder_name),
        pretrained=pretrained,
        checkpoint_name=checkpoint_name,
        log_name=log_name,
        metrics_run_name=metrics_run_name,
        tensorboard_dir=tensorboard_dir,
        metrics_dir=metrics_dir,
        checkpoint_interval=checkpoint_interval,
        save_trial_artifacts=True,
    )

    params_path = Path(metrics_dir) / "best_params.json" if metrics_dir is not None else output_root / "best_params.json"
    with params_path.open("w") as fh:
        json.dump(best_params, fh, indent=2)

    return {
        "study": locals().get("study"),
        "best_params": best_params,
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser(description="Train a single-side NHLA classification model with Optuna tuning before final training.")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to nhla_face_labels.csv")
    parser.add_argument("--output-dir", type=str, default="models/NHLA_Classification_1Side", help="Directory for final model artifacts")
    parser.add_argument("--epochs", type=int, default=12, help="Final training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Initial batch size used for tuning")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate used for tuning")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Initial weight decay used for tuning")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--optuna-trials", type=int, default=5, help="Number of Optuna trials before final training")
    parser.add_argument("--tune-epochs", type=int, default=4, help="Short training epochs during tuning")
    parser.add_argument("--encoder-name", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"], help="Backbone encoder to use")
    parser.add_argument("--pretrained", dest="pretrained", action=argparse.BooleanOptionalAction, default=True, help="Use pretrained encoder weights")
    parser.add_argument("--tensorboard-dir", type=str, default=None, help="Optional directory for TensorBoard logs")
    parser.add_argument("--metrics-dir", type=str, default=None, help="Optional directory for metrics and CSV/PNG outputs")
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="Save periodic checkpoints every N epochs")
    args = parser.parse_args()

    output = train_1side(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        seed=args.seed,
        optuna_trials=args.optuna_trials,
        tune_epochs=args.tune_epochs,
        encoder_name=args.encoder_name,
        pretrained=args.pretrained,
        tensorboard_dir=args.tensorboard_dir,
        metrics_dir=args.metrics_dir,
        checkpoint_interval=args.checkpoint_interval,
    )
    print(json.dumps({"best_params": output["best_params"], "metrics": output["result"]["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
