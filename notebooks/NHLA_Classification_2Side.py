from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notebooks.NHLA_Classification import DEFAULT_CANVAS, DEFAULT_IMAGE_SCALE, run_optuna_tuning, train_model


def train_2side(
    csv_path: str,
    output_dir: str,
    epochs: int = 12,
    batch_size: int = 8,
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
    image_size: int = 224,
    num_workers: int = 2,
    cuda_device_index: int = 0,
    tune_subset_fraction: float = 1.0,
    image_cache_dir: str | None = None,
    image_scale: float = DEFAULT_IMAGE_SCALE,
    canvas: tuple = DEFAULT_CANVAS,
    two_side_fusion: str = "shared",
    init_checkpoint: str | None = None,
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
            mode="board",
            output_root=str(trial_root),
            n_trials=optuna_trials,
            epochs=tune_epochs,
            seed=seed,
            val_split=val_split,
            pretrained=pretrained,
            search_space=search_space,
            study_name=study_name,
            db_path=db_path,
            image_size=image_size,
            num_workers=num_workers,
            cuda_device_index=cuda_device_index,
            subset_fraction=tune_subset_fraction,
            image_cache_dir=image_cache_dir,
            image_scale=image_scale,
            canvas=canvas,
            two_side_fusion=two_side_fusion,
            init_checkpoint=init_checkpoint,
        )
        print(f"Best 2-side Optuna params: {best_params}")

    result = train_model(
        csv_path=csv_path,
        output_dir=str(output_root),
        mode="board",
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
        image_size=image_size,
        num_workers=num_workers,
        cuda_device_index=cuda_device_index,
        image_cache_dir=image_cache_dir,
        image_scale=image_scale,
        canvas=canvas,
        two_side_fusion=two_side_fusion,
        init_checkpoint=init_checkpoint,
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
    parser = argparse.ArgumentParser(description="Train a dual-face NHLA board model with Optuna tuning before final training.")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to nhla_board_labels.csv")
    parser.add_argument("--output-dir", type=str, default="models/NHLA_Classification_2Side", help="Directory for final model artifacts")
    parser.add_argument("--epochs", type=int, default=12, help="Final training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Initial batch size used for tuning")
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
    parser.add_argument("--image-size", type=int, default=224, help="Input image size (square)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--cuda-device-index", type=int, default=0, help="CUDA device index to use (0, 1, ...) when CUDA is available")
    parser.add_argument("--tune-subset-fraction", type=float, default=1.0, help="Fraction of boards (label-stratified) used for Optuna trials")
    parser.add_argument("--image-cache-dir", type=str, default=None, help="Root for the decoded image cache (default ~/.cache/nhla_images or $NHLA_CACHE_DIR)")
    parser.add_argument("--image-scale", type=float, default=DEFAULT_IMAGE_SCALE, help="Fixed resize factor applied to every face (native is 1 mm/px along the board)")
    parser.add_argument("--canvas", type=int, nargs=2, default=list(DEFAULT_CANVAS), metavar=("HEIGHT", "WIDTH"), help="Padded input size")
    parser.add_argument("--two-side-fusion", type=str, default="shared", choices=["shared", "stacked", "channels"], help="How the board model combines the two faces")
    parser.add_argument("--init-checkpoint", type=str, default=None, help="1-side model_best.pt to initialise the shared backbone from")
    args = parser.parse_args()

    output = train_2side(
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
        image_size=args.image_size,
        num_workers=args.num_workers,
        cuda_device_index=args.cuda_device_index,
        tune_subset_fraction=args.tune_subset_fraction,
        image_cache_dir=args.image_cache_dir,
        image_scale=args.image_scale,
        canvas=tuple(args.canvas),
        two_side_fusion=args.two_side_fusion,
        init_checkpoint=args.init_checkpoint,
    )
    print(json.dumps({"best_params": output["best_params"], "metrics": output["result"]["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
