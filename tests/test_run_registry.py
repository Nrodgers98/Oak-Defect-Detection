import json
import os
from datetime import datetime

import pytest

from notebooks.functions.run_registry import (
    discover_runs,
    headline_metric,
    load_tb_scalars,
    load_trials_df,
    load_ultralytics_results,
    parse_array_cell,
    parse_run_name,
    per_class_metrics,
    scalar_metrics,
)


def _touch(path, content=b""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _write_events(log_dir, tag="IoU/Val", values=(0.2, 0.4, 0.3)):
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(log_dir))
    for step, value in enumerate(values):
        writer.add_scalar(tag, value, step)
    writer.close()


@pytest.mark.parametrize(
    "a, b",
    [
        ("STACK_DeepLabV3_FullImage_Optuna_20260803_140657", "STACK_ResNet34_DeepLabV3_FullImage_Optuna_20260803_140657"),
        ("CC_UNetPlusPlus_FullImage_Optuna_v1", "CC_ResNet34_UnetPlusPlus_FullImage_Optuna"),
        ("STACK_YOLO_L_FullImage_v1", "Stack_YOLO_L_FullImage"),
    ],
)
def test_parse_run_name_normalizes_naming_variants(a, b):
    pa, pb = parse_run_name(a), parse_run_name(b)
    assert pa.prefix == pb.prefix
    assert pa.ts == pb.ts


def test_parse_run_name_timestamps_and_legacy():
    parsed = parse_run_name("NHLA_Classification_2Side_20260825_152507")
    assert parsed.prefix == "nhla_classification_2side"
    assert parsed.ts == datetime(2026, 8, 25, 15, 25, 7)
    assert parse_run_name("YOLO_Pith_Detection_Optuna").ts is None
    assert parse_run_name("CC_DeepLabV3_FullImage_Optuna_v1").ts is None


def test_parse_array_cell_handles_json_and_numpy_strings():
    assert parse_array_cell("[0.1, 0.2]") == [0.1, 0.2]
    assert parse_array_cell("[0.9253012  0.57832396 0.49356097 1e-3]") == pytest.approx([0.9253012, 0.57832396, 0.49356097, 0.001])
    assert parse_array_cell("") is None
    assert parse_array_cell(None) is None


def test_load_ultralytics_results_strips_header_whitespace(tmp_path):
    csv_path = _touch(tmp_path / "results.csv", b"  epoch,  metrics/mAP50-95(B)\n1,0.1\n2,0.3\n")
    df = load_ultralytics_results(csv_path)
    assert list(df.columns) == ["epoch", "metrics/mAP50-95(B)"]


def test_model_and_tensorboard_dirs_link_across_name_drift(tmp_path):
    _touch(tmp_path / "models/STACK_DeepLabV3_FullImage_Optuna_20260803_140657/model_best.pt")
    _touch(tmp_path / "checkpoints/STACK_DeepLabV3_FullImage_Optuna_20260803_140657/training_history.json",
           json.dumps({"train_loss": [1.0, 0.5], "val_loss": [1.1, 0.7], "val_iou": [0.3, 0.45]}).encode())
    # TensorBoard dir has the ResNet34 infix and a timestamp one second later.
    _write_events(tmp_path / "logs/tensorboard/STACK_ResNet34_DeepLabV3_FullImage_Optuna_20260803_140658")
    # A TensorBoard dir without events is ignored rather than creating a phantom run.
    (tmp_path / "logs/tensorboard/Stack_YOLO_L_FullImage").mkdir(parents=True)
    _touch(tmp_path / "models/onnx/stack_deeplabv3_fullimage_optuna_20260803_140657.onnx")

    runs = discover_runs(tmp_path, studies=[])
    assert len(runs) == 1
    run = runs[0]
    assert run.status == "complete"
    assert len(run.tb_dirs) == 1
    assert run.checkpoint_dir is not None
    assert [p.name for p in run.onnx] == ["stack_deeplabv3_fullimage_optuna_20260803_140657.onnx"]
    value, label = headline_metric(run)
    assert value == pytest.approx(0.45)
    assert label == "Best val IoU"


def test_legacy_v1_run_collects_unsuffixed_logs_and_loose_metrics(tmp_path):
    _touch(tmp_path / "models/CC_DeepLabV3_FullImage_Optuna_v1/model_best.pt")
    _touch(tmp_path / "checkpoints/CC_DeepLabV3_FullImage_Optuna/model_epoch_5.pt")
    _write_events(tmp_path / "logs/tensorboard/CC_ResNet34_DeepLabV3_FullImage_Optuna")
    _touch(tmp_path / "data/metrics/cc_deeplabv3_fullimage_optuna_20260717_030109_confusion_matrix.png")

    runs = discover_runs(tmp_path, studies=[])
    assert len(runs) == 1
    run = runs[0]
    assert run.ts is None
    assert run.tb_dirs and run.checkpoint_dir is not None
    assert [p.name for p in run.metrics_files] == ["cc_deeplabv3_fullimage_optuna_20260717_030109_confusion_matrix.png"]


def test_nhla_nested_tensorboard_and_metrics_folder(tmp_path):
    name = "NHLA_Classification_1Side_20260818_195250"
    _touch(tmp_path / f"models/{name}/model_best.pt")
    _write_events(tmp_path / f"logs/tensorboard/{name}/NHLA_1Side_20260818_195250", tag="val/macro_f1")
    metrics_dir = tmp_path / f"data/metrics/{name}"
    _touch(metrics_dir / "nhla_1side_20260818_195250_metrics.json", json.dumps({
        "macro_f1": 0.31,
        "per_class_f1": [0.3, 0.4],
        "classification_report": {"1 Com": {}, "2A Com": {}, "accuracy": 0.4},
    }).encode())

    run = discover_runs(tmp_path, studies=[])[0]
    assert run.family == "nhla"
    assert run.tb_dirs[0].name == "NHLA_1Side_20260818_195250"
    assert headline_metric(run) == (pytest.approx(0.31), "Macro F1")
    pc = per_class_metrics(run)
    assert list(pc["class"]) == ["1 Com", "2A Com"]


def test_tb_scalars_loader(tmp_path):
    _write_events(tmp_path / "tb", tag="Loss/Train", values=(3.0, 2.0))
    df = load_tb_scalars(tmp_path / "tb")
    assert list(df["tag"].unique()) == ["Loss/Train"]
    assert list(df["value"]) == pytest.approx([3.0, 2.0])


def test_optuna_study_links_by_timestamp_and_loads_read_only(tmp_path):
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    db = tmp_path / "logs/optuna/TILED_Optuna.db"
    db.parent.mkdir(parents=True)
    study = optuna.create_study(study_name="Unet_tiled_optuna_fixed_resnet34_20260903_214329",
                                storage=f"sqlite:///{db}", direction="maximize")
    study.optimize(lambda t: t.suggest_float("lr", 1e-5, 1e-2, log=True), n_trials=3)
    del study
    _touch(tmp_path / "models/TILE_Unet_Tiled_Optuna_20260903_214329/model_best.pt")

    before = os.path.getmtime(db)
    run = discover_runs(tmp_path)[0]
    assert run.study is not None
    assert run.study.study_name == "Unet_tiled_optuna_fixed_resnet34_20260903_214329"
    assert run.study.confidence == "exact"
    assert run.study.n_complete == 3

    df = load_trials_df(db, run.study.study_name)
    assert len(df) == 3 and "params_lr" in df.columns
    assert os.path.getmtime(db) == before


def test_class_count_read_from_checkpoint_head_without_loading_weights(tmp_path):
    torch = pytest.importorskip("torch")
    for name, outputs in (("STACK_DeepLabV3_FullImage_Optuna_v1", 4), ("TILE_Unet_Tiled_Optuna_20260903_214329", 5)):
        path = tmp_path / f"models/{name}/model_best.pt"
        path.parent.mkdir(parents=True)
        torch.save({"epoch": 3, "model_state_dict": {
            "encoder.conv1.weight": torch.zeros(8, 3, 3, 3),
            "segmentation_head.0.weight": torch.zeros(outputs, 16, 1, 1),
            "segmentation_head.0.bias": torch.zeros(outputs),
        }}, path)

    runs = {r.name: r for r in discover_runs(tmp_path, studies=[])}
    # Background is excluded from the count.
    assert runs["STACK_DeepLabV3_FullImage_Optuna_v1"].n_classes == 3
    assert runs["TILE_Unet_Tiled_Optuna_20260903_214329"].n_classes == 4
    assert runs["TILE_Unet_Tiled_Optuna_20260903_214329"].class_names is None


def test_summary_row_class_names_and_defect_only_miou(tmp_path):
    name = "TILE_Unet_Tiled_Optuna_20260903_214329"
    _touch(tmp_path / f"models/{name}/model_best.pt")
    summary = tmp_path / "data/metrics/TRAINING_RESULTS_SUMMARY.csv"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        "run_name,checkpoint_path,class_names,mean_iou,iou_per_class\n"
        f'tile_x,{tmp_path}/models/{name}/model_best.pt,"[""Background"", ""Knot"", ""Rot""]",0.6,"[0.9 0.5 0.4]"\n'
    )
    run = discover_runs(tmp_path, studies=[])[0]
    assert run.n_classes == 2
    assert run.class_names == ["Knot", "Rot"]
    assert scalar_metrics(run)["mean_iou_defects"] == pytest.approx(0.45)
