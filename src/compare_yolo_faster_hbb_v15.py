"""Compare YOLO-OBB et Faster R-CNN avec une evaluation HBB commune.

Le calcul lourd reste hors du notebook. Les predictions sont sauvegardees
par blocs afin qu'une relance reprenne les images deja terminees.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

from train_faster_rcnn_v15 import (
    DOTAHBBDataset,
    build_coco_ground_truth,
    build_model,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "outputs" / "dota_v15_tiles_1024"
METADATA_DIR = DATASET_DIR / "metadata"
YOLO_CHECKPOINT = (
    PROJECT_DIR
    / "outputs"
    / "remote_checkpoints"
    / "rtx5080_best_v15_current.pt"
)
FASTER_RUN_DIR = (
    PROJECT_DIR
    / "outputs"
    / "faster_rcnn_runs"
    / "faster_rcnn_v15_final_72h"
)
FASTER_CHECKPOINT = FASTER_RUN_DIR / "checkpoints" / "best.pt"

OUTPUT_ROOT = PROJECT_DIR / "outputs" / "model_comparison"
FULL_RUN_NAME = "q3_yolo_faster_hbb_v15_conf001"
SMOKE_RUN_NAME = "q3_yolo_faster_hbb_v15_conf001_smoke"

EXPECTED_YOLO_SHA256 = (
    "a2f864cd3bc7101250ea91a1381b41b6a072003fc18cc61277ff05c52d1da084"
)
EXPECTED_FASTER_SHA256 = (
    "70193cd8d35b93dc0885f14f0436de12beee22d175324d4fe032d169d79d7fc0"
)
EXPECTED_VALIDATION_TILES = 5086
EXPECTED_VALIDATION_OBJECTS = 123901
EXPECTED_BACKGROUND_TILES = 1675
CLASS_COUNT = 16
IMAGE_SIZE = 1024

# Un seuil tres bas conserve presque toute la courbe precision-rappel requise
# pour calculer l'AP. Un seuil de 0,05 serait utile pour afficher des objets,
# mais il supprimerait ici des predictions avant que COCOeval puisse les classer.
SCORE_THRESHOLD = 0.001

# DOTA contient jusqu'a 2 307 objets sur une tuile de validation. Le plafond
# 3 000 couvre donc les annotations et correspond aussi au nombre maximal de
# propositions conservees par le RPN de notre Faster R-CNN.
MAX_DETECTIONS = 3000
YOLO_NMS_IOU = 0.70
FASTER_RPN_NMS_IOU = 0.70
FASTER_ROI_NMS_IOU = 0.50
CHUNK_IMAGES = 25
PROTOCOL_VERSION = "q3-common-hbb-v2-conf001"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def installed_version(package_name: str) -> str:
    """Retourne la version exacte, ou 'unknown' si elle est indisponible."""

    try:
        return package_version(package_name)
    except PackageNotFoundError:
        return "unknown"


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Calcule une empreinte sans charger tout le fichier en RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    """Evite de laisser un JSON partiel si le processus est interrompu."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, ensure_ascii=True)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def atomic_write_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        table.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def normalize_names(names) -> list[str]:
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    return [str(name) for name in names]


def load_class_mapping() -> pd.DataFrame:
    mapping_path = METADATA_DIR / "class_mapping.csv"
    mapping = pd.read_csv(mapping_path).sort_values("yolo_class_id")
    if len(mapping) != CLASS_COUNT:
        raise ValueError("Le mapping doit contenir 16 classes.")
    if mapping["yolo_class_id"].astype(int).tolist() != list(range(16)):
        raise ValueError("Les identifiants YOLO ne couvrent pas 0..15.")
    if mapping["faster_rcnn_class_id"].astype(int).tolist() != list(
        range(1, 17)
    ):
        raise ValueError("Les identifiants Faster R-CNN ne couvrent pas 1..16.")
    return mapping.reset_index(drop=True)


def validation_statistics(dataset: DOTAHBBDataset) -> dict:
    annotation_counts = []
    for tile_id in dataset.tiles["tile_id"].astype(str):
        boxes, _ = dataset.annotations.get(
            tile_id,
            (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            ),
        )
        annotation_counts.append(len(boxes))
    counts = np.asarray(annotation_counts, dtype=np.int64)
    return {
        "tiles": int(len(dataset)),
        "objects": int(counts.sum()),
        "background_tiles": int(np.sum(counts == 0)),
        "positive_tiles": int(np.sum(counts > 0)),
        "max_objects_per_tile": int(counts.max(initial=0)),
        "annotation_counts": counts,
    }


def validate_inputs(load_checkpoints: bool = True) -> dict:
    required = [
        YOLO_CHECKPOINT,
        FASTER_CHECKPOINT,
        METADATA_DIR / "tile_manifest.csv",
        METADATA_DIR / "object_annotations.csv",
        METADATA_DIR / "class_mapping.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Fichiers absents : " + ", ".join(missing))

    yolo_sha = file_sha256(YOLO_CHECKPOINT)
    faster_sha = file_sha256(FASTER_CHECKPOINT)
    if yolo_sha != EXPECTED_YOLO_SHA256:
        raise ValueError("Le checkpoint YOLO n'est pas l'epoque 70 figee.")
    if faster_sha != EXPECTED_FASTER_SHA256:
        raise ValueError("Le checkpoint Faster R-CNN best a change.")

    mapping = load_class_mapping()
    dataset = DOTAHBBDataset(DATASET_DIR, "validation")
    statistics = validation_statistics(dataset)
    if statistics["tiles"] != EXPECTED_VALIDATION_TILES:
        raise ValueError("Le nombre de tuiles validation a change.")
    if statistics["objects"] != EXPECTED_VALIDATION_OBJECTS:
        raise ValueError("Le nombre d'objets validation a change.")
    if statistics["background_tiles"] != EXPECTED_BACKGROUND_TILES:
        raise ValueError("Le nombre de tuiles de fond a change.")

    if load_checkpoints:
        yolo_checkpoint = torch.load(
            YOLO_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )
        if int(yolo_checkpoint["epoch"]) + 1 != 70:
            raise ValueError("Le checkpoint YOLO n'est pas l'epoque 70.")
        yolo_names = normalize_names(yolo_checkpoint["ema"].names)
        expected_names = mapping["class_name"].astype(str).tolist()
        if yolo_names != expected_names:
            raise ValueError("Les classes du checkpoint YOLO ont change.")

        faster_checkpoint = torch.load(
            FASTER_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )
        progress = faster_checkpoint.get("progress", {})
        if int(progress.get("best_epoch", -1)) != 12:
            raise ValueError("Le checkpoint Faster R-CNN n'est pas best epoch 12.")
        if "model_state" not in faster_checkpoint:
            raise ValueError("Les poids Faster R-CNN sont absents.")
        del yolo_checkpoint, faster_checkpoint
        gc.collect()

    return {
        "dataset": dataset,
        "mapping": mapping,
        "statistics": statistics,
        "yolo_sha256": yolo_sha,
        "faster_sha256": faster_sha,
    }


def representative_smoke_indices(
    annotation_counts: np.ndarray,
    requested: int,
) -> list[int]:
    """Garde des fonds et plusieurs niveaux de densite."""

    requested = max(4, min(int(requested), len(annotation_counts)))
    background = np.flatnonzero(annotation_counts == 0)
    positive = np.flatnonzero(annotation_counts > 0)
    positive = positive[np.argsort(annotation_counts[positive], kind="stable")]

    background_count = min(len(background), max(2, requested // 4))
    positive_count = min(len(positive), requested - background_count)

    def quantile_pick(values: np.ndarray, count: int) -> list[int]:
        if count <= 0:
            return []
        positions = np.linspace(0, len(values) - 1, count).round().astype(int)
        return values[positions].astype(int).tolist()

    selected = set(quantile_pick(background, background_count))
    selected.update(quantile_pick(positive, positive_count))
    for index in range(len(annotation_counts)):
        if len(selected) >= requested:
            break
        selected.add(index)
    return sorted(selected)


def selected_image_ids(
    mode: str,
    statistics: dict,
    smoke_images: int,
) -> list[int]:
    if mode == "run":
        return list(range(statistics["tiles"]))
    return representative_smoke_indices(
        statistics["annotation_counts"],
        smoke_images,
    )


def build_protocol(
    mode: str,
    selected_ids: list[int],
    inputs: dict,
) -> dict:
    protocol = {
        "version": PROTOCOL_VERSION,
        "mode": mode,
        "selected_image_ids": selected_ids,
        "selected_image_ids_sha256": json_sha256(selected_ids),
        "selected_tiles": len(selected_ids),
        "validation_tiles": inputs["statistics"]["tiles"],
        "validation_objects": inputs["statistics"]["objects"],
        "background_tiles": inputs["statistics"]["background_tiles"],
        "yolo_checkpoint": str(YOLO_CHECKPOINT.relative_to(PROJECT_DIR)),
        "yolo_sha256": inputs["yolo_sha256"],
        "faster_checkpoint": str(FASTER_CHECKPOINT.relative_to(PROJECT_DIR)),
        "faster_sha256": inputs["faster_sha256"],
        "tile_manifest_sha256": file_sha256(
            METADATA_DIR / "tile_manifest.csv"
        ),
        "annotations_sha256": file_sha256(
            METADATA_DIR / "object_annotations.csv"
        ),
        "class_mapping_sha256": file_sha256(
            METADATA_DIR / "class_mapping.csv"
        ),
        "image_size": IMAGE_SIZE,
        "score_threshold": SCORE_THRESHOLD,
        "max_detections": MAX_DETECTIONS,
        "yolo_native_obb_nms_iou": YOLO_NMS_IOU,
        "yolo_conversion": "minmax des 4 sommets OBB, sans seconde NMS HBB",
        "yolo_inference_precision": "float32",
        "faster_rpn_hbb_nms_iou": FASTER_RPN_NMS_IOU,
        "faster_roi_hbb_nms_iou": FASTER_ROI_NMS_IOU,
        "faster_inference_precision": "float16 autocast sur CUDA",
        "post_conversion_common_nms": False,
        "nms_comparability_note": (
            "Chaque modele conserve sa NMS native; seule la geometrie de "
            "sortie et l'evaluation COCO HBB sont communes."
        ),
        "coco_max_dets": [1, 100, MAX_DETECTIONS],
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": installed_version("torchvision"),
        "ultralytics_version": installed_version("ultralytics"),
        "pycocotools_version": installed_version("pycocotools"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
    protocol["protocol_id"] = json_sha256(protocol)
    return protocol


def prepare_output_dir(mode: str, protocol: dict) -> Path:
    run_name = FULL_RUN_NAME if mode == "run" else SMOKE_RUN_NAME
    output_dir = OUTPUT_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    if protocol_path.is_file():
        existing = read_json(protocol_path)
        if existing != protocol:
            raise RuntimeError(
                "Le protocole existant differe. Ne melangez pas deux evaluations."
            )
    else:
        atomic_write_json(protocol_path, protocol)
    return output_dir


def update_status(output_dir: Path, **changes) -> None:
    path = output_dir / "status.json"
    status = read_json(path) if path.is_file() else {}
    status.update(changes)
    status["updated_at_utc"] = utc_now()
    atomic_write_json(path, status)


def chunk_directory(output_dir: Path, model_name: str) -> Path:
    path = output_dir / "chunks" / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_model_records(
    output_dir: Path,
    model_name: str,
    protocol_id: str,
) -> dict[int, dict]:
    """Charge uniquement les blocs atomiques valides du modele."""

    records: dict[int, dict] = {}
    for path in sorted(chunk_directory(output_dir, model_name).glob("*.json")):
        payload = read_json(path)
        if payload.get("protocol_id") != protocol_id:
            raise ValueError(f"Mauvais protocole dans {path}.")
        if payload.get("model") != model_name:
            raise ValueError(f"Mauvais modele dans {path}.")
        for record in payload.get("records", []):
            image_id = int(record["image_id"])
            if image_id in records:
                raise ValueError(f"Image dupliquee dans les blocs : {image_id}.")
            records[image_id] = record
    return records


def save_model_chunk(
    output_dir: Path,
    model_name: str,
    protocol_id: str,
    records: list[dict],
) -> None:
    image_ids = [int(record["image_id"]) for record in records]
    filename = f"images_{min(image_ids):06d}_{max(image_ids):06d}.json"
    path = chunk_directory(output_dir, model_name) / filename
    payload = {
        "protocol_id": protocol_id,
        "model": model_name,
        "created_at_utc": utc_now(),
        "image_ids": image_ids,
        "records": records,
    }
    if path.exists():
        if read_json(path) != payload:
            raise RuntimeError(f"Le bloc {path} existe deja avec un autre contenu.")
        return
    atomic_write_json(path, payload)


def yolo_result_to_record(result, image_id: int, seconds: float) -> dict:
    predictions = []
    obb = result.obb
    if obb is not None and len(obb) > 0:
        vertices = obb.xyxyxyxy.detach().cpu().numpy().astype(np.float64)
        scores = obb.conf.detach().cpu().numpy().astype(np.float64)
        classes = obb.cls.detach().cpu().numpy().astype(np.int64)
        if not np.isfinite(vertices).all() or not np.isfinite(scores).all():
            raise ValueError(
                f"YOLO a produit une valeur non finie pour l'image {image_id}."
            )
        if np.any((classes < 0) | (classes >= CLASS_COUNT)):
            raise ValueError(
                f"YOLO a produit une classe hors de 0..15 pour l'image {image_id}."
            )
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError(
                f"YOLO a produit un score hors de [0, 1] pour l'image {image_id}."
            )
        x1 = np.clip(vertices[:, :, 0].min(axis=1), 0.0, IMAGE_SIZE)
        y1 = np.clip(vertices[:, :, 1].min(axis=1), 0.0, IMAGE_SIZE)
        x2 = np.clip(vertices[:, :, 0].max(axis=1), 0.0, IMAGE_SIZE)
        y2 = np.clip(vertices[:, :, 1].max(axis=1), 0.0, IMAGE_SIZE)
        for left, top, right, bottom, score, class_id in zip(
            x1, y1, x2, y2, scores, classes
        ):
            width = float(right - left)
            height = float(bottom - top)
            if not np.isfinite([left, top, right, bottom]).all():
                raise ValueError(
                    f"HBB YOLO non finie pour l'image {image_id}."
                )
            if width <= 0.0 or height <= 0.0:
                continue
            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(class_id) + 1,
                    "bbox": [float(left), float(top), width, height],
                    "score": float(score),
                }
            )
    return {
        "image_id": int(image_id),
        "seconds": float(seconds),
        "detections": len(predictions),
        "predictions": predictions,
    }


def faster_prediction_to_record(
    prediction: dict,
    image_id: int,
    seconds: float,
) -> dict:
    boxes = prediction["boxes"].detach().cpu().numpy().astype(np.float64)
    labels = prediction["labels"].detach().cpu().numpy().astype(np.int64)
    scores = prediction["scores"].detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError(
            f"Faster R-CNN a produit une valeur non finie pour l'image {image_id}."
        )
    if np.any((labels < 1) | (labels > CLASS_COUNT)):
        raise ValueError(
            "Faster R-CNN a produit une classe hors de 1..16 "
            f"pour l'image {image_id}."
        )
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError(
            "Faster R-CNN a produit un score hors de [0, 1] "
            f"pour l'image {image_id}."
        )
    predictions = []
    for box, label, score in zip(boxes, labels, scores):
        left, top, right, bottom = np.clip(box, 0.0, IMAGE_SIZE)
        width = float(right - left)
        height = float(bottom - top)
        if not np.isfinite([left, top, right, bottom]).all():
            raise ValueError(
                f"HBB Faster R-CNN non finie pour l'image {image_id}."
            )
        if width <= 0.0 or height <= 0.0:
            continue
        predictions.append(
            {
                "image_id": int(image_id),
                "category_id": int(label),
                "bbox": [float(left), float(top), width, height],
                "score": float(score),
            }
        )
    return {
        "image_id": int(image_id),
        "seconds": float(seconds),
        "detections": len(predictions),
        "predictions": predictions,
    }


def iter_chunks(values: list[int], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def run_yolo_predictions(
    dataset: DOTAHBBDataset,
    selected_ids: list[int],
    output_dir: Path,
    protocol: dict,
    device: str,
) -> None:
    model_name = "yolo"
    existing = load_model_records(
        output_dir, model_name, protocol["protocol_id"]
    )
    pending = [image_id for image_id in selected_ids if image_id not in existing]
    if not pending:
        print("YOLO : toutes les tuiles sont deja sauvegardees.")
        return

    print(f"YOLO : {len(pending)} tuile(s) a traiter.")
    model = YOLO(str(YOLO_CHECKPOINT), task="obb")
    if normalize_names(model.names) != load_class_mapping()[
        "class_name"
    ].astype(str).tolist():
        raise ValueError("Les classes YOLO chargees sont incoherentes.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    completed = len(existing)
    for ids in iter_chunks(pending, CHUNK_IMAGES):
        chunk_records = []
        for image_id in ids:
            row = dataset.tiles.iloc[int(image_id)]
            image_path = DATASET_DIR / row["tile_image_path"]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = model.predict(
                source=str(image_path),
                imgsz=IMAGE_SIZE,
                conf=SCORE_THRESHOLD,
                iou=YOLO_NMS_IOU,
                max_det=MAX_DETECTIONS,
                device=device,
                verbose=False,
            )[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            chunk_records.append(
                yolo_result_to_record(result, image_id, elapsed)
            )
        save_model_chunk(
            output_dir,
            model_name,
            protocol["protocol_id"],
            chunk_records,
        )
        completed += len(chunk_records)
        update_status(
            output_dir,
            status="running",
            phase="yolo",
            model="YOLO11s-OBB vers HBB",
            processed=completed,
            total=len(selected_ids),
            percent=100.0 * completed / len(selected_ids),
        )
        print(f"YOLO : {completed}/{len(selected_ids)}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_faster_predictions(
    dataset: DOTAHBBDataset,
    selected_ids: list[int],
    output_dir: Path,
    protocol: dict,
    device: str,
) -> None:
    model_name = "faster_rcnn"
    existing = load_model_records(
        output_dir, model_name, protocol["protocol_id"]
    )
    pending = [image_id for image_id in selected_ids if image_id not in existing]
    if not pending:
        print("Faster R-CNN : toutes les tuiles sont deja sauvegardees.")
        return

    print(f"Faster R-CNN : {len(pending)} tuile(s) a traiter.")
    device_object = torch.device(
        f"cuda:{device}" if str(device).isdigit() else str(device)
    )
    checkpoint = torch.load(
        FASTER_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    model = build_model(load_coco_weights=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.roi_heads.score_thresh = SCORE_THRESHOLD
    model.roi_heads.detections_per_img = MAX_DETECTIONS
    if not np.isclose(model.roi_heads.nms_thresh, FASTER_ROI_NMS_IOU):
        raise ValueError("Le seuil NMS ROI de Faster R-CNN a change.")
    if not np.isclose(model.rpn.nms_thresh, FASTER_RPN_NMS_IOU):
        raise ValueError("Le seuil NMS RPN de Faster R-CNN a change.")
    model.to(device_object).eval()
    del checkpoint
    gc.collect()
    if device_object.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    completed = len(existing)
    for ids in iter_chunks(pending, CHUNK_IMAGES):
        chunk_records = []
        for image_id in ids:
            image, _ = dataset[int(image_id)]
            image_device = image.to(device_object, non_blocking=True)
            if device_object.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                with torch.autocast(
                    device_type=device_object.type,
                    dtype=torch.float16,
                    enabled=device_object.type == "cuda",
                ):
                    prediction = model([image_device])[0]
            if device_object.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            chunk_records.append(
                faster_prediction_to_record(
                    prediction,
                    image_id,
                    elapsed,
                )
            )
            del image_device, prediction
        save_model_chunk(
            output_dir,
            model_name,
            protocol["protocol_id"],
            chunk_records,
        )
        completed += len(chunk_records)
        update_status(
            output_dir,
            status="running",
            phase="faster_rcnn",
            model="Faster R-CNN HBB",
            processed=completed,
            total=len(selected_ids),
            percent=100.0 * completed / len(selected_ids),
        )
        print(f"Faster R-CNN : {completed}/{len(selected_ids)}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mean_valid(values: np.ndarray, empty_value: float = float("nan")) -> float:
    valid = values[values >= 0]
    return float(valid.mean()) if valid.size else float(empty_value)


def evaluate_coco_hbb(
    coco_ground_truth,
    selected_ids: list[int],
    predictions: list[dict],
) -> tuple[dict, pd.DataFrame]:
    if not predictions:
        raise ValueError("Aucune prediction n'est disponible pour COCOeval.")
    coco_predictions = coco_ground_truth.loadRes(predictions)
    evaluator = COCOeval(coco_ground_truth, coco_predictions, "bbox")
    evaluator.params.imgIds = sorted(selected_ids)
    evaluator.params.maxDets = [1, 100, MAX_DETECTIONS]
    evaluator.evaluate()
    evaluator.accumulate()

    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    recall = evaluator.eval["recall"][:, :, 0, -1]
    iou50 = int(
        np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.50))[0]
    )
    iou75 = int(
        np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.75))[0]
    )
    metrics = {
        "mAP50_95": mean_valid(precision),
        "mAP50": mean_valid(precision[iou50]),
        "mAP75": mean_valid(precision[iou75]),
        "average_recall_50_95": mean_valid(recall),
    }

    per_class = []
    for class_position, category_id in enumerate(evaluator.params.catIds):
        per_class.append(
            {
                "category_id": int(category_id),
                "class_name": str(coco_ground_truth.cats[category_id]["name"]),
                "AP50_95": mean_valid(precision[:, :, class_position]),
                "AP50": mean_valid(precision[iou50, :, class_position]),
                "AP75": mean_valid(precision[iou75, :, class_position]),
                "AR50_95": mean_valid(recall[:, class_position]),
            }
        )
    return metrics, pd.DataFrame(per_class)


def flatten_records(records: dict[int, dict], selected_ids: list[int]):
    missing = [image_id for image_id in selected_ids if image_id not in records]
    if missing:
        raise RuntimeError(f"Predictions manquantes : {missing[:10]}")
    predictions = []
    for image_id in selected_ids:
        predictions.extend(records[image_id]["predictions"])
    return predictions


def finalize_results(
    dataset: DOTAHBBDataset,
    selected_ids: list[int],
    output_dir: Path,
    protocol: dict,
) -> pd.DataFrame:
    coco_ground_truth = build_coco_ground_truth(dataset)
    summary_rows = []
    class_tables = []

    selected_set = set(selected_ids)
    ground_truth_objects = sum(
        1
        for annotation in coco_ground_truth.dataset["annotations"]
        if int(annotation["image_id"]) in selected_set
    )

    for model_name, display_name in [
        ("yolo", "YOLO11s-OBB converti en HBB"),
        ("faster_rcnn", "Faster R-CNN HBB"),
    ]:
        records = load_model_records(
            output_dir,
            model_name,
            protocol["protocol_id"],
        )
        predictions = flatten_records(records, selected_ids)
        metrics, per_class = evaluate_coco_hbb(
            coco_ground_truth,
            selected_ids,
            predictions,
        )
        seconds = float(sum(records[index]["seconds"] for index in selected_ids))
        saturated = int(
            sum(
                records[index]["detections"] >= MAX_DETECTIONS
                for index in selected_ids
            )
        )
        if saturated:
            print(
                f"ATTENTION {display_name} : {saturated} tuile(s) ont atteint "
                f"le plafond de {MAX_DETECTIONS} predictions."
            )
        summary_rows.append(
            {
                "model_key": model_name,
                "model": display_name,
                "tiles": len(selected_ids),
                "ground_truth_objects": ground_truth_objects,
                "predictions": len(predictions),
                "tiles_at_max_det": saturated,
                "score_threshold": SCORE_THRESHOLD,
                "max_detections": MAX_DETECTIONS,
                "mAP50_95": metrics["mAP50_95"],
                "mAP50": metrics["mAP50"],
                "mAP75": metrics["mAP75"],
                "average_recall_50_95": metrics[
                    "average_recall_50_95"
                ],
                "indicative_inference_seconds": seconds,
                "indicative_images_per_second": (
                    len(selected_ids) / seconds if seconds > 0 else 0.0
                ),
            }
        )
        per_class.insert(0, "model_key", model_name)
        per_class.insert(1, "model", display_name)
        class_tables.append(per_class)

    summary = pd.DataFrame(summary_rows)
    per_class_summary = pd.concat(class_tables, ignore_index=True)
    atomic_write_csv(output_dir / "metrics_common_hbb.csv", summary)
    atomic_write_csv(
        output_dir / "metrics_common_hbb_per_class.csv",
        per_class_summary,
    )
    atomic_write_json(
        output_dir / "metrics_common_hbb.json",
        {
            "protocol_id": protocol["protocol_id"],
            "created_at_utc": utc_now(),
            "metrics": summary.to_dict(orient="records"),
        },
    )
    update_status(
        output_dir,
        status="completed",
        phase="completed",
        model=None,
        processed=len(selected_ids),
        total=len(selected_ids),
        percent=100.0,
        metrics_path=str(
            (output_dir / "metrics_common_hbb.csv").relative_to(PROJECT_DIR)
        ),
    )
    return summary


def run_pipeline(
    mode: str,
    smoke_images: int = 32,
    device: str = "0",
) -> pd.DataFrame:
    if mode not in {"smoke", "run"}:
        raise ValueError("mode doit valoir smoke ou run.")
    if str(device).lower() != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA est requis pour cette evaluation.")

    inputs = validate_inputs(load_checkpoints=True)
    dataset = inputs["dataset"]
    selected_ids = selected_image_ids(
        mode,
        inputs["statistics"],
        smoke_images,
    )
    protocol = build_protocol(mode, selected_ids, inputs)
    output_dir = prepare_output_dir(mode, protocol)
    update_status(
        output_dir,
        status="running",
        phase="starting",
        mode=mode,
        pid=os.getpid(),
        started_at_utc=utc_now(),
        selected_tiles=len(selected_ids),
        processed=0,
        total=len(selected_ids),
        percent=0.0,
    )

    try:
        run_yolo_predictions(
            dataset,
            selected_ids,
            output_dir,
            protocol,
            device,
        )
        run_faster_predictions(
            dataset,
            selected_ids,
            output_dir,
            protocol,
            device,
        )
        summary = finalize_results(
            dataset,
            selected_ids,
            output_dir,
            protocol,
        )
    except Exception as error:
        update_status(
            output_dir,
            status="error",
            phase="error",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise

    print("\nEvaluation HBB commune terminee :", output_dir)
    print(summary.to_string(index=False))
    if mode == "smoke":
        print(
            "\nSMOKE : ces scores portent sur un petit sous-ensemble et "
            "ne sont pas des resultats scientifiques finaux."
        )
    return summary


def summarize_existing(
    mode: str,
    smoke_images: int = 32,
) -> pd.DataFrame:
    inputs = validate_inputs(load_checkpoints=False)
    selected_ids = selected_image_ids(
        mode,
        inputs["statistics"],
        smoke_images,
    )
    protocol = build_protocol(mode, selected_ids, inputs)
    output_dir = prepare_output_dir(mode, protocol)
    return finalize_results(
        inputs["dataset"],
        selected_ids,
        output_dir,
        protocol,
    )


def check_environment() -> None:
    inputs = validate_inputs(load_checkpoints=True)
    stats = inputs["statistics"]
    print("Environnement valide.")
    print("GPU disponible :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU :", torch.cuda.get_device_name(0))
    print("Tuiles validation :", stats["tiles"])
    print("Objets HBB :", stats["objects"])
    print("Tuiles de fond :", stats["background_tiles"])
    print("Maximum d'objets par tuile :", stats["max_objects_per_tile"])
    print("YOLO : epoque 70, SHA-256 valide")
    print("Faster R-CNN : best epoque 12, SHA-256 valide")
    print("Seuil commun :", SCORE_THRESHOLD)
    print("max_det commun :", MAX_DETECTIONS)
    print(
        "Important : les temps du fichier final resteront indicatifs; "
        "le benchmark de vitesse sera effectue separement."
    )
    print("Aucune inference n'a ete lancee.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation HBB commune YOLO et Faster R-CNN."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Verifie les entrees sans inference.")

    smoke = commands.add_parser(
        "smoke",
        help="Teste le protocole sur un petit sous-ensemble.",
    )
    smoke.add_argument("--images", type=int, default=32)
    smoke.add_argument("--device", default="0")

    run = commands.add_parser(
        "run",
        help="Evalue les 5 086 tuiles avec reprise automatique.",
    )
    run.add_argument("--device", default="0")

    summarize = commands.add_parser(
        "summarize",
        help="Recalcule COCOeval depuis les blocs deja sauvegardes.",
    )
    summarize.add_argument(
        "--mode", choices=["smoke", "run"], default="run"
    )
    summarize.add_argument("--images", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.command == "check":
        check_environment()
    elif arguments.command == "smoke":
        run_pipeline(
            "smoke",
            smoke_images=arguments.images,
            device=arguments.device,
        )
    elif arguments.command == "run":
        run_pipeline("run", device=arguments.device)
    elif arguments.command == "summarize":
        summarize_existing(
            arguments.mode,
            smoke_images=arguments.images,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrompu. Les blocs deja sauvegardes seront repris.")
        raise SystemExit(130)
