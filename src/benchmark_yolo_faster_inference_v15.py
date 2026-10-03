"""Benchmark pratique et commun de YOLO-OBB et Faster R-CNN.

Le script mesure les deux modeles sur les memes images deja chargees en RAM.
Le chargement depuis le disque est donc exclu, tandis que le pretraitement,
l'inference et la NMS propres a chaque modele restent inclus.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from ultralytics import YOLO

from compare_yolo_faster_hbb_v15 import (
    DATASET_DIR,
    EXPECTED_FASTER_SHA256,
    EXPECTED_YOLO_SHA256,
    FASTER_CHECKPOINT,
    FASTER_ROI_NMS_IOU,
    FASTER_RPN_NMS_IOU,
    IMAGE_SIZE,
    MAX_DETECTIONS,
    YOLO_CHECKPOINT,
    YOLO_NMS_IOU,
    file_sha256,
    json_sha256,
    representative_smoke_indices,
    validate_inputs,
)
from train_faster_rcnn_v15 import build_model


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_DIR / "outputs" / "model_comparison"
OFFICIAL_OUTPUT_DIR = OUTPUT_ROOT / "q3_inference_benchmark_200"

PROTOCOL_VERSION = "q3-inference-efficiency-v1"
OFFICIAL_IMAGE_COUNT = 200
OFFICIAL_WARMUP_COUNT = 20
SCORE_THRESHOLD = 0.25
SEED = 2026


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def installed_version(package_name: str) -> str:
    try:
        return package_version(package_name)
    except PackageNotFoundError:
        return "unknown"


def atomic_write_json(path: Path, value: object) -> None:
    """Remplace un JSON seulement lorsque son ecriture est terminee."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, ensure_ascii=True)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        table.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def benchmark_output_dir(image_count: int) -> Path:
    if image_count == OFFICIAL_IMAGE_COUNT:
        return OFFICIAL_OUTPUT_DIR
    return OUTPUT_ROOT / f"q3_inference_benchmark_demo_{image_count}"


def select_representative_images(statistics: dict, image_count: int) -> list[int]:
    """Selection deterministe : fonds, faibles et fortes densites."""

    selected = representative_smoke_indices(
        statistics["annotation_counts"],
        image_count,
    )
    # Melanger le meme ordre pour les deux modeles evite que toutes les scenes
    # denses arrivent au debut ou a la fin d'un seul benchmark.
    generator = np.random.default_rng(SEED)
    generator.shuffle(selected)
    return [int(image_id) for image_id in selected]


def preload_images(dataset, image_ids: list[int]) -> list[Image.Image]:
    """Charge les pixels une fois avant de demarrer le chronometre."""

    images = []
    for image_id in image_ids:
        row = dataset.tiles.iloc[image_id]
        path = DATASET_DIR / str(row["tile_image_path"])
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def latency_summary(latencies_seconds: list[float]) -> dict:
    values_ms = 1000.0 * np.asarray(latencies_seconds, dtype=np.float64)
    total_seconds = float(np.sum(latencies_seconds))
    return {
        "mean_latency_ms": float(values_ms.mean()),
        "median_latency_ms": float(np.median(values_ms)),
        "p95_latency_ms": float(np.percentile(values_ms, 95)),
        "images_per_second": (
            float(len(values_ms) / total_seconds) if total_seconds > 0 else 0.0
        ),
        "measured_seconds": total_seconds,
    }


def memory_summary(device: torch.device, baseline_bytes: int) -> dict:
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    mib = 1024.0**2
    return {
        "model_baseline_memory_mib": baseline_bytes / mib,
        "peak_allocated_memory_mib": peak_allocated / mib,
        "peak_reserved_memory_mib": peak_reserved / mib,
        "runtime_extra_memory_mib": max(0, peak_allocated - baseline_bytes) / mib,
    }


def benchmark_yolo(
    images: list[Image.Image],
    warmup_count: int,
    device_index: str,
) -> dict:
    """Mesure le pipeline YOLO en memoire, y compris pre/post-traitement."""

    device = torch.device(f"cuda:{device_index}")
    model = YOLO(str(YOLO_CHECKPOINT), task="obb")

    # Les repetitions de chauffe creent les buffers CUDA et retirent du
    # chronometrage le cout exceptionnel de la toute premiere prediction.
    for image in images[:warmup_count]:
        warmup_result = model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=SCORE_THRESHOLD,
            iou=YOLO_NMS_IOU,
            max_det=MAX_DETECTIONS,
            device=device_index,
            verbose=False,
        )[0]
        del warmup_result
    cuda_sync(device)
    gc.collect()
    torch.cuda.empty_cache()

    # Apres la chauffe, cette base contient surtout les poids du modele.
    torch.cuda.reset_peak_memory_stats(device)
    baseline_bytes = torch.cuda.memory_allocated(device)
    latencies = []
    detection_counts = []
    for image in images:
        cuda_sync(device)
        started = time.perf_counter()
        result = model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=SCORE_THRESHOLD,
            iou=YOLO_NMS_IOU,
            max_det=MAX_DETECTIONS,
            device=device_index,
            verbose=False,
        )[0]
        cuda_sync(device)
        latencies.append(time.perf_counter() - started)
        detection_counts.append(0 if result.obb is None else len(result.obb))
        del result

    precision = "float16" if model.predictor.model.fp16 else "float32"
    if precision != "float32":
        raise RuntimeError("YOLO n'utilise pas la precision FP32 attendue.")
    model_parameters = sum(parameter.numel() for parameter in model.model.parameters())
    result = {
        "model_key": "yolo",
        "model": "YOLO11s-OBB",
        "precision": precision,
        "parameters_millions": model_parameters / 1_000_000.0,
        "checkpoint_size_mib": YOLO_CHECKPOINT.stat().st_size / 1024.0**2,
        **latency_summary(latencies),
        **memory_summary(device, baseline_bytes),
        "mean_detections": float(np.mean(detection_counts)),
        "p95_detections": float(np.percentile(detection_counts, 95)),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def faster_predict_one(model, image: Image.Image, device: torch.device):
    """Inclut la conversion PIL->tenseur et le transfert vers le GPU."""

    image_tensor = pil_to_tensor(image).to(dtype=torch.float32).div_(255.0)
    image_device = image_tensor.to(device, non_blocking=False)
    with torch.inference_mode():
        prediction = model([image_device])[0]
    del image_tensor, image_device
    return prediction


def benchmark_faster_rcnn(
    images: list[Image.Image],
    warmup_count: int,
    device_index: str,
) -> dict:
    """Mesure Faster R-CNN en FP32 avec le meme lot unitaire."""

    device = torch.device(f"cuda:{device_index}")
    checkpoint = torch.load(
        FASTER_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    model = build_model(load_coco_weights=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.roi_heads.score_thresh = SCORE_THRESHOLD
    model.roi_heads.detections_per_img = MAX_DETECTIONS
    model.to(device).eval()
    del checkpoint

    if not np.isclose(model.roi_heads.nms_thresh, FASTER_ROI_NMS_IOU):
        raise RuntimeError("Le seuil NMS ROI de Faster R-CNN a change.")
    if not np.isclose(model.rpn.nms_thresh, FASTER_RPN_NMS_IOU):
        raise RuntimeError("Le seuil NMS RPN de Faster R-CNN a change.")
    if next(model.parameters()).dtype != torch.float32:
        raise RuntimeError("Faster R-CNN n'utilise pas la precision FP32 attendue.")

    for image in images[:warmup_count]:
        warmup_prediction = faster_predict_one(model, image, device)
        del warmup_prediction
    cuda_sync(device)
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(device)
    baseline_bytes = torch.cuda.memory_allocated(device)
    latencies = []
    detection_counts = []
    for image in images:
        cuda_sync(device)
        started = time.perf_counter()
        prediction = faster_predict_one(model, image, device)
        cuda_sync(device)
        latencies.append(time.perf_counter() - started)
        detection_counts.append(len(prediction["boxes"]))
        del prediction

    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "model_key": "faster_rcnn",
        "model": "Faster R-CNN ResNet50-FPN v2",
        "precision": "float32",
        "parameters_millions": model_parameters / 1_000_000.0,
        "checkpoint_size_mib": FASTER_CHECKPOINT.stat().st_size / 1024.0**2,
        **latency_summary(latencies),
        **memory_summary(device, baseline_bytes),
        "mean_detections": float(np.mean(detection_counts)),
        "p95_detections": float(np.percentile(detection_counts, 95)),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def build_protocol(
    image_ids: list[int],
    warmup_count: int,
    device_index: str,
) -> dict:
    protocol = {
        "version": PROTOCOL_VERSION,
        "image_ids": image_ids,
        "image_ids_sha256": json_sha256(image_ids),
        "images": len(image_ids),
        "warmup_images": warmup_count,
        "batch_size": 1,
        "seed": SEED,
        "device_index": device_index,
        "gpu_name": torch.cuda.get_device_name(int(device_index)),
        "precision": "float32 pour les deux modeles",
        "score_threshold": SCORE_THRESHOLD,
        "max_detections": MAX_DETECTIONS,
        "image_size": IMAGE_SIZE,
        "disk_io_timed": False,
        "timed_operations": (
            "pretraitement en memoire, transfert GPU, modele et NMS native"
        ),
        "yolo_checkpoint_sha256": file_sha256(YOLO_CHECKPOINT),
        "faster_checkpoint_sha256": file_sha256(FASTER_CHECKPOINT),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": installed_version("torchvision"),
        "ultralytics_version": installed_version("ultralytics"),
        "cuda_runtime": torch.version.cuda,
    }
    protocol["protocol_id"] = json_sha256(protocol)
    return protocol


def run_benchmark(
    image_count: int = OFFICIAL_IMAGE_COUNT,
    warmup_count: int = OFFICIAL_WARMUP_COUNT,
    device_index: str = "0",
) -> pd.DataFrame:
    if not torch.cuda.is_available():
        raise RuntimeError("Ce benchmark comparatif requiert CUDA.")
    if image_count < 4:
        raise ValueError("Il faut au moins quatre images representatives.")
    if warmup_count < 1 or warmup_count > image_count:
        raise ValueError("Le nombre de chauffes doit etre entre 1 et images.")

    inputs = validate_inputs(load_checkpoints=False)
    if inputs["yolo_sha256"] != EXPECTED_YOLO_SHA256:
        raise RuntimeError("Le checkpoint YOLO fige a change.")
    if inputs["faster_sha256"] != EXPECTED_FASTER_SHA256:
        raise RuntimeError("Le checkpoint Faster R-CNN fige a change.")

    image_ids = select_representative_images(inputs["statistics"], image_count)
    output_dir = benchmark_output_dir(image_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol(image_ids, warmup_count, device_index)
    atomic_write_json(output_dir / "protocol.json", protocol)
    atomic_write_json(
        output_dir / "status.json",
        {
            "status": "running",
            "started_at_utc": utc_now(),
            "protocol_id": protocol["protocol_id"],
        },
    )

    print(f"Chargement en RAM des {len(image_ids)} images communes...")
    images = preload_images(inputs["dataset"], image_ids)
    print("Benchmark YOLO...")
    yolo_result = benchmark_yolo(images, warmup_count, device_index)
    atomic_write_json(
        output_dir / "partial_yolo.json",
        {"protocol_id": protocol["protocol_id"], **yolo_result},
    )

    print("Benchmark Faster R-CNN...")
    faster_result = benchmark_faster_rcnn(images, warmup_count, device_index)
    results = pd.DataFrame([yolo_result, faster_result])
    results.insert(2, "images", len(image_ids))
    results.insert(3, "warmup_images", warmup_count)
    results.insert(4, "batch_size", 1)
    results.insert(5, "score_threshold", SCORE_THRESHOLD)

    atomic_write_csv(output_dir / "efficiency_metrics.csv", results)
    atomic_write_json(
        output_dir / "efficiency_metrics.json",
        {
            "protocol_id": protocol["protocol_id"],
            "created_at_utc": utc_now(),
            "metrics": results.to_dict(orient="records"),
        },
    )
    atomic_write_json(
        output_dir / "status.json",
        {
            "status": "completed",
            "completed_at_utc": utc_now(),
            "protocol_id": protocol["protocol_id"],
            "metrics": "efficiency_metrics.csv",
        },
    )
    print("\nBenchmark termine :", output_dir)
    print(results.to_string(index=False))
    if image_count != OFFICIAL_IMAGE_COUNT:
        print("\nDEMO : ne pas utiliser ce petit echantillon comme resultat final.")
    return results


def check_environment() -> None:
    inputs = validate_inputs(load_checkpoints=False)
    print("Environnement valide.")
    print("GPU disponible :", torch.cuda.is_available())
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        print("GPU :", torch.cuda.get_device_name(0))
        print(f"VRAM libre : {free_bytes / 1024**3:.2f}/{total_bytes / 1024**3:.2f} Gio")
    print("Tuiles de validation :", inputs["statistics"]["tiles"])
    print("Benchmark officiel :", OFFICIAL_IMAGE_COUNT, "images")
    print("Chauffes :", OFFICIAL_WARMUP_COUNT)
    print("Batch : 1 | precision : FP32 | confiance :", SCORE_THRESHOLD)
    print("Aucune inference n'a ete lancee.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark commun YOLO et Faster R-CNN."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Verifie sans lancer d'inference.")
    run = commands.add_parser("run", help="Lance le benchmark.")
    run.add_argument("--images", type=int, default=OFFICIAL_IMAGE_COUNT)
    run.add_argument("--warmup", type=int, default=OFFICIAL_WARMUP_COUNT)
    run.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.command == "check":
        check_environment()
    else:
        run_benchmark(
            image_count=arguments.images,
            warmup_count=arguments.warmup,
            device_index=arguments.device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
