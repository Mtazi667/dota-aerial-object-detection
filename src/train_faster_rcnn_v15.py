"""Entraine Faster R-CNN HBB sur DOTA-v1.5 avec reprise et pause GPU."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import msvcrt
import os
import random
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection._utils import Matcher
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "outputs" / "dota_v15_tiles_1024"
METADATA_DIR = DATASET_DIR / "metadata"
RUNS_DIR = PROJECT_DIR / "outputs" / "faster_rcnn_runs"
CURRENT_RUN_FILE = RUNS_DIR / "current_training.json"

# Le meme verrou est utilise par YOLO : les deux modeles ne peuvent donc pas
# monopoliser simultanement le GPU du projet.
GPU_LOCK_FILE = PROJECT_DIR / "outputs" / "yolo_runs" / "training_gpu.lock"

DEFAULT_RUN_NAME = "faster_rcnn_v15_pilot_1h"
DEFAULT_HOURS = 1.0
DEFAULT_EPOCHS = 100
CLASS_COUNT = 16
MODEL_CLASS_COUNT = CLASS_COUNT + 1  # La sortie 0 est le fond interne.
IMAGE_SIZE = 1024
BATCH_SIZE = 1
RPN_GT_CHUNK_SIZE = 32
MAX_DETECTIONS = 2500
RPN_PROPOSALS = 3000
SCORE_THRESHOLD = 0.05
INITIAL_LEARNING_RATE = 0.0025
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
GRADIENT_CLIP_NORM = 10.0
SEED = 42
CHECKPOINT_EVERY_BATCHES = 250
PERIODIC_EVERY_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 8
HEARTBEAT_SECONDS = 5.0
MIN_FREE_DISK_GIB = 8.0


class StopTraining(Exception):
    """Interrompt proprement la boucle sans transformer l'arret en erreur."""


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def relative(path):
    """Enregistre un chemin relatif lorsqu'il appartient au projet."""

    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_DIR))
    except ValueError:
        return str(resolved)


def atomic_json_write(path, value):
    """Remplace un JSON seulement apres une ecriture complete sur disque."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, ensure_ascii=True)
            file.flush()
            os.fsync(file.fileno())
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (2**attempt))
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def read_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def atomic_torch_save(value, path):
    """Protege un checkpoint contre une coupure pendant torch.save."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as file:
            torch.save(value, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


class SessionLogger:
    """Duplique les sorties dans le terminal et dans un journal persistant."""

    def __init__(self, run_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = (
            Path(run_dir) / "logs" / f"session_{timestamp}_{os.getpid()}.log"
        )
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.file = None
        self.lock = threading.RLock()

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a", encoding="utf-8", buffering=1)
        sys.stdout = self
        sys.stderr = self
        print("Session :", utc_now(), flush=True)
        return self

    def write(self, text):
        with self.lock:
            self.stdout.write(text)
            self.file.write(text)
        return len(text)

    def flush(self):
        with self.lock:
            self.stdout.flush()
            self.file.flush()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, file=self
            )
        self.flush()
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()


class GpuLock:
    """Empeche un second processus du projet d'utiliser le GPU."""

    def __enter__(self):
        GPU_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.file = GPU_LOCK_FILE.open("a+b")
        self.file.seek(0, os.SEEK_END)
        if self.file.tell() == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            self.file.close()
            raise RuntimeError(
                "Un autre entrainement utilise deja le GPU."
            ) from error
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.file.seek(0)
        msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        self.file.close()


class Tracker:
    """Maintient le statut atomique et un heartbeat pendant les calculs."""

    def __init__(self, run_dir, config, session_log, previous=None):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "training_status.json"
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        previous = previous or {}
        self.active_base = float(previous.get("active_seconds", 0.0) or 0.0)
        self.active_started = time.monotonic()
        self.session_started = time.monotonic()
        self.status = {
            "schema_version": 1,
            "model": "fasterrcnn_resnet50_fpn_v2",
            "run_name": config["run_name"],
            "status": "initializing",
            "phase": "initialization",
            "message": "Preparation du run",
            "pid": os.getpid(),
            "session_log": relative(session_log),
            "started_at_utc": previous.get("started_at_utc", utc_now()),
            "session_started_at_utc": utc_now(),
            "heartbeat_at_utc": utc_now(),
            "last_progress_at_utc": utc_now(),
            "epoch": previous.get("epoch"),
            "epochs": config["epochs"],
            "batch": None,
            "batches": None,
            "batch_percent": None,
            "last_completed_epoch": previous.get("last_completed_epoch"),
            "losses": previous.get("losses", {}),
            "validation": previous.get("validation", {}),
            "best": previous.get("best", {}),
            "checkpoint": previous.get("checkpoint", {}),
            "learning_rate": config["learning_rate"],
            "session_seconds": 0.0,
            "active_seconds": self.active_base,
            "target_seconds": config["hours"] * 3600,
            "remaining_seconds": config["hours"] * 3600,
            "paused_gpu": False,
            "error": None,
        }
        self.write()

    def active_seconds(self):
        value = self.active_base
        if self.active_started is not None:
            value += time.monotonic() - self.active_started
        return value

    def pause_clock(self):
        with self.lock:
            if self.active_started is not None:
                self.active_base += time.monotonic() - self.active_started
                self.active_started = None

    def resume_clock(self):
        with self.lock:
            if self.active_started is None:
                self.active_started = time.monotonic()

    def snapshot(self):
        with self.lock:
            value = dict(self.status)
            value["heartbeat_at_utc"] = utc_now()
            value["session_seconds"] = time.monotonic() - self.session_started
            value["active_seconds"] = self.active_seconds()
            value["remaining_seconds"] = max(
                0.0, value["target_seconds"] - value["active_seconds"]
            )
            return value

    def write(self):
        atomic_json_write(self.path, self.snapshot())

    def update(self, progress=False, **values):
        with self.lock:
            self.status.update(values)
            if progress:
                self.status["last_progress_at_utc"] = utc_now()
        self.write()

    def start(self):
        def heartbeat_loop():
            while not self.stop_event.wait(HEARTBEAT_SECONDS):
                try:
                    self.write()
                except Exception:
                    pass

        self.thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.write()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Entrainement robuste de Faster R-CNN sur DOTA-v1.5."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("start", help="Commence un nouveau run.")
    start.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    start.add_argument("--hours", type=float, default=DEFAULT_HOURS)
    start.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    start.add_argument("--workers", type=int, default=0)
    start.add_argument(
        "--initial-checkpoint",
        type=Path,
        help=(
            "Initialise un nouveau run avec les poids d'un checkpoint, "
            "mais cree un nouvel optimiseur."
        ),
    )
    start.add_argument(
        "--pilot-train-images",
        type=int,
        help="Sous-ensemble train representatif reserve au pilote.",
    )
    start.add_argument(
        "--pilot-val-images",
        type=int,
        help="Sous-ensemble validation representatif reserve au pilote.",
    )
    start.add_argument(
        "--smoke-batches",
        type=int,
        help="Teste quelques lots sans produire un resultat scientifique.",
    )
    start.add_argument("--smoke-val-batches", type=int, default=2)

    resume = commands.add_parser("resume", help="Reprend un checkpoint.")
    resume.add_argument("--run-name", default=DEFAULT_RUN_NAME)

    for name, help_text in (
        ("pause-gpu", "Libere temporairement la VRAM."),
        ("resume-gpu", "Rend le GPU au processus en pause."),
        ("stop", "Demande un arret propre."),
    ):
        control = commands.add_parser(name, help=help_text)
        control.add_argument("--run-name", default=DEFAULT_RUN_NAME)

    commands.add_parser("check", help="Verifie les prerequis sans entrainer.")
    return parser.parse_args()


def validate_run_name(name):
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Le nom du run doit etre un simple nom de dossier.")


def set_reproducible_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_detection_batch(batch):
    return tuple(zip(*batch))


class DOTAHBBDataset(Dataset):
    """Charge les PNG et les HBB derivees des OBB de DOTA-v1.5."""

    def __init__(self, dataset_dir, split):
        self.dataset_dir = Path(dataset_dir)
        metadata_dir = self.dataset_dir / "metadata"
        manifest = pd.read_csv(
            metadata_dir / "tile_manifest.csv",
            usecols=["tile_id", "split", "tile_image_path"],
        )
        self.tiles = manifest.loc[manifest["split"] == split].reset_index(
            drop=True
        )
        annotations = pd.read_csv(
            metadata_dir / "object_annotations.csv",
            usecols=[
                "tile_id",
                "split",
                "hbb_x_min",
                "hbb_y_min",
                "hbb_x_max",
                "hbb_y_max",
                "faster_rcnn_class_id",
            ],
        )
        annotations = annotations.loc[annotations["split"] == split]
        self.annotations = {}
        box_columns = [
            "hbb_x_min",
            "hbb_y_min",
            "hbb_x_max",
            "hbb_y_max",
        ]
        for tile_id, group in annotations.groupby("tile_id", sort=False):
            self.annotations[tile_id] = (
                group[box_columns].to_numpy(dtype=np.float32, copy=True),
                group["faster_rcnn_class_id"].to_numpy(
                    dtype=np.int64, copy=True
                ),
            )
        self.split = split

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        row = self.tiles.iloc[index]
        image_path = self.dataset_dir / row["tile_image_path"]
        with Image.open(image_path) as image:
            image_array = np.asarray(
                image.convert("RGB"), dtype=np.float32
            ).copy()
        image_tensor = (
            torch.from_numpy(image_array).permute(2, 0, 1) / 255.0
        )

        boxes, labels = self.annotations.get(
            row["tile_id"],
            (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            ),
        )
        boxes_tensor = torch.from_numpy(boxes.copy())
        labels_tensor = torch.from_numpy(labels.copy())
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": (
                (boxes_tensor[:, 2] - boxes_tensor[:, 0])
                * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            ),
            "iscrowd": torch.zeros(len(boxes_tensor), dtype=torch.int64),
        }
        return image_tensor, target


def representative_subset(dataset, requested_size, seed):
    """Conserve les classes rares, les cas denses et un reste aleatoire."""

    if requested_size is None:
        return dataset
    if requested_size <= 0 or requested_size > len(dataset):
        raise ValueError("Taille du sous-ensemble pilote invalide.")

    tile_to_index = {
        str(row.tile_id): int(index)
        for index, row in dataset.tiles.iterrows()
    }
    object_counts = []
    class_candidates = {class_id: [] for class_id in range(1, 17)}
    for tile_id, (boxes, labels) in dataset.annotations.items():
        index = tile_to_index[tile_id]
        object_counts.append((len(boxes), index))
        for class_id in np.unique(labels):
            class_candidates[int(class_id)].append(index)

    selected = set()
    # Les 32 tuiles les plus denses verifient regulierement le RPN par blocs.
    for _, index in sorted(object_counts, reverse=True)[:32]:
        selected.add(index)

    generator = np.random.default_rng(seed)
    # Ajouter au moins une tuile de chaque classe, notamment container-crane.
    for class_id in range(1, 17):
        candidates = class_candidates[class_id]
        if candidates:
            selected.add(int(generator.choice(candidates)))

    # Completer sans remise avec des tuiles du split entier, fonds compris.
    remaining = np.array(
        [index for index in range(len(dataset)) if index not in selected],
        dtype=np.int64,
    )
    generator.shuffle(remaining)
    needed = requested_size - len(selected)
    if needed < 0:
        selected = set(sorted(selected)[:requested_size])
    else:
        selected.update(int(index) for index in remaining[:needed])

    return Subset(dataset, sorted(selected))


def match_anchors_in_gt_chunks(gt_boxes, anchors, matcher, chunk_size):
    """Reproduit Matcher sans conserver toute la matrice objet-ancre."""

    anchor_count = len(anchors)
    if len(gt_boxes) == 0:
        return torch.full(
            (anchor_count,),
            Matcher.BELOW_LOW_THRESHOLD,
            dtype=torch.int64,
            device=anchors.device,
        )

    best_iou = torch.full((anchor_count,), -torch.inf, device=anchors.device)
    best_gt = torch.zeros(
        anchor_count, dtype=torch.int64, device=anchors.device
    )
    low_quality_groups = []
    for start in range(0, len(gt_boxes), chunk_size):
        quality = box_iou(gt_boxes[start : start + chunk_size], anchors)
        chunk_best_iou, chunk_best_gt = quality.max(dim=0)
        improved = chunk_best_iou > best_iou
        best_iou[improved] = chunk_best_iou[improved]
        best_gt[improved] = chunk_best_gt[improved] + start
        if matcher.allow_low_quality_matches:
            highest_per_gt = quality.max(dim=1).values
            _, anchor_indices = torch.where(
                quality == highest_per_gt[:, None]
            )
            low_quality_groups.append(anchor_indices)

    all_best_gt = best_gt.clone()
    matches = best_gt.clone()
    matches[best_iou < matcher.low_threshold] = Matcher.BELOW_LOW_THRESHOLD
    between = (best_iou >= matcher.low_threshold) & (
        best_iou < matcher.high_threshold
    )
    matches[between] = Matcher.BETWEEN_THRESHOLDS
    if matcher.allow_low_quality_matches and low_quality_groups:
        indices = torch.cat(low_quality_groups)
        matches[indices] = all_best_gt[indices]
    return matches


def assign_targets_to_anchors_chunked(self, anchors, targets):
    labels = []
    matched_boxes = []
    for anchors_per_image, target in zip(anchors, targets):
        gt_boxes = target["boxes"]
        if gt_boxes.numel() == 0:
            device = anchors_per_image.device
            boxes_per_image = torch.zeros(
                anchors_per_image.shape, dtype=torch.float32, device=device
            )
            labels_per_image = torch.zeros(
                len(anchors_per_image), dtype=torch.float32, device=device
            )
        else:
            matched_indices = match_anchors_in_gt_chunks(
                gt_boxes,
                anchors_per_image,
                self.proposal_matcher,
                self.gt_chunk_size,
            )
            boxes_per_image = gt_boxes[matched_indices.clamp(min=0)]
            labels_per_image = (matched_indices >= 0).to(torch.float32)
            labels_per_image[
                matched_indices
                == self.proposal_matcher.BELOW_LOW_THRESHOLD
            ] = 0.0
            labels_per_image[
                matched_indices
                == self.proposal_matcher.BETWEEN_THRESHOLDS
            ] = -1.0
        labels.append(labels_per_image)
        matched_boxes.append(boxes_per_image)
    return labels, matched_boxes


def enable_chunked_rpn_matching(model, chunk_size=RPN_GT_CHUNK_SIZE):
    model.rpn.gt_chunk_size = int(chunk_size)
    model.rpn.assign_targets_to_anchors = MethodType(
        assign_targets_to_anchors_chunked, model.rpn
    )


def build_model(load_coco_weights):
    weights = (
        FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        if load_coco_weights
        else None
    )
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        # Les poids de detection COCO contiennent deja le backbone. Lors
        # d'une reprise, le checkpoint restaurera lui-meme tous les poids.
        weights_backbone=None,
        min_size=IMAGE_SIZE,
        max_size=IMAGE_SIZE,
        rpn_post_nms_top_n_train=RPN_PROPOSALS,
        rpn_post_nms_top_n_test=RPN_PROPOSALS,
        box_score_thresh=SCORE_THRESHOLD,
        box_detections_per_img=MAX_DETECTIONS,
    )
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        input_features, MODEL_CLASS_COUNT
    )
    enable_chunked_rpn_matching(model)
    return model


def optimizer_to(optimizer, device):
    """Deplace les buffers SGD afin que la pause libere vraiment la VRAM."""

    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(device)


def build_coco_ground_truth(dataset):
    categories_path = METADATA_DIR / "class_mapping.csv"
    categories_df = pd.read_csv(categories_path)
    coco = COCO()
    images = []
    annotations = []
    annotation_id = 1
    for image_id, row in dataset.tiles.iterrows():
        images.append(
            {
                "id": int(image_id),
                "file_name": str(row["tile_image_path"]),
                "width": IMAGE_SIZE,
                "height": IMAGE_SIZE,
            }
        )
        boxes, labels = dataset.annotations.get(
            row["tile_id"],
            (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            ),
        )
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = [float(value) for value in box]
            width = x2 - x1
            height = y2 - y1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    coco.dataset = {
        "info": {"description": "DOTA-v1.5 HBB validation"},
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": int(row.faster_rcnn_class_id),
                "name": str(row.class_name),
            }
            for row in categories_df.itertuples(index=False)
        ],
    }
    coco.createIndex()
    return coco


def predictions_to_coco(image_ids, predictions):
    results = []
    for image_id, prediction in zip(image_ids, predictions):
        boxes = prediction["boxes"].detach().cpu().numpy()
        labels = prediction["labels"].detach().cpu().numpy()
        scores = prediction["scores"].detach().cpu().numpy()
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = [float(value) for value in box]
            results.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    return results


def mean_valid(values):
    valid = values[values >= 0]
    return float(valid.mean()) if valid.size else 0.0


def evaluate_model(
    model,
    loader,
    device,
    coco_ground_truth,
    tracker,
    control,
    optimizer,
    save_pause_checkpoint,
):
    model.eval()
    results = []
    image_ids = []
    started = time.monotonic()
    with torch.inference_mode():
        for batch_number, (images, targets) in enumerate(loader, start=1):
            images_gpu = [image.to(device, non_blocking=True) for image in images]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predictions = model(images_gpu)
            batch_ids = [int(target["image_id"].item()) for target in targets]
            results.extend(predictions_to_coco(batch_ids, predictions))
            image_ids.extend(batch_ids)
            tracker.update(
                progress=True,
                status="running",
                phase="validation",
                message="Validation COCO",
                validation_batch=batch_number,
                validation_batches=len(loader),
                validation_percent=100.0 * batch_number / len(loader),
            )
            del images_gpu, predictions
            control.handle_pause(
                model, optimizer, device, tracker, save_pause_checkpoint
            )
            if control.stop_requested():
                raise StopTraining("Arret demande pendant la validation.")

    if not results:
        return {
            "mAP50_95": 0.0,
            "mAP50": 0.0,
            "mAP75": 0.0,
            "recall": 0.0,
            "predictions": 0,
            "seconds": time.monotonic() - started,
        }

    coco_predictions = coco_ground_truth.loadRes(results)
    evaluator = COCOeval(coco_ground_truth, coco_predictions, "bbox")
    evaluator.params.imgIds = sorted(image_ids)
    evaluator.params.maxDets = [1, 100, MAX_DETECTIONS]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    recall = evaluator.eval["recall"][:, :, 0, -1]
    iou50 = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.50))[0])
    iou75 = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.75))[0])
    return {
        "mAP50_95": mean_valid(precision),
        "mAP50": mean_valid(precision[iou50]),
        "mAP75": mean_valid(precision[iou75]),
        "recall": mean_valid(recall),
        "predictions": len(results),
        "seconds": time.monotonic() - started,
    }


class ControlFiles:
    """Recoit les commandes externes sans interrompre brutalement Python."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.pause_path = self.run_dir / "PAUSE_GPU_REQUESTED"
        self.stop_path = self.run_dir / "STOP_REQUESTED"

    def stop_requested(self):
        return self.stop_path.exists()

    def handle_pause(self, model, optimizer, device, tracker, save_checkpoint):
        if not self.pause_path.exists():
            return

        tracker.update(
            status="pause_requested",
            phase="checkpoint",
            message="Sauvegarde avant liberation du GPU",
        )
        save_checkpoint()
        model.to("cpu")
        if optimizer is not None:
            optimizer_to(optimizer, "cpu")
        gc.collect()
        torch.cuda.empty_cache()
        tracker.pause_clock()
        tracker.update(
            status="paused_gpu",
            phase="paused_gpu",
            message="GPU libere; attente de resume-gpu",
            paused_gpu=True,
        )
        print("GPU libere. Le processus attend resume-gpu.", flush=True)

        while self.pause_path.exists():
            if self.stop_requested():
                raise StopTraining("Arret demande pendant la pause GPU.")
            time.sleep(1)
            tracker.write()

        tracker.update(
            status="resuming_gpu",
            phase="resuming_gpu",
            message="Rechargement du modele sur CUDA",
        )
        model.to(device)
        if optimizer is not None:
            optimizer_to(optimizer, device)
        torch.cuda.empty_cache()
        tracker.resume_clock()
        tracker.update(
            status="running",
            phase="training",
            message="GPU recharge; entrainement repris",
            paused_gpu=False,
        )
        print("GPU recharge. Reprise du calcul.", flush=True)


def build_config(args):
    return {
        "schema_version": 1,
        "run_name": args.run_name,
        "dataset": relative(DATASET_DIR),
        "model": "fasterrcnn_resnet50_fpn_v2",
        "class_count": CLASS_COUNT,
        "model_class_count": MODEL_CLASS_COUNT,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "workers": args.workers,
        "rpn_gt_chunk_size": RPN_GT_CHUNK_SIZE,
        "rpn_proposals": RPN_PROPOSALS,
        "max_detections": MAX_DETECTIONS,
        "score_threshold": SCORE_THRESHOLD,
        "learning_rate": INITIAL_LEARNING_RATE,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "seed": SEED,
        "epochs": args.epochs,
        "hours": args.hours,
        "checkpoint_every_batches": CHECKPOINT_EVERY_BATCHES,
        "periodic_every_epochs": PERIODIC_EVERY_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "smoke_batches": args.smoke_batches,
        "smoke_val_batches": args.smoke_val_batches,
        "pilot_train_images": args.pilot_train_images,
        "pilot_val_images": args.pilot_val_images,
        "initial_checkpoint": (
            relative(args.initial_checkpoint)
            if args.initial_checkpoint is not None
            else None
        ),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }


def project_path(path_text):
    """Retrouve un chemin absolu enregistre relativement au projet."""

    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_DIR / path


def load_initial_checkpoint(path):
    """Valide un checkpoint utilise seulement comme poids de depart."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint initial introuvable : {path}")
    checkpoint = checkpoint_is_valid(path)
    if checkpoint is None:
        raise RuntimeError("Le checkpoint initial est incomplet ou corrompu.")

    source_config = checkpoint["config"]
    if (
        source_config.get("model") != "fasterrcnn_resnet50_fpn_v2"
        or int(source_config.get("class_count", -1)) != CLASS_COUNT
        or int(source_config.get("image_size", -1)) != IMAGE_SIZE
        or int(source_config.get("rpn_gt_chunk_size", -1))
        != RPN_GT_CHUNK_SIZE
    ):
        raise RuntimeError(
            "Le checkpoint initial n'est pas compatible avec ce modele."
        )
    return checkpoint


def check_environment():
    expected_python = PROJECT_DIR / ".venv_gpu" / "Scripts" / "python.exe"
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise RuntimeError("Utilise .\\.venv_gpu\\Scripts\\python.exe.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA n'est pas disponible dans PyTorch.")
    required = [
        METADATA_DIR / "tile_manifest.csv",
        METADATA_DIR / "object_annotations.csv",
        METADATA_DIR / "class_mapping.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Fichiers dataset absents : {missing}")
    classes = pd.read_csv(METADATA_DIR / "class_mapping.csv")
    if len(classes) != CLASS_COUNT:
        raise RuntimeError("Le mapping DOTA-v1.5 doit contenir 16 classes.")
    free_disk_gib = shutil.disk_usage(RUNS_DIR.parent).free / 1024**3
    if free_disk_gib < MIN_FREE_DISK_GIB:
        raise RuntimeError("Espace disque libre insuffisant.")
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    print("Environnement valide.")
    print("PyTorch :", torch.__version__)
    print("Torchvision :", torchvision.__version__)
    print("GPU :", torch.cuda.get_device_name(0))
    print(
        "VRAM libre :",
        f"{free_vram / 1024**3:.2f}/{total_vram / 1024**3:.2f} Gio",
    )
    print("Espace disque libre :", f"{free_disk_gib:.2f} Gio")
    print("Classes :", len(classes))
    print("RPN dense : blocs de", RPN_GT_CHUNK_SIZE)
    print("Aucun entrainement n'a ete lance.")


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    scaler,
    config,
    progress,
    tracker,
):
    return {
        "schema_version": 1,
        "config": config,
        "progress": dict(progress),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "active_seconds": tracker.active_seconds(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "saved_at_utc": utc_now(),
    }


def save_checkpoint(
    run_dir,
    model,
    optimizer,
    scheduler,
    scaler,
    config,
    progress,
    tracker,
    save_best=False,
    save_periodic=False,
):
    payload = checkpoint_payload(
        model, optimizer, scheduler, scaler, config, progress, tracker
    )
    checkpoints_dir = Path(run_dir) / "checkpoints"
    recovery_dir = Path(run_dir) / "recovery"
    last_path = checkpoints_dir / "last.pt"
    latest_path = recovery_dir / "resume_latest.pt"
    previous_path = recovery_dir / "resume_previous.pt"

    atomic_torch_save(payload, last_path)
    if latest_path.exists():
        previous_path.parent.mkdir(parents=True, exist_ok=True)
        # Le renommage reste atomique : apres une coupure, latest ou previous
        # contient toujours un checkpoint complet.
        os.replace(latest_path, previous_path)
    atomic_torch_save(payload, latest_path)
    if save_best:
        atomic_torch_save(payload, checkpoints_dir / "best.pt")
    if save_periodic:
        epoch_human = int(progress["epoch"]) + 1
        atomic_torch_save(
            payload, checkpoints_dir / f"epoch_{epoch_human:04d}.pt"
        )

    tracker.update(
        checkpoint={
            "path": relative(latest_path),
            "epoch": int(progress["epoch"]) + 1,
            "stage": progress["stage"],
            "samples_completed_in_epoch": progress[
                "samples_completed_in_epoch"
            ],
            "saved_at_utc": utc_now(),
            "size_mib": latest_path.stat().st_size / 1024**2,
        }
    )


def checkpoint_is_valid(path, expected_config=None):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    required = {
        "config",
        "progress",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
    }
    if checkpoint.get("schema_version") != 1 or not required.issubset(checkpoint):
        return None
    if expected_config is not None:
        stored = checkpoint["config"]
        fixed_keys = (
            "run_name",
            "class_count",
            "image_size",
            "batch_size",
            "rpn_gt_chunk_size",
            "max_detections",
            "pilot_train_images",
            "pilot_val_images",
        )
        if any(stored.get(key) != expected_config.get(key) for key in fixed_keys):
            return None
    return checkpoint


def find_resume_checkpoint(run_dir, config):
    run_dir = Path(run_dir)
    candidates = [
        run_dir / "checkpoints" / "last.pt",
        run_dir / "recovery" / "resume_latest.pt",
        run_dir / "recovery" / "resume_previous.pt",
    ]
    candidates.extend(
        sorted((run_dir / "checkpoints").glob("epoch_*.pt"), reverse=True)
    )
    valid = []
    for path in candidates:
        if path.is_file():
            checkpoint = checkpoint_is_valid(path, config)
            if checkpoint is not None:
                progress = checkpoint["progress"]
                valid.append(
                    (
                        int(progress.get("global_step", 0)),
                        path.stat().st_mtime_ns,
                        path,
                        checkpoint,
                    )
                )
    if not valid:
        raise FileNotFoundError("Aucun checkpoint Faster R-CNN valide.")
    _, _, path, checkpoint = max(valid, key=lambda item: (item[0], item[1]))
    return path, checkpoint


def write_results_row(run_dir, row):
    path = Path(run_dir) / "results.csv"
    rows = {}
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as file:
            for existing in csv.DictReader(file):
                rows[int(existing["epoch"])] = existing
    rows[int(row["epoch"])] = row
    ordered = [rows[index] for index in sorted(rows)]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerows(ordered)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def epoch_indices(dataset_length, epoch):
    generator = torch.Generator().manual_seed(SEED + epoch)
    return torch.randperm(dataset_length, generator=generator).tolist()


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    tracker,
    control,
    progress,
    save_partial,
    smoke_limit=None,
):
    model.train()
    loss_sums = progress.get("loss_sums") or {
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
    }
    loss_batches = int(progress.get("loss_batches", 0))
    for local_batch, (images, targets) in enumerate(loader, start=1):
        images_gpu = [image.to(device, non_blocking=True) for image in images]
        targets_gpu = [
            {
                name: value.to(device, non_blocking=True)
                for name, value in target.items()
            }
            for target in targets
        ]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss_tensors = model(images_gpu, targets_gpu)
            total_loss = sum(loss_tensors.values())
        if not torch.isfinite(total_loss):
            raise RuntimeError("Une perte Faster R-CNN n'est pas finie.")
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRADIENT_CLIP_NORM
        )
        scaler.step(optimizer)
        scaler.update()

        current_losses = {
            name: float(value.detach().cpu())
            for name, value in loss_tensors.items()
        }
        for name in loss_sums:
            loss_sums[name] += current_losses[name]
        loss_batches += 1
        progress["samples_completed_in_epoch"] += len(images)
        progress["global_step"] += 1
        progress["loss_sums"] = loss_sums
        progress["loss_batches"] = loss_batches
        mean_losses = {
            name: value / loss_batches for name, value in loss_sums.items()
        }
        tracker.update(
            progress=True,
            status="running",
            phase="training",
            message="Entrainement des lots",
            batch=progress["samples_completed_in_epoch"],
            batches=progress["epoch_dataset_size"],
            batch_percent=(
                100.0
                * progress["samples_completed_in_epoch"]
                / progress["epoch_dataset_size"]
            ),
            losses=mean_losses,
            learning_rate=optimizer.param_groups[0]["lr"],
        )
        del images_gpu, targets_gpu, loss_tensors, total_loss

        if progress["global_step"] % CHECKPOINT_EVERY_BATCHES == 0:
            save_partial()
        control.handle_pause(
            model, optimizer, device, tracker, save_partial
        )
        if control.stop_requested():
            save_partial()
            raise StopTraining("Arret demande apres un lot termine.")
        if tracker.active_seconds() >= tracker.status["target_seconds"]:
            save_partial()
            raise StopTraining("Budget de temps atteint apres un lot termine.")
        if smoke_limit is not None and progress["global_step"] >= smoke_limit:
            return mean_losses, True

    return {
        name: value / max(1, loss_batches)
        for name, value in loss_sums.items()
    }, False


def initial_progress():
    return {
        "epoch": 0,
        "stage": "training",
        "samples_completed_in_epoch": 0,
        "epoch_dataset_size": 0,
        "global_step": 0,
        "loss_sums": None,
        "loss_batches": 0,
        "train_losses": {},
        "validation": {},
        "best_map50_95": -math.inf,
        "best_epoch": None,
        "epochs_without_improvement": 0,
    }


def run_training(args, mode):
    validate_run_name(args.run_name)
    run_dir = RUNS_DIR / args.run_name
    config_path = run_dir / "training_config.json"
    status_path = run_dir / "training_status.json"

    if mode == "start":
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                "Le dossier du run existe deja; utilise resume ou un autre nom."
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        config = build_config(args)
        atomic_json_write(config_path, config)
        previous_status = None
        checkpoint = None
        checkpoint_path = None
    else:
        if not config_path.is_file():
            raise FileNotFoundError("Configuration du run introuvable.")
        config = read_json(config_path)
        previous_status = read_json(status_path) if status_path.is_file() else None
        checkpoint_path, checkpoint = find_resume_checkpoint(run_dir, config)

    try:
        check_environment()
    except Exception as error:
        # Start-Process masque la fenetre. Ce statut rend tout de meme une
        # erreur de preflight visible dans le moniteur CPU.
        atomic_json_write(
            status_path,
            {
                "schema_version": 1,
                "model": "fasterrcnn_resnet50_fpn_v2",
                "run_name": args.run_name,
                "status": "failed",
                "phase": "preflight",
                "message": "Echec des controles avant entrainement",
                "pid": os.getpid(),
                "heartbeat_at_utc": utc_now(),
                "last_progress_at_utc": utc_now(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    control = ControlFiles(run_dir)
    # Une commande explicite start/resume commence une session active. Les
    # anciens fichiers de controle ne doivent pas l'arreter immediatement.
    control.pause_path.unlink(missing_ok=True)
    control.stop_path.unlink(missing_ok=True)

    with GpuLock(), SessionLogger(run_dir) as session:
        tracker = Tracker(run_dir, config, session.path, previous_status)
        tracker.start()
        atomic_json_write(
            CURRENT_RUN_FILE,
            {
                "run_name": args.run_name,
                "status_path": relative(tracker.path),
            },
        )
        try:
            set_reproducible_seeds(config["seed"])
            tracker.update(message="Chargement des datasets")
            full_train_dataset = DOTAHBBDataset(DATASET_DIR, "train")
            full_validation_dataset = DOTAHBBDataset(
                DATASET_DIR, "validation"
            )
            coco_ground_truth = build_coco_ground_truth(
                full_validation_dataset
            )
            train_dataset = representative_subset(
                full_train_dataset,
                config.get("pilot_train_images"),
                config["seed"] + 10_000,
            )
            validation_dataset = representative_subset(
                full_validation_dataset,
                config.get("pilot_val_images"),
                config["seed"] + 20_000,
            )
            print(
                "Tuiles utilisees :",
                f"train={len(train_dataset)},",
                f"validation={len(validation_dataset)}",
                flush=True,
            )
            tracker.update(message="Construction de Faster R-CNN")

            initial_checkpoint = None
            initial_checkpoint_path = config.get("initial_checkpoint")
            if checkpoint is None and initial_checkpoint_path is not None:
                # Un nouveau run reprend uniquement les poids du meilleur
                # modele pilote. L'optimiseur et son scheduler seront neufs.
                initial_checkpoint_path = project_path(
                    initial_checkpoint_path
                )
                initial_checkpoint = load_initial_checkpoint(
                    initial_checkpoint_path
                )

            model = build_model(
                load_coco_weights=(
                    checkpoint is None and initial_checkpoint is None
                )
            )
            if initial_checkpoint is not None:
                model.load_state_dict(initial_checkpoint["model_state"])
                source_progress = initial_checkpoint["progress"]
                print(
                    "Poids initiaux :",
                    relative(initial_checkpoint_path),
                    f"(meilleure epoque pilote "
                    f"{source_progress.get('best_epoch')})",
                    flush=True,
                )
                tracker.update(
                    message="Poids du meilleur pilote charges"
                )
                # Le gros dictionnaire contient aussi l'ancien optimiseur;
                # le supprimer evite de garder inutilement cette RAM.
                del initial_checkpoint

            device = torch.device("cuda:0")
            model.to(device)
            optimizer = torch.optim.SGD(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=config["learning_rate"],
                momentum=config["momentum"],
                weight_decay=config["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2
            )
            scaler = torch.amp.GradScaler("cuda")
            progress = initial_progress()

            if checkpoint is not None:
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                optimizer_to(optimizer, device)
                scheduler.load_state_dict(checkpoint["scheduler_state"])
                scaler.load_state_dict(checkpoint["scaler_state"])
                progress = checkpoint["progress"]
                # Ajouter au temps restaure la preparation de cette nouvelle
                # session deja ecoulee depuis la creation du Tracker.
                current_session_setup = (
                    time.monotonic() - tracker.active_started
                )
                tracker.active_base = float(
                    checkpoint.get("active_seconds", tracker.active_base)
                ) + current_session_setup
                tracker.active_started = time.monotonic()
                if checkpoint.get("torch_rng_state") is not None:
                    torch.set_rng_state(checkpoint["torch_rng_state"])
                if checkpoint.get("cuda_rng_state") is not None:
                    torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
                # Un checkpoint complet appartient a l'epoque terminee. La
                # nouvelle session doit commencer a l'epoque suivante sans
                # refaire sa validation ni appliquer deux fois le scheduler.
                if progress.get("stage") == "complete":
                    progress.update(
                        {
                            "epoch": int(progress["epoch"]) + 1,
                            "stage": "training",
                            "samples_completed_in_epoch": 0,
                            "epoch_dataset_size": len(train_dataset),
                            "loss_sums": None,
                            "loss_batches": 0,
                        }
                    )
                print("Checkpoint repris :", relative(checkpoint_path), flush=True)

            def save_current(save_best=False, save_periodic=False):
                save_checkpoint(
                    run_dir,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    progress,
                    tracker,
                    save_best=save_best,
                    save_periodic=save_periodic,
                )

            smoke_batches = config.get("smoke_batches")
            if smoke_batches is not None:
                indices = epoch_indices(len(train_dataset), 0)[:smoke_batches]
                loader = DataLoader(
                    Subset(train_dataset, indices),
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=config["workers"],
                    pin_memory=True,
                    collate_fn=collate_detection_batch,
                )
                progress.update(
                    {
                        "epoch": 0,
                        "stage": "training",
                        "samples_completed_in_epoch": 0,
                        "epoch_dataset_size": len(indices),
                        "loss_sums": None,
                        "loss_batches": 0,
                    }
                )
                tracker.update(epoch=1, epochs=config["epochs"])
                losses, _ = train_epoch(
                    model,
                    loader,
                    optimizer,
                    scaler,
                    device,
                    tracker,
                    control,
                    progress,
                    save_current,
                    smoke_limit=smoke_batches,
                )
                val_count = min(
                    len(validation_dataset),
                    int(config.get("smoke_val_batches", 2)),
                )
                val_loader = DataLoader(
                    Subset(validation_dataset, list(range(val_count))),
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    collate_fn=collate_detection_batch,
                )
                metrics = evaluate_model(
                    model,
                    val_loader,
                    device,
                    coco_ground_truth,
                    tracker,
                    control,
                    optimizer,
                    save_current,
                )
                tracker.update(
                    status="smoke_completed",
                    phase="completed",
                    message="Smoke test termine; metriques non scientifiques",
                    losses=losses,
                    validation=metrics,
                )
                return

            while progress["epoch"] < config["epochs"]:
                epoch = int(progress["epoch"])
                tracker.update(epoch=epoch + 1, epochs=config["epochs"])
                if progress["stage"] == "training":
                    all_indices = epoch_indices(len(train_dataset), epoch)
                    start_sample = int(progress["samples_completed_in_epoch"])
                    remaining_indices = all_indices[start_sample:]
                    progress["epoch_dataset_size"] = len(train_dataset)
                    train_loader = DataLoader(
                        Subset(train_dataset, remaining_indices),
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=config["workers"],
                        pin_memory=True,
                        collate_fn=collate_detection_batch,
                    )
                    losses, _ = train_epoch(
                        model,
                        train_loader,
                        optimizer,
                        scaler,
                        device,
                        tracker,
                        control,
                        progress,
                        save_current,
                    )
                    progress["train_losses"] = losses
                    progress["stage"] = "validation_pending"
                    save_current()

                validation_loader = DataLoader(
                    validation_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=config["workers"],
                    pin_memory=True,
                    collate_fn=collate_detection_batch,
                )
                metrics = evaluate_model(
                    model,
                    validation_loader,
                    device,
                    coco_ground_truth,
                    tracker,
                    control,
                    optimizer,
                    save_current,
                )
                progress["validation"] = metrics
                scheduler.step(metrics["mAP50_95"])
                improved = metrics["mAP50_95"] > progress["best_map50_95"]
                if improved:
                    progress["best_map50_95"] = metrics["mAP50_95"]
                    progress["best_epoch"] = epoch + 1
                    progress["epochs_without_improvement"] = 0
                else:
                    progress["epochs_without_improvement"] += 1

                row = {
                    "epoch": epoch + 1,
                    **progress["train_losses"],
                    "train_total_loss": sum(progress["train_losses"].values()),
                    **metrics,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "active_seconds": tracker.active_seconds(),
                }
                write_results_row(run_dir, row)
                progress["stage"] = "complete"
                save_current(
                    save_best=improved,
                    save_periodic=(epoch + 1) % PERIODIC_EVERY_EPOCHS == 0,
                )
                tracker.update(
                    progress=True,
                    last_completed_epoch=epoch + 1,
                    losses=progress["train_losses"],
                    validation=metrics,
                    best={
                        "mAP50_95": progress["best_map50_95"],
                        "epoch": progress["best_epoch"],
                    },
                )
                if (
                    progress["epochs_without_improvement"]
                    >= config["early_stopping_patience"]
                ):
                    tracker.update(
                        status="completed",
                        phase="early_stopping",
                        message="Arret anticipe sans nouvelle amelioration",
                    )
                    return

                progress.update(
                    {
                        "epoch": epoch + 1,
                        "stage": "training",
                        "samples_completed_in_epoch": 0,
                        "epoch_dataset_size": len(train_dataset),
                        "loss_sums": None,
                        "loss_batches": 0,
                    }
                )
                if control.stop_requested():
                    raise StopTraining("Arret demande apres validation.")

            tracker.update(
                status="completed",
                phase="completed",
                message="Nombre maximal d'epoques atteint",
            )
        except StopTraining as reason:
            tracker.update(
                status="interrupted",
                phase="stopped",
                message=str(reason),
            )
        except Exception as error:
            tracker.update(
                status="failed",
                phase="error",
                message="Echec de l'entrainement",
                error=f"{type(error).__name__}: {error}",
            )
            raise
        finally:
            tracker.close()


def control_command(args):
    validate_run_name(args.run_name)
    run_dir = RUNS_DIR / args.run_name
    status_path = run_dir / "training_status.json"
    if not run_dir.is_dir() or not status_path.is_file():
        raise FileNotFoundError("Run Faster R-CNN introuvable.")
    pause_path = run_dir / "PAUSE_GPU_REQUESTED"
    stop_path = run_dir / "STOP_REQUESTED"
    if args.command == "pause-gpu":
        pause_path.touch()
        print("Pause GPU demandee. Attends le statut paused_gpu.")
    elif args.command == "resume-gpu":
        pause_path.unlink(missing_ok=True)
        print("Reprise GPU demandee.")
    elif args.command == "stop":
        stop_path.touch()
        pause_path.unlink(missing_ok=True)
        print("Arret propre demande.")


def main():
    args = parse_arguments()
    if args.command == "check":
        check_environment()
        return 0
    if args.command in {"pause-gpu", "resume-gpu", "stop"}:
        control_command(args)
        return 0
    if args.command == "start":
        if args.hours <= 0 or args.epochs <= 0 or args.workers < 0:
            raise ValueError("hours, epochs ou workers invalide.")
        if (args.pilot_train_images is None) != (
            args.pilot_val_images is None
        ):
            raise ValueError(
                "Les tailles pilote train et validation vont ensemble."
            )
        if args.smoke_batches is not None and args.smoke_batches <= 0:
            raise ValueError("smoke-batches doit etre positif.")
        run_training(args, "start")
        return 0
    if args.command == "resume":
        run_training(args, "resume")
        return 0
    raise RuntimeError("Commande inconnue.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERREUR : {type(error).__name__}: {error}", file=sys.stderr)
        raise
