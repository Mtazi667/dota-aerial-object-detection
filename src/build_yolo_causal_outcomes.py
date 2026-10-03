"""Construit le resultat causal Y a partir des predictions YOLO-OBB.

Le traitement causal est defini au niveau de l'image source. Ce script
effectue donc aussi l'evaluation au niveau source : il predit les tuiles,
replace les OBB dans l'image complete, retire les doublons inter-tuiles,
puis apparie predictions et annotations une a une.

Le calcul est volontairement separe du notebook : il peut durer longtemps,
est sauvegarde image par image et reprend sans perdre les images terminees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.ops import xyxyxyxy2xywhr


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "outputs" / "dota_v15_tiles_1024"
METADATA_DIR = DATASET_DIR / "metadata"
MANIFEST_PATH = METADATA_DIR / "tile_manifest.csv"
OBJECT_ANNOTATIONS_PATH = METADATA_DIR / "object_annotations.csv"
CLASS_MAPPING_PATH = METADATA_DIR / "class_mapping.csv"
DATASET_CONFIGURATION_PATH = METADATA_DIR / "configuration.json"
VALIDATION_LABELS_DIR = PROJECT_DIR / "DOTA 1.5" / "Val"
DEFAULT_CHECKPOINT = (
    PROJECT_DIR
    / "outputs"
    / "remote_checkpoints"
    / "rtx5080_best_v15_current.pt"
)
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "outputs" / "causal_yolo_outcomes"

DOTA_V15_CLASS_NAMES = [
    "plane",
    "ship",
    "storage-tank",
    "baseball-diamond",
    "tennis-court",
    "basketball-court",
    "ground-track-field",
    "harbor",
    "bridge",
    "large-vehicle",
    "small-vehicle",
    "helicopter",
    "roundabout",
    "soccer-ball-field",
    "swimming-pool",
    "container-crane",
]

# Ces valeurs ont ete decidees avant de regarder l'effet causal.
TILE_SIZE = 1024
MINIMUM_CONFIDENCE = 0.25
# Ultralytics applique un test strict > au seuil. On predit legerement sous
# 0,25, puis on applique nous-memes la regle exacte confidence >= 0,25.
INFERENCE_CONFIDENCE = 0.249
TILE_NMS_PROBIOU = 0.70
SOURCE_NMS_PROBIOU = 0.70
MATCHING_PROBIOU = 0.50
MAX_DETECTIONS_PER_TILE = 5000
SIMILARITY_BLOCK_ROWS = 256

EXPECTED_CAUSAL_OBJECTS = 66968
EXPECTED_CAUSAL_IMAGES = 375
EXPECTED_CAUSAL_TILES = 3059

TILE_PREDICTION_COLUMNS = [
    "source_prediction_id",
    "tile_id",
    "source_image_id",
    "local_prediction_index",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "x3",
    "y3",
    "x4",
    "y4",
]
SUPPRESSION_COLUMNS = [
    "source_image_id",
    "suppressed_prediction_id",
    "kept_prediction_id",
    "class_id",
    "class_name",
    "probiou",
]
OUTCOME_COLUMNS = [
    "source_object_id",
    "source_image_id",
    "object_index",
    "class_id",
    "class_name",
    "difficult",
    "detected",
    "matched_prediction_id",
    "matched_confidence",
    "matched_probiou",
]
FALSE_POSITIVE_COLUMNS = TILE_PREDICTION_COLUMNS + ["reason"]


def parse_arguments() -> argparse.Namespace:
    """Lit une commande check, smoke ou run."""

    parser = argparse.ArgumentParser(
        description="Construit Y=1/0 pour l'analyse causale YOLO DOTA-v1.5."
    )
    parser.add_argument("mode", choices=["check", "smoke", "run"])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint OBB 16 classes a figer pour Y.",
    )
    parser.add_argument(
        "--run-name",
        default="q5_yolo_v15_final",
        help="Nom du dossier d'artefacts. Un smoke ajoute automatiquement _smoke.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--image-id",
        default=None,
        help="Image causale precise pour le mode smoke.",
    )
    return parser.parse_args()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calcule une empreinte stable sans charger tout le fichier en RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def labels_sha256(image_ids: list[str]) -> str:
    """Empreinte collective des LabelTxt utilises par l'analyse."""

    digest = hashlib.sha256()
    for image_id in sorted(image_ids):
        label_path = VALIDATION_LABELS_DIR / f"{image_id}.txt"
        digest.update(label_path.name.encode("utf-8"))
        digest.update(file_sha256(label_path).encode("ascii"))
    return digest.hexdigest()


def atomic_write_json(data: dict, path: Path) -> None:
    """Ecrit un JSON complet avant de remplacer l'ancienne version."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def atomic_write_csv(table: pd.DataFrame, path: Path) -> None:
    """Evite qu'une interruption laisse un CSV partiel considere valide."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    table.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def dataframe_with_columns(records: list[dict], columns: list[str]) -> pd.DataFrame:
    """Conserve le schema CSV meme quand une image ne produit aucune ligne."""

    return pd.DataFrame.from_records(records, columns=columns)


def normalize_model_names(names) -> list[str]:
    """Transforme la liste ou le dictionnaire Ultralytics en liste ordonnee."""

    if isinstance(names, dict):
        normalized = []
        for class_id in range(len(names)):
            if class_id in names:
                normalized.append(str(names[class_id]))
            elif str(class_id) in names:
                normalized.append(str(names[str(class_id)]))
            else:
                raise ValueError(f"Nom de classe absent pour l'identifiant {class_id}.")
        return normalized
    return [str(value) for value in names]


def load_causal_population() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruit la population choisie avant Y depuis les metadonnees."""

    required_paths = [
        MANIFEST_PATH,
        OBJECT_ANNOTATIONS_PATH,
        CLASS_MAPPING_PATH,
        DATASET_CONFIGURATION_PATH,
        VALIDATION_LABELS_DIR,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Fichiers DOTA-v1.5 absents: {missing_paths}")

    dataset_configuration = json.loads(
        DATASET_CONFIGURATION_PATH.read_text(encoding="utf-8")
    )
    if dataset_configuration.get("dataset_version") != "1.5":
        raise ValueError("Le dossier derive n'est pas declare DOTA-v1.5.")
    class_mapping = pd.read_csv(CLASS_MAPPING_PATH).sort_values("yolo_class_id")
    observed_class_names = class_mapping["class_name"].astype(str).tolist()
    if observed_class_names != DOTA_V15_CLASS_NAMES:
        raise ValueError("Le mapping des classes du dataset DOTA-v1.5 a change.")

    object_columns = [
        "source_object_id",
        "split",
        "source_image_id",
        "object_index",
        "class_name",
        "difficult",
        "image_source",
        "gsd",
    ]
    tiled_objects = pd.read_csv(
        OBJECT_ANNOTATIONS_PATH,
        usecols=object_columns,
    )
    tiled_objects = tiled_objects.loc[
        tiled_objects["split"].eq("validation")
    ].copy()

    # Un objet peut apparaitre dans plusieurs tuiles. Ses caracteristiques
    # sources doivent rester identiques avant de ne garder qu'une ligne.
    stable_columns = [
        "source_image_id",
        "object_index",
        "class_name",
        "difficult",
        "image_source",
        "gsd",
    ]
    inconsistent = (
        tiled_objects.groupby("source_object_id", sort=False)[stable_columns]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if inconsistent.any():
        bad_ids = inconsistent.index[inconsistent].tolist()[:5]
        raise ValueError(f"Metadonnees objet incoherentes: {bad_ids}")

    unique_objects = tiled_objects.drop_duplicates(
        "source_object_id", keep="first"
    ).reset_index(drop=True)
    causal_objects = unique_objects.loc[
        unique_objects["image_source"].eq("GoogleEarth")
        & unique_objects["gsd"].notna()
        & unique_objects["class_name"].ne("container-crane")
    ].copy()
    causal_objects = causal_objects.sort_values(
        ["source_image_id", "object_index"], kind="mergesort"
    ).reset_index(drop=True)

    if len(causal_objects) != EXPECTED_CAUSAL_OBJECTS:
        raise ValueError(
            f"Population causale: {len(causal_objects)} objets au lieu de "
            f"{EXPECTED_CAUSAL_OBJECTS}."
        )
    if not causal_objects["source_object_id"].is_unique:
        raise ValueError("La population causale contient des IDs dupliques.")
    causal_image_ids = sorted(causal_objects["source_image_id"].unique())
    if len(causal_image_ids) != EXPECTED_CAUSAL_IMAGES:
        raise ValueError(
            f"Population causale: {len(causal_image_ids)} images au lieu de "
            f"{EXPECTED_CAUSAL_IMAGES}."
        )

    manifest_columns = [
        "tile_id",
        "split",
        "source_image_id",
        "source_width",
        "source_height",
        "x_start",
        "y_start",
        "content_width",
        "content_height",
        "tile_image_path",
    ]
    manifest = pd.read_csv(MANIFEST_PATH, usecols=manifest_columns)
    manifest = manifest.loc[
        manifest["split"].eq("validation")
        & manifest["source_image_id"].isin(causal_image_ids)
    ].copy()
    manifest = manifest.sort_values(
        ["source_image_id", "y_start", "x_start"], kind="mergesort"
    ).reset_index(drop=True)
    if len(manifest) != EXPECTED_CAUSAL_TILES:
        raise ValueError(
            f"Population causale: {len(manifest)} tuiles au lieu de "
            f"{EXPECTED_CAUSAL_TILES}."
        )
    if not manifest["tile_id"].is_unique:
        raise ValueError("Le manifeste causal contient des tuiles dupliquees.")
    missing_images = sorted(set(causal_image_ids) - set(manifest["source_image_id"]))
    if missing_images:
        raise ValueError(f"Images sans tuile: {missing_images[:5]}")

    tile_paths = manifest["tile_image_path"].map(
        lambda value: DATASET_DIR / Path(str(value))
    )
    missing_tiles = [str(path) for path in tile_paths if not path.is_file()]
    if missing_tiles:
        raise FileNotFoundError(f"Tuiles PNG absentes: {missing_tiles[:5]}")

    return causal_objects, manifest


def clip_vertices_to_image(
    vertices: list[tuple[float, float]], width: int, height: int
) -> list[tuple[float, float]]:
    """Reproduit le bornage applique pendant la generation du dataset."""

    return [
        (
            min(max(float(x_value), 0.0), float(width)),
            min(max(float(y_value), 0.0), float(height)),
        )
        for x_value, y_value in vertices
    ]


def is_strictly_convex_quadrilateral(vertices) -> bool:
    """Rejette les memes quadrilateres degeneres que generate_tiles.py."""

    if len(vertices) != 4:
        return False
    cross_products = []
    for index in range(4):
        x1, y1 = vertices[index]
        x2, y2 = vertices[(index + 1) % 4]
        x3, y3 = vertices[(index + 2) % 4]
        cross_products.append(
            (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        )
    return all(value > 0 for value in cross_products) or all(
        value < 0 for value in cross_products
    )


def parse_source_ground_truth(
    image_id: str,
    image_width: int,
    image_height: int,
) -> pd.DataFrame:
    """Lit toutes les annotations officielles valides d'une image source."""

    label_path = VALIDATION_LABELS_DIR / f"{image_id}.txt"
    lines = label_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"En-tete DOTA absent dans {label_path}.")
    if not lines[0].startswith("imagesource:") or not lines[1].startswith("gsd:"):
        raise ValueError(f"En-tete DOTA invalide dans {label_path}.")

    records = []
    for object_index, line in enumerate(lines[2:]):
        fields = line.split()
        if len(fields) != 10:
            raise ValueError(
                f"{label_path}, objet {object_index}: {len(fields)} champs."
            )
        coordinates = [float(value) for value in fields[:8]]
        vertices = [
            (coordinates[index], coordinates[index + 1])
            for index in range(0, 8, 2)
        ]
        vertices = clip_vertices_to_image(vertices, image_width, image_height)
        if not is_strictly_convex_quadrilateral(vertices):
            continue
        class_name = fields[8]
        if class_name not in DOTA_V15_CLASS_NAMES:
            raise ValueError(f"Classe inconnue dans {label_path}: {class_name}")
        class_id = DOTA_V15_CLASS_NAMES.index(class_name)
        difficult = int(fields[9])
        if difficult not in (0, 1):
            raise ValueError(f"Difficulte invalide dans {label_path}.")
        record = {
            "source_object_id": f"validation__{image_id}__object_{object_index}",
            "source_image_id": image_id,
            "object_index": object_index,
            "class_id": class_id,
            "class_name": class_name,
            "difficult": difficult,
        }
        for vertex_number, (x_value, y_value) in enumerate(vertices, start=1):
            record[f"x{vertex_number}"] = x_value
            record[f"y{vertex_number}"] = y_value
        records.append(record)
    ground_truth = pd.DataFrame.from_records(records)
    if ground_truth.empty or not ground_truth["source_object_id"].is_unique:
        raise ValueError(f"Ground truth vide ou duplique pour {image_id}.")
    return ground_truth


def vertices_to_xywhr(table: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """Convertit les huit colonnes de sommets au format OBB Ultralytics."""

    if table.empty:
        return torch.empty((0, 5), dtype=torch.float32, device=device)
    coordinate_columns = [
        "x1",
        "y1",
        "x2",
        "y2",
        "x3",
        "y3",
        "x4",
        "y4",
    ]
    vertices = torch.as_tensor(
        table[coordinate_columns]
        .to_numpy(dtype=np.float32, copy=True)
        .reshape(-1, 4, 2),
        dtype=torch.float32,
        device=device,
    )
    return xyxyxyxy2xywhr(vertices)


def compute_overlap_adjacency(
    boxes: torch.Tensor,
    threshold: float,
    block_rows: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Calcule par blocs les recouvrements avec les boites placees apres.

    Le bloc evite de creer une matrice N x N complete. L'ordre des boites est
    deja l'ordre de priorite de la NMS.
    """

    box_count = len(boxes)
    target_indices = [np.empty(0, dtype=np.int64) for _ in range(box_count)]
    target_scores = [np.empty(0, dtype=np.float32) for _ in range(box_count)]
    for row_start in range(0, box_count, block_rows):
        row_stop = min(row_start + block_rows, box_count)
        similarities = batch_probiou(boxes[row_start:row_stop], boxes)
        similarities = similarities.detach().cpu().numpy()
        for local_row, global_row in enumerate(range(row_start, row_stop)):
            valid_columns = np.flatnonzero(
                (similarities[local_row] >= threshold)
                & (np.arange(box_count) > global_row)
            )
            target_indices[global_row] = valid_columns.astype(np.int64, copy=False)
            target_scores[global_row] = similarities[
                local_row, valid_columns
            ].astype(np.float32, copy=False)
        del similarities
    return target_indices, target_scores


def apply_source_nms(
    predictions: pd.DataFrame,
    similarity_device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retire les doublons inter-tuiles avec une NMS OBB par classe."""

    if predictions.empty:
        return (
            dataframe_with_columns([], TILE_PREDICTION_COLUMNS),
            dataframe_with_columns([], SUPPRESSION_COLUMNS),
        )

    kept_ids: list[str] = []
    suppression_records: list[dict] = []
    for class_id, class_rows in predictions.groupby("class_id", sort=True):
        ordered = class_rows.sort_values(
            ["confidence", "source_prediction_id"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        boxes = vertices_to_xywhr(ordered, similarity_device)
        targets, scores = compute_overlap_adjacency(
            boxes,
            SOURCE_NMS_PROBIOU,
            SIMILARITY_BLOCK_ROWS,
        )
        suppressed = np.zeros(len(ordered), dtype=bool)
        for keeper_position in range(len(ordered)):
            if suppressed[keeper_position]:
                continue
            keeper = ordered.iloc[keeper_position]
            kept_ids.append(str(keeper["source_prediction_id"]))
            for target_position, probiou in zip(
                targets[keeper_position], scores[keeper_position], strict=True
            ):
                if suppressed[target_position]:
                    continue
                suppressed[target_position] = True
                target = ordered.iloc[int(target_position)]
                suppression_records.append(
                    {
                        "source_image_id": keeper["source_image_id"],
                        "suppressed_prediction_id": target[
                            "source_prediction_id"
                        ],
                        "kept_prediction_id": keeper["source_prediction_id"],
                        "class_id": int(class_id),
                        "class_name": keeper["class_name"],
                        "probiou": float(probiou),
                    }
                )
        del boxes, targets, scores

    kept = predictions.loc[
        predictions["source_prediction_id"].isin(kept_ids)
    ].copy()
    kept_order = {prediction_id: index for index, prediction_id in enumerate(kept_ids)}
    kept["_nms_order"] = kept["source_prediction_id"].map(kept_order)
    kept = kept.sort_values("_nms_order").drop(columns="_nms_order").reset_index(drop=True)
    suppressions = dataframe_with_columns(
        suppression_records, SUPPRESSION_COLUMNS
    )
    if len(kept) + len(suppressions) != len(predictions):
        raise ValueError("La NMS source a perdu ou duplique une prediction.")
    return kept, suppressions


def collect_valid_matching_pairs(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    similarity_device: torch.device,
) -> list[tuple[float, float, str, str, int, int]]:
    """Produit seulement les paires meme classe dont ProbIoU atteint 0,50."""

    candidate_pairs: list[tuple[float, float, str, str, int, int]] = []
    shared_classes = sorted(
        set(ground_truth["class_id"]).intersection(predictions["class_id"])
    )
    for class_id in shared_classes:
        gt_class = ground_truth.loc[
            ground_truth["class_id"].eq(class_id)
        ].sort_values("source_object_id", kind="mergesort").reset_index()
        pred_class = predictions.loc[
            predictions["class_id"].eq(class_id)
        ].sort_values("source_prediction_id", kind="mergesort").reset_index()
        gt_boxes = vertices_to_xywhr(gt_class, similarity_device)
        pred_boxes = vertices_to_xywhr(pred_class, similarity_device)
        for row_start in range(0, len(gt_class), SIMILARITY_BLOCK_ROWS):
            row_stop = min(row_start + SIMILARITY_BLOCK_ROWS, len(gt_class))
            similarities = batch_probiou(
                gt_boxes[row_start:row_stop], pred_boxes
            ).detach().cpu().numpy()
            valid_rows, valid_columns = np.where(
                similarities >= MATCHING_PROBIOU
            )
            for local_row, pred_position in zip(
                valid_rows, valid_columns, strict=True
            ):
                gt_position = row_start + int(local_row)
                gt_row = gt_class.iloc[gt_position]
                pred_row = pred_class.iloc[int(pred_position)]
                candidate_pairs.append(
                    (
                        float(similarities[local_row, pred_position]),
                        float(pred_row["confidence"]),
                        str(pred_row["source_prediction_id"]),
                        str(gt_row["source_object_id"]),
                        int(pred_row["index"]),
                        int(gt_row["index"]),
                    )
                )
            del similarities
        del gt_boxes, pred_boxes
    return candidate_pairs


def match_predictions_to_ground_truth(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    similarity_device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attribue une seule prediction a une seule annotation officielle."""

    candidate_pairs = collect_valid_matching_pairs(
        ground_truth, predictions, similarity_device
    )
    # Le protocole Q4 fige d'abord la meilleure ProbIoU. La confiance, puis
    # les deux identifiants, rendent les rares egalites deterministes.
    candidate_pairs.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))

    used_prediction_ids: set[str] = set()
    used_object_ids: set[str] = set()
    accepted_by_object: dict[str, tuple[str, float, float]] = {}
    for probiou, confidence, prediction_id, object_id, _, _ in candidate_pairs:
        if prediction_id in used_prediction_ids or object_id in used_object_ids:
            continue
        used_prediction_ids.add(prediction_id)
        used_object_ids.add(object_id)
        accepted_by_object[object_id] = (prediction_id, confidence, probiou)

    outcome_records = []
    for gt_row in ground_truth.itertuples(index=False):
        match = accepted_by_object.get(gt_row.source_object_id)
        outcome_records.append(
            {
                "source_object_id": gt_row.source_object_id,
                "source_image_id": gt_row.source_image_id,
                "object_index": int(gt_row.object_index),
                "class_id": int(gt_row.class_id),
                "class_name": gt_row.class_name,
                "difficult": int(gt_row.difficult),
                "detected": int(match is not None),
                "matched_prediction_id": None if match is None else match[0],
                "matched_confidence": np.nan if match is None else match[1],
                "matched_probiou": np.nan if match is None else match[2],
            }
        )
    outcomes = dataframe_with_columns(outcome_records, OUTCOME_COLUMNS)

    false_positives = predictions.loc[
        ~predictions["source_prediction_id"].isin(used_prediction_ids)
    ].copy()
    false_positives["reason"] = "unmatched_after_one_to_one_matching"
    false_positives = false_positives.reindex(columns=FALSE_POSITIVE_COLUMNS)

    if outcomes["source_object_id"].duplicated().any():
        raise ValueError("Un objet apparait plusieurs fois dans les resultats.")
    matched = outcomes.loc[outcomes["detected"].eq(1)]
    if matched["matched_prediction_id"].duplicated().any():
        raise ValueError("Une prediction a ete attribuee plusieurs fois.")
    if not matched["matched_probiou"].ge(MATCHING_PROBIOU).all():
        raise ValueError("Une paire acceptee se trouve sous le seuil ProbIoU.")
    return outcomes, false_positives


def infer_source_image(
    model: YOLO,
    image_manifest: pd.DataFrame,
    device: str,
    batch: int,
) -> tuple[pd.DataFrame, dict]:
    """Predit toutes les tuiles d'une image puis remappe leurs sommets."""

    tile_paths = [
        DATASET_DIR / Path(str(value))
        for value in image_manifest["tile_image_path"]
    ]
    rows_by_tile = {
        row.tile_id: row for row in image_manifest.itertuples(index=False)
    }
    prediction_records: list[dict] = []
    raw_counts: dict[str, int] = {}
    padding_rejections = 0

    seen_tile_ids: set[str] = set()
    # Pour une liste de fichiers, Ultralytics peut ignorer son argument
    # `batch`. Nous decoupons donc explicitement la liste : --batch 1 veut
    # reellement dire une seule tuile chargee a la fois.
    for batch_start in range(0, len(tile_paths), batch):
        batch_paths = tile_paths[batch_start : batch_start + batch]
        results = model.predict(
            source=[str(path) for path in batch_paths],
            stream=True,
            imgsz=TILE_SIZE,
            conf=INFERENCE_CONFIDENCE,
            iou=TILE_NMS_PROBIOU,
            max_det=MAX_DETECTIONS_PER_TILE,
            agnostic_nms=False,
            device=device,
            verbose=False,
            save=False,
        )
        for result in results:
            tile_id = Path(result.path).stem
            if tile_id not in rows_by_tile:
                raise ValueError(f"Resultat inattendu pour la tuile {tile_id}.")
            if tile_id in seen_tile_ids:
                raise ValueError(f"Resultat duplique pour la tuile {tile_id}.")
            seen_tile_ids.add(tile_id)
            tile_row = rows_by_tile[tile_id]
            obb = result.obb
            raw_count = 0 if obb is None else len(obb)
            raw_counts[tile_id] = raw_count
            if raw_count >= MAX_DETECTIONS_PER_TILE:
                raise RuntimeError(
                    f"{tile_id} atteint max_det={MAX_DETECTIONS_PER_TILE}; "
                    "les predictions sont potentiellement censurees."
                )
            if obb is None or raw_count == 0:
                continue

            vertices = obb.xyxyxyxy.detach().cpu().numpy().astype(np.float64)
            confidences = obb.conf.detach().cpu().numpy().astype(np.float64)
            class_ids = obb.cls.detach().cpu().numpy().astype(np.int64)
            for local_index, (local_vertices, confidence, class_id) in enumerate(
                zip(vertices, confidences, class_ids, strict=True)
            ):
                if float(confidence) < MINIMUM_CONFIDENCE:
                    continue
                center_x, center_y = local_vertices.mean(axis=0)
                if (
                    center_x < 0
                    or center_y < 0
                    or center_x >= float(tile_row.content_width)
                    or center_y >= float(tile_row.content_height)
                ):
                    padding_rejections += 1
                    continue
                source_vertices = local_vertices.copy()
                source_vertices[:, 0] += float(tile_row.x_start)
                source_vertices[:, 1] += float(tile_row.y_start)
                prediction_id = f"{tile_id}__prediction_{local_index}"
                record = {
                    "source_prediction_id": prediction_id,
                    "tile_id": tile_id,
                    "source_image_id": tile_row.source_image_id,
                    "local_prediction_index": local_index,
                    "class_id": int(class_id),
                    "class_name": DOTA_V15_CLASS_NAMES[int(class_id)],
                    "confidence": float(confidence),
                }
                for vertex_number, (x_value, y_value) in enumerate(
                    source_vertices, start=1
                ):
                    record[f"x{vertex_number}"] = float(x_value)
                    record[f"y{vertex_number}"] = float(y_value)
                prediction_records.append(record)
            del result, obb, vertices, confidences, class_ids

    expected_tile_ids = set(rows_by_tile)
    if seen_tile_ids != expected_tile_ids:
        missing = sorted(expected_tile_ids - seen_tile_ids)
        raise RuntimeError(f"Inference incomplete; tuiles absentes: {missing[:5]}")
    predictions = dataframe_with_columns(
        prediction_records, TILE_PREDICTION_COLUMNS
    )
    if predictions["source_prediction_id"].duplicated().any():
        raise ValueError("Identifiants de predictions dupliques.")
    audit = {
        "tile_count": len(image_manifest),
        "raw_tile_prediction_count": int(sum(raw_counts.values())),
        "retained_tile_prediction_count": len(predictions),
        "padding_rejection_count": padding_rejections,
        "maximum_raw_predictions_on_one_tile": int(max(raw_counts.values(), default=0)),
    }
    return predictions, audit


def load_and_validate_model(checkpoint: Path) -> tuple[YOLO, dict]:
    """Charge sur CPU et refuse un modele qui n'est pas l'OBB v1.5 attendu."""

    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint absent: {checkpoint}. Copie d'abord le meilleur "
            "checkpoint RTX 5080 sous ce nom, ou utilise --checkpoint."
        )
    model = YOLO(str(checkpoint), task="obb")
    model_task = str(getattr(model, "task", ""))
    model_names = normalize_model_names(model.names)
    if model_task != "obb":
        raise ValueError(f"Le checkpoint est de type {model_task}, pas OBB.")
    if model_names != DOTA_V15_CLASS_NAMES:
        raise ValueError(
            "Les classes du checkpoint ne correspondent pas exactement a DOTA-v1.5."
        )
    checkpoint_data = getattr(model, "ckpt", None)
    metadata = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_epoch_zero_based": None,
        "checkpoint_best_fitness": None,
        "checkpoint_run_name": None,
    }
    if isinstance(checkpoint_data, dict):
        metadata["checkpoint_epoch_zero_based"] = checkpoint_data.get("epoch")
        best_fitness = checkpoint_data.get("best_fitness")
        if torch.is_tensor(best_fitness):
            best_fitness = best_fitness.detach().cpu().item()
        metadata["checkpoint_best_fitness"] = best_fitness
        train_args = checkpoint_data.get("train_args") or {}
        if isinstance(train_args, dict):
            metadata["checkpoint_run_name"] = train_args.get("name")
    return model, metadata


def build_protocol(
    checkpoint_metadata: dict,
    causal_objects: pd.DataFrame,
    manifest: pd.DataFrame,
    prediction_device: str,
    prediction_batch: int,
) -> dict:
    """Fige toutes les decisions capables de modifier Y."""

    image_ids = sorted(causal_objects["source_image_id"].unique())
    reproducible = {
        **checkpoint_metadata,
        "dataset_version": "DOTA-v1.5",
        "dataset_configuration_sha256": file_sha256(DATASET_CONFIGURATION_PATH),
        "tile_manifest_sha256": file_sha256(MANIFEST_PATH),
        "object_annotations_sha256": file_sha256(OBJECT_ANNOTATIONS_PATH),
        "class_mapping_sha256": file_sha256(CLASS_MAPPING_PATH),
        "selected_labeltxt_sha256": labels_sha256(image_ids),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "class_names": DOTA_V15_CLASS_NAMES,
        "causal_object_count": len(causal_objects),
        "causal_image_count": len(image_ids),
        "causal_tile_count": len(manifest),
        "minimum_confidence": MINIMUM_CONFIDENCE,
        "inference_confidence": INFERENCE_CONFIDENCE,
        "tile_nms_probiou": TILE_NMS_PROBIOU,
        "source_nms_probiou": SOURCE_NMS_PROBIOU,
        "matching_probiou": MATCHING_PROBIOU,
        "max_detections_per_tile": MAX_DETECTIONS_PER_TILE,
        "prediction_device": str(prediction_device),
        "prediction_batch": int(prediction_batch),
        "similarity_device": "cpu",
        "similarity_dtype": "float32",
        "image_size": TILE_SIZE,
        "agnostic_nms": False,
        "padding_filter": "reject_if_obb_center_outside_real_tile_content",
        "matching_order": [
            "probiou_desc",
            "confidence_desc",
            "prediction_id_asc",
            "source_object_id_asc",
        ],
        "source_nms_order": ["confidence_desc", "prediction_id_asc"],
        "global_processing_order": [
            "causal_object_count_desc",
            "source_image_id_asc",
        ],
        "causal_population_rule": (
            "validation tiled objects; image_source=GoogleEarth; gsd known; "
            "container-crane excluded"
        ),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
    }
    protocol_id = hashlib.sha256(
        json.dumps(reproducible, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return {
        "protocol_id": protocol_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **reproducible,
    }


def choose_smoke_image(causal_objects: pd.DataFrame, requested: str | None) -> str:
    """Choisit une image de taille mediane si aucune n'est demandee."""

    counts = causal_objects.groupby("source_image_id").size().sort_index()
    if requested is not None:
        if requested not in counts.index:
            raise ValueError(f"{requested} n'appartient pas au support causal.")
        return requested
    median_count = float(counts.median())
    candidates = pd.DataFrame(
        {
            "source_image_id": counts.index,
            "object_count": counts.to_numpy(),
        }
    )
    candidates["distance_to_median"] = (
        candidates["object_count"] - median_count
    ).abs()
    return str(
        candidates.sort_values(
            ["distance_to_median", "source_image_id"], kind="mergesort"
        ).iloc[0]["source_image_id"]
    )


def validate_or_write_protocol(protocol: dict, output_dir: Path) -> None:
    """Interdit de melanger deux checkpoints ou deux seuils dans un run."""

    protocol_path = output_dir / "q5_protocol.json"
    if protocol_path.is_file():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous.get("protocol_id") != protocol["protocol_id"]:
            raise ValueError(
                f"Le dossier {output_dir} appartient a un autre protocole. "
                "Choisis un autre --run-name."
            )
        return
    atomic_write_json(protocol, protocol_path)


def image_is_complete(image_dir: Path, protocol_id: str) -> bool:
    """Une image est reprise uniquement si son dernier marqueur est valide."""

    summary_path = image_dir / "summary.json"
    required_names = [
        "q5_tile_predictions.csv",
        "q5_source_predictions.csv",
        "q5_source_nms_suppressions.csv",
        "q5_all_gt_outcomes.csv",
        "q5_false_positives.csv",
    ]
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if summary.get("completed") is not True or summary.get("protocol_id") != protocol_id:
        return False
    csv_manifest = summary.get("csv_manifest")
    if not isinstance(csv_manifest, dict) or set(csv_manifest) != set(required_names):
        return False
    for file_name in required_names:
        csv_path = image_dir / file_name
        expected = csv_manifest[file_name]
        if not csv_path.is_file():
            return False
        if csv_path.stat().st_size != expected.get("size_bytes"):
            return False
        if file_sha256(csv_path) != expected.get("sha256"):
            return False
    return True


def build_csv_manifest(image_dir: Path, tables: dict[str, pd.DataFrame]) -> dict:
    """Lie le marqueur final au contenu exact de chaque CSV d'une image."""

    manifest = {}
    for file_name, table in tables.items():
        csv_path = image_dir / file_name
        manifest[file_name] = {
            "rows": len(table),
            "columns": table.columns.tolist(),
            "size_bytes": csv_path.stat().st_size,
            "sha256": file_sha256(csv_path),
        }
    return manifest


def process_one_image(
    model: YOLO,
    source_image_id: str,
    image_manifest: pd.DataFrame,
    image_causal_objects: pd.DataFrame,
    image_dir: Path,
    protocol_id: str,
    device: str,
    batch: int,
    similarity_device: torch.device,
) -> dict:
    """Execute et sauvegarde toute la chaine pour une image source."""

    dimensions = image_manifest[["source_width", "source_height"]].drop_duplicates()
    if len(dimensions) != 1:
        raise ValueError(f"Dimensions incoherentes pour {source_image_id}.")
    width = int(dimensions.iloc[0]["source_width"])
    height = int(dimensions.iloc[0]["source_height"])
    ground_truth = parse_source_ground_truth(source_image_id, width, height)
    missing_causal_ids = sorted(
        set(image_causal_objects["source_object_id"])
        - set(ground_truth["source_object_id"])
    )
    if missing_causal_ids:
        raise ValueError(
            f"Objets causaux absents du GT {source_image_id}: {missing_causal_ids[:5]}"
        )
    causal_identity_check = image_causal_objects[
        ["source_object_id", "object_index", "class_name", "difficult"]
    ].merge(
        ground_truth[
            ["source_object_id", "object_index", "class_name", "difficult"]
        ],
        on="source_object_id",
        how="left",
        validate="one_to_one",
        suffixes=("_metadata", "_labeltxt"),
    )
    for column_name in ["object_index", "class_name", "difficult"]:
        if not causal_identity_check[f"{column_name}_metadata"].equals(
            causal_identity_check[f"{column_name}_labeltxt"]
        ):
            raise ValueError(
                f"Identite {column_name} incoherente pour {source_image_id}."
            )

    tile_predictions, inference_audit = infer_source_image(
        model, image_manifest, device, batch
    )
    source_predictions, suppressions = apply_source_nms(
        tile_predictions, similarity_device
    )
    all_outcomes, false_positives = match_predictions_to_ground_truth(
        ground_truth, source_predictions, similarity_device
    )
    causal_outcomes = all_outcomes.loc[
        all_outcomes["source_object_id"].isin(
            set(image_causal_objects["source_object_id"])
        )
    ]
    if len(causal_outcomes) != len(image_causal_objects):
        raise ValueError(f"Resultats causaux incomplets pour {source_image_id}.")

    # Le marqueur summary.json est toujours ecrit en dernier. Une coupure
    # avant ce point force simplement le recalcul de cette image.
    csv_tables = {
        "q5_tile_predictions.csv": tile_predictions,
        "q5_source_predictions.csv": source_predictions,
        "q5_source_nms_suppressions.csv": suppressions,
        "q5_all_gt_outcomes.csv": all_outcomes,
        "q5_false_positives.csv": false_positives,
    }
    for file_name, table in csv_tables.items():
        atomic_write_csv(table, image_dir / file_name)
    summary = {
        "completed": True,
        "protocol_id": protocol_id,
        "source_image_id": source_image_id,
        **inference_audit,
        "source_prediction_count": len(source_predictions),
        "source_nms_suppression_count": len(suppressions),
        "official_valid_gt_count": len(ground_truth),
        "causal_object_count": len(causal_outcomes),
        "causal_detected_count": int(causal_outcomes["detected"].sum()),
        "false_positive_count": len(false_positives),
        "csv_manifest": build_csv_manifest(image_dir, csv_tables),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(summary, image_dir / "summary.json")
    return summary


def concatenate_csv_files(paths: list[Path], columns: list[str]) -> pd.DataFrame:
    """Assemble les resultats par image sans imposer leur presence en RAM avant."""

    tables = []
    for path in paths:
        table = pd.read_csv(path)
        if table.columns.tolist() != columns:
            raise ValueError(
                f"Schema CSV inattendu dans {path}: {table.columns.tolist()}"
            )
        tables.append(table)
    if not tables:
        return pd.DataFrame(columns=columns)
    return pd.concat(tables, ignore_index=True)


def finalize_outputs(
    output_dir: Path,
    image_ids: list[str],
    causal_objects: pd.DataFrame,
    protocol_id: str,
    global_run: bool,
) -> None:
    """Assemble les fichiers par image et applique les assertions finales."""

    image_dirs = [output_dir / "per_image" / image_id for image_id in image_ids]
    summaries = []
    for image_dir in image_dirs:
        summary = json.loads((image_dir / "summary.json").read_text(encoding="utf-8"))
        if summary.get("protocol_id") != protocol_id or not summary.get("completed"):
            raise ValueError(f"Image non terminee: {image_dir.name}")
        summaries.append(summary)

    all_outcomes = concatenate_csv_files(
        [directory / "q5_all_gt_outcomes.csv" for directory in image_dirs],
        OUTCOME_COLUMNS,
    )
    causal_ids = set(causal_objects["source_object_id"])
    object_outcomes = all_outcomes.loc[
        all_outcomes["source_object_id"].isin(causal_ids)
    ].copy()
    object_outcomes = object_outcomes.sort_values(
        ["source_image_id", "object_index"], kind="mergesort"
    ).reset_index(drop=True)
    source_predictions = concatenate_csv_files(
        [directory / "q5_source_predictions.csv" for directory in image_dirs],
        TILE_PREDICTION_COLUMNS,
    )
    tile_predictions = concatenate_csv_files(
        [directory / "q5_tile_predictions.csv" for directory in image_dirs],
        TILE_PREDICTION_COLUMNS,
    )
    suppressions = concatenate_csv_files(
        [directory / "q5_source_nms_suppressions.csv" for directory in image_dirs],
        SUPPRESSION_COLUMNS,
    )
    false_positives = concatenate_csv_files(
        [directory / "q5_false_positives.csv" for directory in image_dirs],
        FALSE_POSITIVE_COLUMNS,
    )
    image_summary = pd.DataFrame(summaries).sort_values("source_image_id")

    if tile_predictions["source_prediction_id"].duplicated().any():
        raise ValueError("Predictions de tuiles dupliquees dans l'assemblage.")
    if source_predictions["source_prediction_id"].duplicated().any():
        raise ValueError("Predictions sources dupliquees dans l'assemblage.")
    if all_outcomes["source_object_id"].duplicated().any():
        raise ValueError("Ground truth source duplique dans l'assemblage.")
    if object_outcomes["source_object_id"].duplicated().any():
        raise ValueError("Le resultat final contient des objets dupliques.")
    if set(object_outcomes["detected"].dropna().astype(int).unique()) - {0, 1}:
        raise ValueError("Y contient une valeur differente de 0 ou 1.")
    if object_outcomes["detected"].isna().any():
        raise ValueError("Y contient des valeurs manquantes.")
    if not tile_predictions.empty and not tile_predictions["confidence"].ge(
        MINIMUM_CONFIDENCE
    ).all():
        raise ValueError("Une prediction sauvegardee se trouve sous confiance 0,25.")

    tile_prediction_ids = set(tile_predictions["source_prediction_id"])
    source_prediction_ids = set(source_predictions["source_prediction_id"])
    suppressed_ids = set(suppressions["suppressed_prediction_id"])
    if source_prediction_ids & suppressed_ids:
        raise ValueError("Une prediction est a la fois gardee et supprimee par la NMS.")
    if source_prediction_ids | suppressed_ids != tile_prediction_ids:
        raise ValueError("La NMS ne partitionne pas toutes les predictions de tuiles.")
    if suppressions["suppressed_prediction_id"].duplicated().any():
        raise ValueError("Une prediction a plusieurs suppresseurs NMS.")
    if not set(suppressions["kept_prediction_id"]).issubset(source_prediction_ids):
        raise ValueError("Un suppresseur NMS ne fait pas partie des boites gardees.")

    matched_all = all_outcomes.loc[all_outcomes["detected"].eq(1)].copy()
    if matched_all["matched_prediction_id"].duplicated().any():
        raise ValueError("Une prediction est appariee a plusieurs annotations.")
    false_positive_ids = set(false_positives["source_prediction_id"])
    matched_prediction_ids = set(matched_all["matched_prediction_id"])
    if matched_prediction_ids & false_positive_ids:
        raise ValueError("Une prediction est a la fois TP et faux positif.")
    if matched_prediction_ids | false_positive_ids != source_prediction_ids:
        raise ValueError("Le matching ne partitionne pas les predictions sources.")
    if not matched_all.empty:
        matching_audit = matched_all.merge(
            source_predictions[
                ["source_prediction_id", "class_name", "confidence"]
            ],
            left_on="matched_prediction_id",
            right_on="source_prediction_id",
            how="left",
            validate="one_to_one",
            suffixes=("_gt", "_prediction"),
        )
        if matching_audit["source_prediction_id"].isna().any():
            raise ValueError("Une prediction appariee est absente des sorties source.")
        if not matching_audit["class_name_gt"].equals(
            matching_audit["class_name_prediction"]
        ):
            raise ValueError("Un appariement relie deux classes differentes.")
        if not matching_audit["matched_probiou"].ge(MATCHING_PROBIOU).all():
            raise ValueError("Une paire finale se trouve sous ProbIoU 0,50.")
        if not matching_audit["confidence"].ge(MINIMUM_CONFIDENCE).all():
            raise ValueError("Une paire finale se trouve sous confiance 0,25.")

    if global_run:
        if len(object_outcomes) != EXPECTED_CAUSAL_OBJECTS:
            raise ValueError(
                f"Y contient {len(object_outcomes)} objets au lieu de "
                f"{EXPECTED_CAUSAL_OBJECTS}."
            )
        if object_outcomes["source_image_id"].nunique() != EXPECTED_CAUSAL_IMAGES:
            raise ValueError("Y ne couvre pas les 375 images attendues.")
        if set(object_outcomes["source_object_id"]) != causal_ids:
            raise ValueError("Les IDs finaux ne correspondent pas a la population causale.")

    atomic_write_csv(object_outcomes, output_dir / "q5_object_outcomes.csv")
    atomic_write_csv(tile_predictions, output_dir / "q5_tile_predictions.csv")
    atomic_write_csv(source_predictions, output_dir / "q5_source_predictions.csv")
    atomic_write_csv(suppressions, output_dir / "q5_source_nms_suppressions.csv")
    atomic_write_csv(false_positives, output_dir / "q5_false_positives.csv")
    atomic_write_csv(image_summary, output_dir / "q5_image_summary.csv")


def run_pipeline(arguments: argparse.Namespace) -> None:
    """Controle, teste ou execute la construction reprenable de Y."""

    if arguments.batch <= 0:
        raise ValueError("--batch doit etre strictement positif.")
    checkpoint = arguments.checkpoint.resolve()
    causal_objects, manifest = load_causal_population()
    model, checkpoint_metadata = load_and_validate_model(checkpoint)
    protocol = build_protocol(
        checkpoint_metadata,
        causal_objects,
        manifest,
        arguments.device,
        arguments.batch,
    )

    print("Environnement valide.")
    print("Checkpoint :", checkpoint)
    print("SHA-256 :", checkpoint_metadata["checkpoint_sha256"])
    print("Run du checkpoint :", checkpoint_metadata["checkpoint_run_name"])
    print("Epoque interne :", checkpoint_metadata["checkpoint_epoch_zero_based"])
    print("Population causale :", len(causal_objects), "objets")
    print("Images causales :", causal_objects["source_image_id"].nunique())
    print("Tuiles a predire :", len(manifest))
    print("Protocole :", protocol["protocol_id"])
    if arguments.mode == "check":
        print("Aucune inference n'a ete lancee.")
        return

    global_run = arguments.mode == "run"
    run_name = arguments.run_name if global_run else f"{arguments.run_name}_smoke"
    output_dir = arguments.output_root.resolve() / run_name
    validate_or_write_protocol(protocol, output_dir)

    if global_run:
        # Commencer par les images les plus denses transforme le debut du
        # run en stress test. Un probleme de memoire est alors detecte avant
        # d'avoir calcule des centaines d'images faciles.
        image_counts = (
            causal_objects.groupby("source_image_id", as_index=False)
            .size()
            .rename(columns={"size": "causal_object_count"})
        )
        image_ids = image_counts.sort_values(
            ["causal_object_count", "source_image_id"],
            ascending=[False, True],
            kind="mergesort",
        )["source_image_id"].astype(str).tolist()
    else:
        image_ids = [choose_smoke_image(causal_objects, arguments.image_id)]
        print("Image du smoke test :", image_ids[0])

    # Les calculs de ProbIoU restent sur CPU pour ne pas ajouter un second
    # pic de VRAM pendant l'inference YOLO.
    similarity_device = torch.device("cpu")
    per_image_dir = output_dir / "per_image"
    per_image_dir.mkdir(parents=True, exist_ok=True)

    for image_number, source_image_id in enumerate(image_ids, start=1):
        image_dir = per_image_dir / source_image_id
        if image_is_complete(image_dir, protocol["protocol_id"]):
            print(
                f"[{image_number}/{len(image_ids)}] {source_image_id}: deja termine.",
                flush=True,
            )
            continue
        print(
            f"[{image_number}/{len(image_ids)}] {source_image_id}: inference et matching...",
            flush=True,
        )
        image_manifest = manifest.loc[
            manifest["source_image_id"].eq(source_image_id)
        ].copy()
        image_causal_objects = causal_objects.loc[
            causal_objects["source_image_id"].eq(source_image_id)
        ].copy()
        summary = process_one_image(
            model=model,
            source_image_id=source_image_id,
            image_manifest=image_manifest,
            image_causal_objects=image_causal_objects,
            image_dir=image_dir,
            protocol_id=protocol["protocol_id"],
            device=arguments.device,
            batch=arguments.batch,
            similarity_device=similarity_device,
        )
        print(
            f"  GT causaux={summary['causal_object_count']}, "
            f"detectes={summary['causal_detected_count']}, "
            f"predictions source={summary['source_prediction_count']}",
            flush=True,
        )

    finalize_outputs(
        output_dir,
        image_ids,
        causal_objects,
        protocol["protocol_id"],
        global_run,
    )
    print("Construction de Y terminee :", output_dir)
    print("Resultat principal :", output_dir / "q5_object_outcomes.csv")


def main() -> int:
    """Point d'entree avec message d'erreur lisible dans PowerShell."""

    arguments = parse_arguments()
    try:
        run_pipeline(arguments)
    except Exception as error:
        print(f"ERREUR : {type(error).__name__}: {error}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
