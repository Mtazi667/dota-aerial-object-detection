"""Genere le dataset derive commun a YOLO-OBB et Faster R-CNN."""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box as shapely_box


# ---------------------------------------------------------------------------
# Configuration validee dans le notebook
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_IMAGES_DIR = PROJECT_DIR / "Training Data" / "Images"
TRAIN_LABELS_DIR = PROJECT_DIR / "Training Data" / "LabelTxt"
VALID_IMAGES_DIR = PROJECT_DIR / "Validation Data" / "Images"
VALID_LABELS_DIR = PROJECT_DIR / "Validation Data" / "LabelTxt"
TEST_IMAGES_DIR = PROJECT_DIR / "Testing Images" / "Images"
DOTA_V15_TRAIN_LABELS_DIR = PROJECT_DIR / "DOTA 1.5" / "Train"
DOTA_V15_VALID_LABELS_DIR = PROJECT_DIR / "DOTA 1.5" / "Val"

TILE_SIZE = 1024
TILE_OVERLAP = 200
TILE_STRIDE = TILE_SIZE - TILE_OVERLAP
MIN_VISIBLE_FRACTION = 0.70
TRAIN_BACKGROUND_FRACTION = 0.10
RANDOM_SEED = 42
PADDING_COLOR = (114, 114, 114)

# Cette image reste sur le disque, mais elle est exclue du pipeline derive.
EXCLUDED_TRAIN_IMAGE_IDS = {"P1872"}

DOTA_V10_CLASS_NAMES = [
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
]
DOTA_V15_CLASS_NAMES = DOTA_V10_CLASS_NAMES + ["container-crane"]

# Ces cinq tuiles reproduisent le pilote deja valide visuellement.
PILOT_TILE_IDS = [
    "train__P0000__s1024__x0_1024__y0_1024",
    "train__P0000__s1024__x0_1024__y824_1848",
    "validation__P0004__s1024__x0_824__y0_1024",
    "train__P0020__s1024__x2472_3496__y0_1024",
    "test__P0006__s1024__x0_1024__y0_1024",
]
PILOT_IMAGE_IDS = {
    "train": {"P0000", "P0020"},
    "validation": {"P0004"},
    "test": {"P0006"},
}


def parse_arguments():
    """Lit les options fournies dans la commande PowerShell."""

    parser = argparse.ArgumentParser(
        description="Genere les tuiles DOTA et les annotations des deux modeles."
    )
    parser.add_argument(
        "--dataset-version",
        choices=["1.0", "1.5"],
        default="1.5",
        help="Choisit les annotations DOTA-v1.0 ou DOTA-v1.5.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Genere seulement les cinq tuiles de controle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Remplace le dossier de sortie par defaut.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Valide toute la preparation sans ecrire les PNG.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Lecture et validation des annotations sources
# ---------------------------------------------------------------------------

def parse_dota_label(label_path):
    """Lit un fichier LabelTxt DOTA et retourne ses objets structures."""

    lines = Path(label_path).read_text(encoding="utf-8").splitlines()

    if len(lines) < 2:
        raise ValueError(f"{label_path}: metadonnees absentes.")
    if not lines[0].startswith("imagesource:"):
        raise ValueError(f"{label_path}: en-tete imagesource invalide.")
    if not lines[1].startswith("gsd:"):
        raise ValueError(f"{label_path}: en-tete gsd invalide.")

    image_source = lines[0].split(":", 1)[1].strip()
    gsd_text = lines[1].split(":", 1)[1].strip()
    gsd = None if gsd_text.lower() == "null" else float(gsd_text)

    objects = []
    for object_index, line in enumerate(lines[2:]):
        fields = line.split()
        if len(fields) != 10:
            raise ValueError(
                f"{label_path}, objet {object_index}: "
                f"{len(fields)} champs au lieu de 10."
            )

        coordinates = [float(value) for value in fields[:8]]
        difficult = int(fields[9])
        if difficult not in (0, 1):
            raise ValueError(f"{label_path}: difficult doit valoir 0 ou 1.")

        vertices = [
            (coordinates[index], coordinates[index + 1])
            for index in range(0, 8, 2)
        ]
        objects.append({
            "object_index": object_index,
            "vertices": vertices,
            "class_name": fields[8],
            "difficult": difficult,
        })

    return {
        "image_source": image_source,
        "gsd": gsd,
        "objects": objects,
    }


def clip_vertices_to_image(vertices, image_width, image_height):
    """Construit quatre nouveaux sommets bornes dans l'image."""

    return [
        (
            min(max(float(x), 0.0), float(image_width)),
            min(max(float(y), 0.0), float(image_height)),
        )
        for x, y in vertices
    ]


def compute_polygon_area(vertices):
    """Calcule l'aire d'un polygone avec la formule du lacet."""

    doubled_area = 0.0
    for index, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(index + 1) % len(vertices)]
        doubled_area += x1 * y2 - y1 * x2
    return abs(doubled_area) / 2.0


def compute_obb_orientation(vertices):
    """Retourne l'angle du plus long cote dans l'intervalle [0, 180)."""

    sides = []
    for index, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(index + 1) % len(vertices)]
        length = math.hypot(x2 - x1, y2 - y1)
        sides.append((length, x2 - x1, y2 - y1))

    _, dx, dy = max(sides, key=lambda side: side[0])
    return math.degrees(math.atan2(dy, dx)) % 180.0


def is_strictly_convex_quadrilateral(vertices):
    """Indique si les quatre sommets ordonnes tournent toujours dans le meme sens."""

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

    return (
        all(value > 0 for value in cross_products)
        or all(value < 0 for value in cross_products)
    )


def compute_minimum_rectangle_features(vertices):
    """Calcule les cotes du rectangle minimal utilise comme proxy de taille."""

    points = np.asarray(vertices, dtype=np.float32)
    _, (rectangle_width, rectangle_height), _ = cv2.minAreaRect(points)
    long_side = float(max(rectangle_width, rectangle_height))
    short_side = float(min(rectangle_width, rectangle_height))

    if short_side <= 0:
        raise ValueError("Le rectangle minimal possede un cote nul.")

    return long_side, short_side, long_side / short_side


def discover_images(pilot_mode, dataset_version):
    """Construit la table des images autorisees directement depuis les dossiers."""

    if dataset_version == "1.5":
        train_labels_dir = DOTA_V15_TRAIN_LABELS_DIR
        valid_labels_dir = DOTA_V15_VALID_LABELS_DIR
    else:
        train_labels_dir = TRAIN_LABELS_DIR
        valid_labels_dir = VALID_LABELS_DIR

    split_directories = [
        ("train", TRAIN_IMAGES_DIR, train_labels_dir),
        ("validation", VALID_IMAGES_DIR, valid_labels_dir),
        ("test", TEST_IMAGES_DIR, None),
    ]
    image_records = []

    for split_name, image_dir, label_dir in split_directories:
        for image_path in sorted(image_dir.glob("*.png")):
            image_id = image_path.stem

            if split_name == "train" and image_id in EXCLUDED_TRAIN_IMAGE_IDS:
                continue
            if pilot_mode and image_id not in PILOT_IMAGE_IDS[split_name]:
                continue

            with Image.open(image_path) as image:
                image_width, image_height = image.size

            label_path = None
            if label_dir is not None:
                label_path = label_dir / f"{image_id}.txt"
                if not label_path.is_file():
                    raise FileNotFoundError(label_path)

            image_records.append({
                "split": split_name,
                "source_image_id": image_id,
                "source_image_path": image_path,
                "label_path": label_path,
                "source_width": image_width,
                "source_height": image_height,
            })

    return pd.DataFrame(image_records)


def build_object_table(images_df, class_to_yolo_id):
    """Parse, borne et valide les objets de train et de validation."""

    object_records = []
    rejected_records = []

    annotated_images_df = images_df.loc[
        images_df["split"].isin(["train", "validation"])
    ]

    for image_row in annotated_images_df.itertuples(index=False):
        annotation = parse_dota_label(image_row.label_path)

        for object_data in annotation["objects"]:
            vertices = clip_vertices_to_image(
                object_data["vertices"],
                image_row.source_width,
                image_row.source_height,
            )

            if not is_strictly_convex_quadrilateral(vertices):
                rejected_records.append({
                    "split": image_row.split,
                    "source_image_id": image_row.source_image_id,
                    "object_index": object_data["object_index"],
                })
                continue

            class_name = object_data["class_name"]
            if class_name not in class_to_yolo_id:
                raise ValueError(f"Classe DOTA inconnue: {class_name}")

            area_px2 = compute_polygon_area(vertices)
            if area_px2 <= 0:
                raise ValueError("Une annotation valide possede une aire nulle.")

            x_coordinates = [x for x, _ in vertices]
            y_coordinates = [y for _, y in vertices]
            long_side, short_side, aspect_ratio = (
                compute_minimum_rectangle_features(vertices)
            )
            center_x = sum(x_coordinates) / len(x_coordinates)
            center_y = sum(y_coordinates) / len(y_coordinates)
            object_index = object_data["object_index"]
            source_object_id = (
                f"{image_row.split}__{image_row.source_image_id}"
                f"__object_{object_index}"
            )

            object_records.append({
                "source_object_id": source_object_id,
                "split": image_row.split,
                "source_image_id": image_row.source_image_id,
                "object_index": object_index,
                "class_name": class_name,
                "difficult": object_data["difficult"],
                "image_source": annotation["image_source"],
                "gsd": annotation["gsd"],
                "gsd_missing": annotation["gsd"] is None,
                "vertices": vertices,
                "source_area_px2": area_px2,
                "source_relative_area": (
                    area_px2
                    / (image_row.source_width * image_row.source_height)
                ),
                "source_orientation_deg": compute_obb_orientation(vertices),
                "source_long_side_px": long_side,
                "source_short_side_px": short_side,
                "source_aspect_ratio": aspect_ratio,
                "source_center_x_normalized": (
                    center_x / image_row.source_width
                ),
                "source_center_y_normalized": (
                    center_y / image_row.source_height
                ),
                "min_x": min(x_coordinates),
                "min_y": min(y_coordinates),
                "max_x": max(x_coordinates),
                "max_y": max(y_coordinates),
                "yolo_class_id": class_to_yolo_id[class_name],
                "faster_rcnn_class_id": class_to_yolo_id[class_name] + 1,
            })

    objects_df = pd.DataFrame(object_records)
    rejected_df = pd.DataFrame(rejected_records)

    if objects_df["source_object_id"].duplicated().any():
        raise ValueError("Des identifiants d'objets sources sont dupliques.")

    return objects_df, rejected_df


# ---------------------------------------------------------------------------
# Construction de la grille et intersection objet-tuile
# ---------------------------------------------------------------------------

def compute_tile_starts(dimension_size, tile_size, stride):
    """Retourne les positions de depart des tuiles sur un axe."""

    if dimension_size <= 0 or tile_size <= 0 or not 0 < stride <= tile_size:
        raise ValueError("Dimension, taille de tuile ou pas invalide.")
    if dimension_size <= tile_size:
        return [0]

    starts = list(range(0, dimension_size - tile_size + 1, stride))
    last_start = dimension_size - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def compute_tile_windows(image_width, image_height, tile_size, stride):
    """Retourne les fenetres xyxy dont les arrets sont exclusifs."""

    windows = []
    for y_start in compute_tile_starts(image_height, tile_size, stride):
        for x_start in compute_tile_starts(image_width, tile_size, stride):
            windows.append((
                x_start,
                y_start,
                min(x_start + tile_size, image_width),
                min(y_start + tile_size, image_height),
            ))
    return windows


def build_tile_manifest(images_df):
    """Cree une ligne de manifeste pour chaque fenetre de chaque image."""

    tile_records = []
    for image_row in images_df.itertuples(index=False):
        windows = compute_tile_windows(
            image_row.source_width,
            image_row.source_height,
            TILE_SIZE,
            TILE_STRIDE,
        )

        for x_start, y_start, x_stop, y_stop in windows:
            tile_id = (
                f"{image_row.split}__{image_row.source_image_id}"
                f"__s{TILE_SIZE}"
                f"__x{x_start}_{x_stop}__y{y_start}_{y_stop}"
            )
            content_width = x_stop - x_start
            content_height = y_stop - y_start

            tile_records.append({
                "tile_id": tile_id,
                "split": image_row.split,
                "source_image_id": image_row.source_image_id,
                "source_image_path": image_row.source_image_path,
                "source_width": image_row.source_width,
                "source_height": image_row.source_height,
                "x_start": x_start,
                "y_start": y_start,
                "x_stop": x_stop,
                "y_stop": y_stop,
                "content_width": content_width,
                "content_height": content_height,
                "pad_right": TILE_SIZE - content_width,
                "pad_bottom": TILE_SIZE - content_height,
            })

    manifest_df = pd.DataFrame(tile_records)
    if manifest_df["tile_id"].duplicated().any():
        raise ValueError("Des identifiants de tuiles sont dupliques.")
    return manifest_df


def reduce_polygon_to_quadrilateral(vertices):
    """Conserve les quatre sommets qui couvrent le plus de surface."""

    if len(vertices) < 4:
        raise ValueError("Le polygone visible a moins de quatre sommets.")
    if len(vertices) == 4:
        return list(vertices)

    best_vertices = None
    largest_area = 0.0
    for vertex_indices in combinations(range(len(vertices)), 4):
        candidate = [vertices[index] for index in vertex_indices]
        candidate_area = compute_polygon_area(candidate)
        if candidate_area > largest_area:
            best_vertices = candidate
            largest_area = candidate_area

    return best_vertices


def build_object_tile_annotations(objects_df, tile_manifest_df):
    """Calcule les intersections et les deux representations de boites."""

    tile_groups = {
        key: group
        for key, group in tile_manifest_df.groupby(
            ["split", "source_image_id"], sort=False
        )
    }
    retained_records = []
    intersecting_tile_ids = set()
    positive_intersection_count = 0
    truncated_intersection_count = 0

    for image_key, image_objects_df in objects_df.groupby(
        ["split", "source_image_id"], sort=False
    ):
        image_tiles_df = tile_groups[image_key]

        for object_row in image_objects_df.itertuples(index=False):
            candidate_mask = (
                (image_tiles_df["x_start"] < object_row.max_x)
                & (image_tiles_df["x_stop"] > object_row.min_x)
                & (image_tiles_df["y_start"] < object_row.max_y)
                & (image_tiles_df["y_stop"] > object_row.min_y)
            )

            object_polygon = ShapelyPolygon(object_row.vertices)
            for tile_row in image_tiles_df.loc[candidate_mask].itertuples(
                index=False
            ):
                tile_polygon = shapely_box(
                    tile_row.x_start,
                    tile_row.y_start,
                    tile_row.x_stop,
                    tile_row.y_stop,
                )
                visible_geometry = object_polygon.intersection(tile_polygon)
                visible_area_px2 = float(visible_geometry.area)

                # Une ligne ou un point de contact ne forme pas une annotation.
                if visible_area_px2 <= 0:
                    continue

                positive_intersection_count += 1
                intersecting_tile_ids.add(tile_row.tile_id)
                visible_fraction = min(
                    1.0,
                    max(0.0, visible_area_px2 / object_row.source_area_px2),
                )
                is_truncated = visible_fraction < 1.0 - 1e-9
                truncated_intersection_count += int(is_truncated)

                # La table exhaustive sert a categoriser les tuiles, mais seuls
                # les fragments visibles a au moins 70 % deviennent des labels.
                if visible_fraction < MIN_VISIBLE_FRACTION:
                    continue
                if visible_geometry.geom_type != "Polygon":
                    raise ValueError(
                        "Une intersection positive n'est pas un Polygon."
                    )

                visible_vertices_source = [
                    (float(x), float(y))
                    for x, y in visible_geometry.exterior.coords[:-1]
                ]
                visible_vertices_local = [
                    (
                        float(x - tile_row.x_start),
                        float(y - tile_row.y_start),
                    )
                    for x, y in visible_vertices_source
                ]

                yolo_vertices_local = reduce_polygon_to_quadrilateral(
                    visible_vertices_local
                )
                # Limiter les minuscules erreurs flottantes aux bords.
                yolo_vertices_local = [
                    (
                        min(max(x, 0.0), float(TILE_SIZE)),
                        min(max(y, 0.0), float(TILE_SIZE)),
                    )
                    for x, y in yolo_vertices_local
                ]
                yolo_vertices_normalized = [
                    (x / TILE_SIZE, y / TILE_SIZE)
                    for x, y in yolo_vertices_local
                ]

                local_x = [x for x, _ in visible_vertices_local]
                local_y = [y for _, y in visible_vertices_local]
                hbb = (
                    min(local_x), min(local_y), max(local_x), max(local_y)
                )

                record = {
                    "ground_truth_id": (
                        f"{tile_row.tile_id}__{object_row.source_object_id}"
                    ),
                    "tile_id": tile_row.tile_id,
                    "split": object_row.split,
                    "source_image_id": object_row.source_image_id,
                    "source_object_id": object_row.source_object_id,
                    "object_index": object_row.object_index,
                    "class_name": object_row.class_name,
                    "difficult": object_row.difficult,
                    "yolo_class_id": object_row.yolo_class_id,
                    "faster_rcnn_class_id": object_row.faster_rcnn_class_id,
                    "visible_area_px2": visible_area_px2,
                    "visible_fraction": visible_fraction,
                    "is_truncated": is_truncated,
                    "source_area_px2": object_row.source_area_px2,
                    "source_relative_area": object_row.source_relative_area,
                    "source_orientation_deg": object_row.source_orientation_deg,
                    "source_long_side_px": object_row.source_long_side_px,
                    "source_short_side_px": object_row.source_short_side_px,
                    "source_aspect_ratio": object_row.source_aspect_ratio,
                    "source_center_x_normalized": (
                        object_row.source_center_x_normalized
                    ),
                    "source_center_y_normalized": (
                        object_row.source_center_y_normalized
                    ),
                    "image_source": object_row.image_source,
                    "gsd": object_row.gsd,
                    "gsd_missing": object_row.gsd_missing,
                    "hbb_x_min": hbb[0],
                    "hbb_y_min": hbb[1],
                    "hbb_x_max": hbb[2],
                    "hbb_y_max": hbb[3],
                }

                for vertex_number, (x_value, y_value) in enumerate(
                    yolo_vertices_normalized, start=1
                ):
                    record[f"obb_x{vertex_number}"] = x_value
                    record[f"obb_y{vertex_number}"] = y_value

                retained_records.append(record)

    retained_df = pd.DataFrame(retained_records)
    if retained_df["ground_truth_id"].duplicated().any():
        raise ValueError("Des identifiants ground_truth sont dupliques.")

    audit = {
        "positive_intersections": positive_intersection_count,
        "truncated_intersections": truncated_intersection_count,
        "retained_annotations": len(retained_df),
        "retained_source_objects": retained_df["source_object_id"].nunique(),
    }
    return retained_df, intersecting_tile_ids, audit


def select_generation_tiles(
    tile_manifest_df,
    retained_annotations_df,
    intersecting_tile_ids,
    pilot_mode,
):
    """Applique la politique positive, ambigue, fond et test."""

    labeled_df = tile_manifest_df.loc[
        tile_manifest_df["split"].isin(["train", "validation"])
    ].copy()
    retained_tile_ids = set(retained_annotations_df["tile_id"])

    labeled_df["tile_category"] = "background"
    labeled_df.loc[
        labeled_df["tile_id"].isin(intersecting_tile_ids),
        "tile_category",
    ] = "ambiguous"
    labeled_df.loc[
        labeled_df["tile_id"].isin(retained_tile_ids),
        "tile_category",
    ] = "positive"

    test_df = tile_manifest_df.loc[
        tile_manifest_df["split"] == "test"
    ].copy()
    test_df["tile_category"] = "unknown"

    categorized_df = pd.concat([labeled_df, test_df], ignore_index=True)

    if pilot_mode:
        indexed_df = categorized_df.set_index("tile_id")
        missing_ids = set(PILOT_TILE_IDS) - set(indexed_df.index)
        if missing_ids:
            raise ValueError(f"Tuiles pilotes absentes: {sorted(missing_ids)}")
        selected_df = indexed_df.loc[PILOT_TILE_IDS].reset_index()
    else:
        train_positive_df = labeled_df.loc[
            (labeled_df["split"] == "train")
            & (labeled_df["tile_category"] == "positive")
        ]
        train_background_df = labeled_df.loc[
            (labeled_df["split"] == "train")
            & (labeled_df["tile_category"] == "background")
        ].sample(
            frac=TRAIN_BACKGROUND_FRACTION,
            random_state=RANDOM_SEED,
        )
        validation_df = labeled_df.loc[
            (labeled_df["split"] == "validation")
            & labeled_df["tile_category"].isin(["positive", "background"])
        ]

        selected_df = pd.concat(
            [
                train_positive_df,
                train_background_df,
                validation_df,
                test_df,
            ],
            ignore_index=True,
        ).sort_values(
            ["split", "source_image_id", "y_start", "x_start"]
        ).reset_index(drop=True)

    if (selected_df["tile_category"] == "ambiguous").any():
        raise ValueError("Une tuile ambigue a ete selectionnee.")

    return selected_df


# ---------------------------------------------------------------------------
# Ecriture des metadonnees, images et labels
# ---------------------------------------------------------------------------

def make_serializable_manifest(selected_tiles_df, output_dir):
    """Remplace les chemins internes par des chemins relatifs transportables."""

    manifest_df = selected_tiles_df.copy()
    manifest_df["source_image_path"] = manifest_df["source_image_path"].map(
        lambda path: Path(path).relative_to(PROJECT_DIR).as_posix()
    )
    manifest_df["tile_image_path"] = manifest_df.apply(
        lambda row: (
            Path("images") / row["split"] / f"{row['tile_id']}.png"
        ).as_posix(),
        axis=1,
    )
    manifest_df["output_root"] = output_dir.name
    return manifest_df


def format_yolo_label(annotation_row):
    """Construit les neuf valeurs d'une ligne YOLO-OBB."""

    coordinate_values = []
    for vertex_number in range(1, 5):
        coordinate_values.extend([
            annotation_row[f"obb_x{vertex_number}"],
            annotation_row[f"obb_y{vertex_number}"],
        ])
    coordinate_text = " ".join(
        f"{float(value):.6f}" for value in coordinate_values
    )
    return f"{int(annotation_row['yolo_class_id'])} {coordinate_text}"


def existing_image_is_valid(image_path):
    """Permet de reprendre le script sans reecrire un PNG deja complet."""

    if not image_path.is_file():
        return False
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            return image.size == (TILE_SIZE, TILE_SIZE) and image.mode == "RGB"
    except (OSError, SyntaxError):
        return False


def write_dataset_files(
    selected_tiles_df,
    selected_annotations_df,
    output_dir,
):
    """Ecrit les PNG et les TXT, en gardant une image source en memoire."""

    annotations_by_tile = {
        tile_id: group
        for tile_id, group in selected_annotations_df.groupby(
            "tile_id", sort=False
        )
    }

    current_source_path = None
    current_source_image = None
    generated_count = 0
    skipped_count = 0

    try:
        for tile_number, tile_row in enumerate(
            selected_tiles_df.itertuples(index=False), start=1
        ):
            image_dir = output_dir / "images" / tile_row.split
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"{tile_row.tile_id}.png"

            if existing_image_is_valid(image_path):
                skipped_count += 1
            else:
                source_path = Path(tile_row.source_image_path)
                if source_path != current_source_path:
                    if current_source_image is not None:
                        current_source_image.close()
                    with Image.open(source_path) as source_image:
                        current_source_image = source_image.convert("RGB")
                    current_source_path = source_path

                cropped_tile = current_source_image.crop((
                    tile_row.x_start,
                    tile_row.y_start,
                    tile_row.x_stop,
                    tile_row.y_stop,
                ))
                padded_tile = Image.new(
                    "RGB", (TILE_SIZE, TILE_SIZE), PADDING_COLOR
                )
                padded_tile.paste(cropped_tile, (0, 0))

                # compress_level=1 reste sans perte, mais accelere l'ecriture.
                padded_tile.save(image_path, format="PNG", compress_level=1)
                padded_tile.close()
                cropped_tile.close()
                generated_count += 1

            if tile_row.split in ("train", "validation"):
                label_dir = output_dir / "labels" / tile_row.split
                label_dir.mkdir(parents=True, exist_ok=True)
                label_path = label_dir / f"{tile_row.tile_id}.txt"
                tile_annotations_df = annotations_by_tile.get(tile_row.tile_id)

                if tile_annotations_df is None:
                    label_lines = []
                else:
                    label_lines = [
                        format_yolo_label(annotation_row)
                        for _, annotation_row in tile_annotations_df.iterrows()
                    ]
                label_path.write_text(
                    "\n".join(label_lines), encoding="utf-8"
                )

            if tile_number % 250 == 0 or tile_number == len(selected_tiles_df):
                print(
                    f"Fichiers: {tile_number}/{len(selected_tiles_df)} "
                    f"(nouveaux={generated_count}, repris={skipped_count})",
                    flush=True,
                )
    finally:
        if current_source_image is not None:
            current_source_image.close()

    return generated_count, skipped_count


def write_metadata(
    output_dir,
    selected_tiles_df,
    selected_annotations_df,
    rejected_df,
    configuration,
    class_names,
):
    """Sauvegarde les tables necessaires aux deux modeles et aux audits."""

    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Une configuration differente ne doit jamais reutiliser silencieusement
    # les fichiers d'une ancienne generation dans le meme dossier.
    configuration_path = metadata_dir / "configuration.json"
    if configuration_path.is_file():
        previous_configuration = json.loads(
            configuration_path.read_text(encoding="utf-8")
        )
        if previous_configuration != configuration:
            raise ValueError(
                "Le dossier de sortie contient une configuration differente."
            )

    manifest_df = make_serializable_manifest(selected_tiles_df, output_dir)
    annotation_counts = selected_annotations_df["tile_id"].value_counts()
    manifest_df["annotation_count"] = (
        manifest_df["tile_id"].map(annotation_counts).fillna(0).astype(int)
    )
    manifest_df.to_csv(metadata_dir / "tile_manifest.csv", index=False)
    selected_annotations_df.to_csv(
        metadata_dir / "object_annotations.csv", index=False
    )
    rejected_df.to_csv(metadata_dir / "rejected_annotations.csv", index=False)

    class_mapping_df = pd.DataFrame({
        "class_name": class_names,
        "yolo_class_id": range(len(class_names)),
        "faster_rcnn_class_id": range(1, len(class_names) + 1),
    })
    class_mapping_df.to_csv(metadata_dir / "class_mapping.csv", index=False)

    configuration_path.write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )

    # Un chemin relatif au projet reste valide apres un transfert sur un autre
    # PC. Un dossier de sortie externe conserve toutefois son chemin absolu.
    try:
        portable_output_path = output_dir.resolve().relative_to(
            PROJECT_DIR.resolve()
        ).as_posix()
    except ValueError:
        portable_output_path = str(output_dir.resolve())

    yolo_configuration = {
        "path": portable_output_path,
        "train": "images/train",
        "val": "images/validation",
        "test": "images/test",
        "names": {
            class_id: class_name
            for class_id, class_name in enumerate(class_names)
        },
    }
    (output_dir / "dota_obb.yaml").write_text(
        yaml.safe_dump(yolo_configuration, sort_keys=False),
        encoding="utf-8",
    )


def validate_expected_counts(
    dataset_version,
    pilot_mode,
    images_df,
    objects_df,
    tile_manifest_df,
    retained_annotations_df,
    selected_tiles_df,
    rejected_df,
    audit,
):
    """Bloque une generation complete qui diverge du pilote du notebook."""

    if pilot_mode and dataset_version == "1.0":
        expected_labels = {
            PILOT_TILE_IDS[0]: 18,
            PILOT_TILE_IDS[1]: 13,
            PILOT_TILE_IDS[2]: 85,
            PILOT_TILE_IDS[3]: 0,
            PILOT_TILE_IDS[4]: 0,
        }
        observed_labels = retained_annotations_df["tile_id"].value_counts()
        for tile_id, expected_count in expected_labels.items():
            observed_count = int(observed_labels.get(tile_id, 0))
            if observed_count != expected_count:
                raise ValueError(
                    f"{tile_id}: {observed_count} labels au lieu de "
                    f"{expected_count}."
                )
        if len(selected_tiles_df) != 5:
            raise ValueError("Le pilote doit contenir exactement cinq tuiles.")
        return

    if pilot_mode:
        if len(selected_tiles_df) != 5:
            raise ValueError("Le pilote doit contenir exactement cinq tuiles.")
        return

    if dataset_version == "1.0":
        expected_values = {
            "images": (len(images_df), 2805),
            "objets_valides": (len(objects_df), 127656),
            "annotations_rejetees": (len(rejected_df), 1),
            "fenetres": (len(tile_manifest_df), 31849),
            "intersections_positives": (
                audit["positive_intersections"], 269956
            ),
            "annotations_retenues": (len(retained_annotations_df), 245822),
            "tuiles_generees": (len(selected_tiles_df), 26097),
        }
    else:
        # Ces effectifs viennent de l'audit complet des annotations v1.5.
        # Les effectifs derives seront affiches par --validate-only.
        expected_values = {
            "images": (len(images_df), 2805),
            "objets_valides": (len(objects_df), 279961),
            "annotations_rejetees": (len(rejected_df), 4),
            "fenetres": (len(tile_manifest_df), 31849),
            "annotations_retenues": (len(retained_annotations_df), 508378),
            "tuiles_generees": (len(selected_tiles_df), 26923),
        }
    differences = {
        name: {"observe": observed, "attendu": expected}
        for name, (observed, expected) in expected_values.items()
        if observed != expected
    }
    if differences:
        raise ValueError(f"Effectifs inattendus: {differences}")


def main():
    """Execute le pipeline complet dans un ordre reproductible."""

    arguments = parse_arguments()
    pilot_mode = arguments.pilot
    dataset_version = arguments.dataset_version

    if dataset_version == "1.5":
        class_names = DOTA_V15_CLASS_NAMES
    else:
        class_names = DOTA_V10_CLASS_NAMES
    class_to_yolo_id = {
        class_name: class_id
        for class_id, class_name in enumerate(class_names)
    }

    if arguments.output_dir is not None:
        output_dir = arguments.output_dir.resolve()
    elif pilot_mode:
        pilot_directory_name = (
            "script_tiling_pilot_v15"
            if dataset_version == "1.5"
            else "script_tiling_pilot"
        )
        output_dir = PROJECT_DIR / "tmp" / pilot_directory_name
    elif dataset_version == "1.5":
        output_dir = PROJECT_DIR / "outputs" / "dota_v15_tiles_1024"
    else:
        output_dir = PROJECT_DIR / "outputs" / "dota_tiles_1024"

    print("Mode :", "pilote" if pilot_mode else "generation complete")
    print("Annotations : DOTA-v" + dataset_version)
    print("Sortie :", output_dir.relative_to(PROJECT_DIR))

    print("1/6 - Lecture des images et annotations", flush=True)
    images_df = discover_images(pilot_mode, dataset_version)
    objects_df, rejected_df = build_object_table(
        images_df,
        class_to_yolo_id,
    )

    print("2/6 - Construction des fenetres", flush=True)
    tile_manifest_df = build_tile_manifest(images_df)

    print("3/6 - Intersection des objets et des tuiles", flush=True)
    retained_annotations_df, intersecting_tile_ids, audit = (
        build_object_tile_annotations(objects_df, tile_manifest_df)
    )

    print("4/6 - Selection des tuiles", flush=True)
    selected_tiles_df = select_generation_tiles(
        tile_manifest_df,
        retained_annotations_df,
        intersecting_tile_ids,
        pilot_mode,
    )
    selected_tile_ids = set(selected_tiles_df["tile_id"])
    selected_annotations_df = retained_annotations_df.loc[
        retained_annotations_df["tile_id"].isin(selected_tile_ids)
    ].copy()

    validate_expected_counts(
        dataset_version,
        pilot_mode,
        images_df,
        objects_df,
        tile_manifest_df,
        selected_annotations_df if pilot_mode else retained_annotations_df,
        selected_tiles_df,
        rejected_df,
        audit,
    )

    # Ce mode controle les effectifs complets sans lancer l'ecriture longue.
    if arguments.validate_only:
        print("Validation complete terminee sans ecriture de fichiers.")
        print("Images sources :", len(images_df))
        print("Objets valides :", len(objects_df))
        print("Annotations geometriques rejetees :", len(rejected_df))
        print("Fenetres totales :", len(tile_manifest_df))
        print("Intersections positives :", audit["positive_intersections"])
        print("Annotations retenues :", len(retained_annotations_df))
        print("Tuiles selectionnees :", len(selected_tiles_df))
        return

    configuration = {
        "dataset_version": dataset_version,
        "pilot_mode": pilot_mode,
        "tile_size": TILE_SIZE,
        "tile_overlap": TILE_OVERLAP,
        "tile_stride": TILE_STRIDE,
        "minimum_visible_fraction": MIN_VISIBLE_FRACTION,
        "train_background_fraction": TRAIN_BACKGROUND_FRACTION,
        "random_seed": RANDOM_SEED,
        "padding_color": list(PADDING_COLOR),
        "excluded_train_image_ids": sorted(EXCLUDED_TRAIN_IMAGE_IDS),
    }

    print("5/6 - Ecriture des metadonnees", flush=True)
    write_metadata(
        output_dir,
        selected_tiles_df,
        selected_annotations_df,
        rejected_df,
        configuration,
        class_names,
    )

    print("6/6 - Ecriture des images et labels", flush=True)
    generated_count, skipped_count = write_dataset_files(
        selected_tiles_df,
        selected_annotations_df,
        output_dir,
    )

    print("Generation terminee.")
    print("Tuiles selectionnees :", len(selected_tiles_df))
    print("Annotations selectionnees :", len(selected_annotations_df))
    print("PNG nouvellement ecrits :", generated_count)
    print("PNG valides deja presents :", skipped_count)
    print("Intersections positives :", audit["positive_intersections"])
    print("Annotations geometriques rejetees :", len(rejected_df))


if __name__ == "__main__":
    main()
