"""Entraine YOLO11s-OBB sur DOTA-v1.5 depuis le meilleur modele v1.0."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import train_yolo_obb_final as training_engine


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_YAML = (
    PROJECT_DIR / "outputs" / "dota_v15_tiles_1024" / "dota_obb.yaml"
)
TRANSFER_WEIGHTS = (
    PROJECT_DIR
    / "outputs"
    / "transfer_weights"
    / "yolo11s_obb_dota_v10_best_epoch273.pt"
)
PROFILE_ROOT = PROJECT_DIR / "outputs" / "yolo_hardware_profiles"

CLASS_COUNT = 16
SOURCE_CLASS_COUNT = 15
FINAL_RUN_NAME = "yolo11s_obb_v15_final"
DEFAULT_HOURS = 72.0
EPOCH_CEILING = 5000


def safe_name(value):
    """Produit le meme nom de dossier que le script d'auto-reglage."""

    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hardware_profile(torch):
    """Charge seulement le profil mesure sur le GPU actuellement utilise."""

    gpu_name = torch.cuda.get_device_name(0)
    profile_path = (
        PROFILE_ROOT / safe_name(gpu_name) / "hardware_profile.json"
    )
    if not profile_path.is_file():
        raise FileNotFoundError(
            "Profil materiel introuvable pour ce GPU. Execute d'abord "
            "benchmark_yolo_hardware.py run."
        )

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    recommended = profile.get("recommended") or {}
    fixed = profile.get("fixed_training_settings") or {}
    stored_gpu = profile.get("gpu") or {}
    profile_dataset = training_engine.absolute(profile.get("dataset", ""))

    if profile.get("schema_version") != 2:
        raise RuntimeError("Version de profil materiel non prise en charge.")
    if stored_gpu.get("name") != gpu_name:
        raise RuntimeError("Le profil materiel appartient a un autre GPU.")
    if profile.get("model") != "yolo11s-obb":
        raise RuntimeError("Le profil ne correspond pas a YOLO11s-OBB.")
    if profile_dataset.resolve() != DATASET_YAML.resolve():
        raise RuntimeError("Le profil ne correspond pas au dataset DOTA-v1.5.")
    if (
        fixed.get("imgsz") != 1024
        or fixed.get("amp") is not True
        or fixed.get("nbs") != 64
        or fixed.get("multi_scale") is not False
    ):
        raise RuntimeError("Les reglages fixes du profil sont incompatibles.")

    batch = recommended.get("batch")
    workers = recommended.get("workers")
    if not isinstance(batch, int) or batch <= 0:
        raise RuntimeError("Batch recommande invalide dans le profil.")
    if not isinstance(workers, int) or workers < 0:
        raise RuntimeError("Nombre de workers invalide dans le profil.")
    if recommended.get("dense_confirmation_passed") is not True:
        raise RuntimeError(
            "Le profil n'a pas reussi la confirmation sur les tuiles denses."
        )

    benchmark_settings = profile.get("benchmark_settings") or {}
    if benchmark_settings.get("maximum_vram_fraction", 1.0) > 0.82:
        raise RuntimeError("La marge VRAM du profil est insuffisante.")

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    observed_total_gib = total_bytes / 1024**3
    stored_total_gib = float(stored_gpu.get("total_gib", 0))
    if abs(observed_total_gib - stored_total_gib) > 0.5:
        raise RuntimeError("La quantite de VRAM differe de celle du profil.")

    return profile_path, profile


def configure_training_engine():
    """Specialise le moteur robuste existant sans toucher au run v1.0."""

    training_engine.DATASET_YAML = DATASET_YAML
    training_engine.PILOT_BEST = TRANSFER_WEIGHTS
    training_engine.CLASS_COUNT = CLASS_COUNT
    training_engine.FINAL_RUN_NAME = FINAL_RUN_NAME
    training_engine.FINAL_EPOCHS = EPOCH_CEILING
    training_engine.DEFAULT_HOURS = DEFAULT_HOURS
    training_engine.CURRENT_RUN_FILE = (
        training_engine.RUNS_DIR / "current_training_v15.json"
    )

    original_check_environment = training_engine.check_environment
    original_build_config = training_engine.build_config
    original_attach_callbacks = training_engine.attach_callbacks

    def check_environment(torch, yaml, weights):
        original_check_environment(torch, yaml, weights)
        profile_path, profile = load_hardware_profile(torch)
        print(
            "Profil materiel :",
            training_engine.relative(profile_path),
            flush=True,
        )
        print(
            "Configuration retenue :",
            f"batch={profile['recommended']['batch']},",
            f"workers={profile['recommended']['workers']}",
            flush=True,
        )

    def build_config(run_name, hours, smoke_minutes=None):
        import torch

        profile_path, profile = load_hardware_profile(torch)
        config = original_build_config(run_name, hours, smoke_minutes)
        config["dataset_version"] = "1.5"
        config["class_count"] = CLASS_COUNT
        config["initialization"] = {
            "method": "transfer_learning",
            "source_dataset": "DOTA-v1.0",
            "source_classes": SOURCE_CLASS_COUNT,
            "source_checkpoint": training_engine.relative(TRANSFER_WEIGHTS),
            "source_checkpoint_sha256": file_sha256(TRANSFER_WEIGHTS),
            "source_best_epoch": 273,
            "source_best_map50_95": 0.63738,
            "optimizer_restarted": True,
        }
        config["hardware_profile"] = {
            "path": training_engine.relative(profile_path),
            "sha256": file_sha256(profile_path),
            "gpu_name": profile["gpu"]["name"],
            "batch": profile["recommended"]["batch"],
            "workers": profile["recommended"]["workers"],
        }
        config["trainer_args"].update({
            "batch": profile["recommended"]["batch"],
            "workers": profile["recommended"]["workers"],
        })
        return config

    def validate_model(model):
        # Avant le premier train, le checkpoint source possede encore 15
        # classes. Les checkpoints de reprise v1.5 en possedent deja 16.
        class_number = len(model.names)
        if model.task != "obb" or class_number not in {
            SOURCE_CLASS_COUNT,
            CLASS_COUNT,
        }:
            raise RuntimeError(
                "Le checkpoint doit etre un modele OBB compatible DOTA."
            )

    def validate_target_model(trainer):
        names = getattr(trainer.model, "names", {})
        if len(names) != CLASS_COUNT:
            raise RuntimeError(
                "La tete du modele n'a pas ete adaptee aux 16 classes."
            )
        print(
            "Tete OBB adaptee a DOTA-v1.5 : 16 classes.",
            flush=True,
        )

    def attach_callbacks(model, tracker):
        original_attach_callbacks(model, tracker)
        model.add_callback("on_train_start", validate_target_model)

    def run_check(torch, yaml, YOLO):
        check_environment(torch, yaml, TRANSFER_WEIGHTS)
        model = YOLO(str(TRANSFER_WEIGHTS), task="obb", verbose=False)
        validate_model(model)
        if len(model.names) != SOURCE_CLASS_COUNT:
            raise RuntimeError(
                "Le checkpoint de transfert doit provenir des 15 classes v1.0."
            )
        print(
            "Checkpoint v1.0 valide : meilleur modele de l'epoque 273.",
            flush=True,
        )
        print(
            "La tete sera adaptee aux 16 classes au lancement du train v1.5.",
            flush=True,
        )
        print("Aucun entrainement n'a ete lance.", flush=True)

    training_engine.check_environment = check_environment
    training_engine.build_config = build_config
    training_engine.validate_model = validate_model
    training_engine.attach_callbacks = attach_callbacks
    training_engine.run_check = run_check


def main():
    configure_training_engine()
    return training_engine.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERREUR : {type(error).__name__}: {error}", file=sys.stderr)
        raise
