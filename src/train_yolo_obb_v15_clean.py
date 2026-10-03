"""Entraine YOLO11s-OBB sur DOTA-v1.5 sans checkpoint DOTA-v1.0.

Le modele part des poids generiques COCO de YOLO11s. L'architecture OBB est
construite avant le transfert afin de ne pas confondre un modele HBB de
detection classique avec un modele de boites orientees.
"""

from __future__ import annotations

import sys
from pathlib import Path

import train_yolo_obb_final as training_engine
from train_yolo_obb_v15 import file_sha256, load_hardware_profile


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_YAML = (
    PROJECT_DIR / "outputs" / "dota_v15_tiles_1024" / "dota_obb.yaml"
)
COCO_WEIGHTS = PROJECT_DIR / "yolo11s.pt"
OBB_ARCHITECTURE = "yolo11s-obb.yaml"
COCO_WEIGHTS_SHA256 = (
    "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"
)

CLASS_COUNT = 16
COCO_CLASS_COUNT = 80
RANDOM_SEED = 42
FINAL_RUN_NAME = "yolo11s_obb_v15_clean_coco_55h"
DEFAULT_HOURS = 55.0

# La RTX 5080 termine actuellement environ 7 epoques par heure. Un plafond de
# 400 epoques correspond donc au budget de 55 h, tout en donnant au scheduler
# un horizon realiste (contrairement a l'ancien plafond artificiel de 5000).
EPOCH_CEILING = 400


def configure_training_engine():
    """Specialise le moteur robuste sans modifier les anciens runs."""

    training_engine.DATASET_YAML = DATASET_YAML
    training_engine.PILOT_BEST = COCO_WEIGHTS
    training_engine.CLASS_COUNT = CLASS_COUNT
    training_engine.FINAL_RUN_NAME = FINAL_RUN_NAME
    training_engine.FINAL_EPOCHS = EPOCH_CEILING
    training_engine.DEFAULT_HOURS = DEFAULT_HOURS
    training_engine.CURRENT_RUN_FILE = (
        training_engine.RUNS_DIR / "current_training_v15_clean.json"
    )

    original_check_environment = training_engine.check_environment
    original_build_config = training_engine.build_config
    original_attach_callbacks = training_engine.attach_callbacks
    original_run_training = training_engine.run_training

    def checked_hardware_profile(torch):
        """Charge le profil mesure sur le GPU present, jamais celui d'un autre PC."""

        profile_path, profile = load_hardware_profile(torch)
        fixed = profile.get("fixed_training_settings") or {}
        if fixed.get("optimizer") != "AdamW":
            raise RuntimeError(
                "Le profil materiel n'a pas ete mesure avec AdamW."
            )
        return profile_path, profile

    def check_environment(torch, yaml, weights):
        original_check_environment(torch, yaml, weights)
        if (
            Path(weights).resolve() == COCO_WEIGHTS.resolve()
            and file_sha256(COCO_WEIGHTS).lower() != COCO_WEIGHTS_SHA256
        ):
            raise RuntimeError("L'empreinte de yolo11s.pt est inattendue.")
        profile_path, profile = checked_hardware_profile(torch)
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
        """Ajoute la provenance et les reglages du run propre a sa configuration."""

        import torch

        profile_path, profile = checked_hardware_profile(torch)
        config = original_build_config(run_name, hours, smoke_minutes)
        config["dataset_version"] = "1.5"
        config["class_count"] = CLASS_COUNT
        config["initialization"] = {
            "method": "coco_hbb_compatible_transfer_to_obb",
            "source_dataset": "COCO",
            "source_task": "detect",
            "source_classes": COCO_CLASS_COUNT,
            "source_checkpoint": training_engine.relative(COCO_WEIGHTS),
            "source_checkpoint_sha256": file_sha256(COCO_WEIGHTS),
            "target_architecture": OBB_ARCHITECTURE,
            "target_classes": CLASS_COUNT,
            "dota_v10_checkpoint_loaded": False,
            "optimizer_restarted": True,
        }
        config["hardware_profile"] = {
            "path": training_engine.relative(profile_path),
            "sha256": file_sha256(profile_path),
            "gpu_name": profile["gpu"]["name"],
            "batch": profile["recommended"]["batch"],
            "workers": profile["recommended"]["workers"],
        }

        # Ces valeurs sont sauvegardees dans chaque checkpoint. Une reprise
        # recharge donc exactement le meme optimiseur et le meme scheduler.
        config["trainer_args"].update({
            "batch": profile["recommended"]["batch"],
            "workers": profile["recommended"]["workers"],
            "epochs": EPOCH_CEILING,
            "optimizer": "AdamW",
            "lr0": 0.0005,
            "warmup_epochs": 1.0,
            "cos_lr": True,
            "lrf": 0.01,
            "patience": 150,
            "save_period": 25,
        })
        return config

    def make_model(YOLO, weights, task="obb", verbose=False):
        """Construit la bonne architecture au depart, puis reprend les .pt OBB."""

        weights_path = Path(str(weights))
        if weights_path.resolve() == COCO_WEIGHTS.resolve():
            # yolo11s.pt est un modele HBB. On cree d'abord une architecture
            # OBB, puis .load() copie seulement les couches compatibles COCO.
            # La tete OBB non presente dans COCO est initialisee aleatoirement;
            # fixer la graine ici rend ce depart reproductible entre deux PC.
            import torch

            torch.manual_seed(RANDOM_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(RANDOM_SEED)
            model = YOLO(OBB_ARCHITECTURE, task="obb", verbose=verbose)
            model.load(str(COCO_WEIGHTS))
            return model

        # Lors d'une reprise, le checkpoint contient deja la tete OBB a 16
        # classes, l'optimiseur, le scaler AMP et l'EMA. Il doit etre charge tel quel.
        return YOLO(str(weights), task=task, verbose=verbose)

    def validate_model(model):
        # Avant le premier train, la nouvelle architecture expose encore les
        # 80 noms COCO. Ultralytics remplace ensuite sa tete par les 16 classes
        # du YAML. Un checkpoint de reprise en possede deja 16.
        class_number = len(model.names)
        if model.task != "obb" or class_number not in {
            COCO_CLASS_COUNT,
            CLASS_COUNT,
        }:
            raise RuntimeError(
                "Le modele doit etre une architecture OBB compatible COCO/DOTA-v1.5."
            )

    def validate_target_model(trainer):
        names = getattr(trainer.model, "names", {})
        if len(names) != CLASS_COUNT:
            raise RuntimeError(
                "La tete OBB n'a pas ete adaptee aux 16 classes DOTA-v1.5."
            )
        print(
            "Tete OBB adaptee a DOTA-v1.5 : 16 classes.",
            flush=True,
        )

    def attach_callbacks(model, tracker):
        original_attach_callbacks(model, tracker)
        model.add_callback("on_train_start", validate_target_model)

    def run_check(torch, yaml, YOLO):
        """Verifie les deux etapes de l'initialisation sans entrainer."""

        check_environment(torch, yaml, COCO_WEIGHTS)

        source_model = YOLO(str(COCO_WEIGHTS), verbose=False)
        if source_model.task != "detect" or len(source_model.names) != 80:
            raise RuntimeError("yolo11s.pt n'est pas le checkpoint COCO attendu.")

        target_model = make_model(
            YOLO,
            COCO_WEIGHTS,
            task="obb",
            verbose=False,
        )
        validate_model(target_model)
        if len(target_model.names) != COCO_CLASS_COUNT:
            raise RuntimeError("Le transfert COCO vers l'architecture OBB a echoue.")

        print("Poids source : YOLO11s COCO, detection HBB, 80 classes.", flush=True)
        print(
            "Architecture cible : YOLO11s-OBB; la tete sera adaptee "
            "aux 16 classes au demarrage.",
            flush=True,
        )
        print("Aucun checkpoint DOTA-v1.0 n'est charge.", flush=True)
        print("Aucun entrainement n'a ete lance.", flush=True)

    def run_training(args, torch, yaml, YOLO, session_log):
        # Le moteur robuste appelle cette fabrique aussi bien au premier depart
        # qu'apres une coupure. Elle distingue les poids COCO des checkpoints OBB.
        def model_factory(weights, task="obb", verbose=False):
            return make_model(YOLO, weights, task=task, verbose=verbose)

        return original_run_training(
            args,
            torch,
            yaml,
            model_factory,
            session_log,
        )

    training_engine.check_environment = check_environment
    training_engine.build_config = build_config
    training_engine.validate_model = validate_model
    training_engine.attach_callbacks = attach_callbacks
    training_engine.run_check = run_check
    training_engine.run_training = run_training


def main():
    configure_training_engine()
    return training_engine.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERREUR : {type(error).__name__}: {error}", file=sys.stderr)
        raise
