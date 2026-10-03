"""Demarre, suit et reprend l'entrainement final YOLO11s-OBB."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import msvcrt
import os
import shutil
import sys
import threading
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_YAML = PROJECT_DIR / "outputs" / "dota_tiles_1024" / "dota_obb.yaml"
PILOT_BEST = (
    PROJECT_DIR
    / "outputs"
    / "yolo_runs"
    / "yolo11s_obb_pilot_1h"
    / "weights"
    / "best.pt"
)
RUNS_DIR = PROJECT_DIR / "outputs" / "yolo_runs"
CURRENT_RUN_FILE = RUNS_DIR / "current_training.json"
GPU_LOCK_FILE = RUNS_DIR / "training_gpu.lock"

ULTRALYTICS_VERSION = "8.4.118"
CLASS_COUNT = 15
FINAL_RUN_NAME = "yolo11s_obb_final_7d"
FINAL_EPOCHS = 810
DEFAULT_HOURS = 168.0
HEARTBEAT_SECONDS = 5.0
MIN_FREE_DISK_GIB = 10.0


class SmokeTestFinished(Exception):
    """Arrete volontairement le test court sans lancer la validation finale."""


def utc_now():
    """Retourne l'heure UTC dans un format lisible par JSON."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def relative(path):
    """Evite d'enregistrer un chemin personnel lorsqu'un chemin relatif suffit."""

    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)


def absolute(path_text):
    """Reconstruit un chemin absolu depuis la configuration du run."""

    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_DIR / path


def stored_path_matches(saved_path_text, expected_path):
    """Accepte un meme chemin, ou le meme nom apres transfert sur un autre PC."""

    if not saved_path_text:
        return False

    saved_path = Path(saved_path_text)
    expected_path = Path(expected_path).resolve()
    if saved_path.resolve() == expected_path:
        return True

    # Un ancien chemin absolu n'existe plus apres le transfert. Le nom final
    # doit tout de meme correspondre afin de ne pas accepter un autre dataset.
    return (
        not saved_path.exists()
        and saved_path.name.casefold() == expected_path.name.casefold()
    )


def relocate_config_paths(config):
    """Remplace seulement les chemins de machine par les chemins du PC actuel."""

    trainer_args = config.get("trainer_args") or {}
    expected_values = {
        "data": str(DATASET_YAML.resolve()),
        "project": str(RUNS_DIR.resolve()),
        "name": config["run_name"],
    }
    changed = any(
        trainer_args.get(key) != value
        for key, value in expected_values.items()
    )
    trainer_args.update(expected_values)
    config["trainer_args"] = trainer_args
    return changed


def read_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path, value):
    """Ecrit dans un fichier temporaire avant de remplacer le JSON precedent."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Chaque thread utilise son propre fichier temporaire. Le heartbeat et les
    # callbacks d'entrainement ne peuvent donc pas ecrire dans le meme fichier.
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, ensure_ascii=True)
            file.flush()
            os.fsync(file.fileno())

        # Sous Windows, un lecteur peut verrouiller le JSON pendant un tres
        # court instant. Quelques tentatives evitent de perdre le heartbeat.
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (2 ** attempt))
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


class SessionLogger:
    """Copie les sorties du processus dans un journal propre a la session."""

    def __init__(self, run_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(run_dir) / "logs" / f"session_{timestamp}_{os.getpid()}.log"
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
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
            self.original_stdout.write(text)
            self.file.write(text)
        return len(text)

    def flush(self):
        with self.lock:
            self.original_stdout.flush()
            self.file.flush()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, file=self
            )
        self.flush()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.file.close()


class GpuLock:
    """Empeche deux scripts du projet d'entrainer en meme temps."""

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
            raise RuntimeError("Un autre entrainement utilise deja le GPU.") from error
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.file.seek(0)
        msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        self.file.close()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Entrainement final robuste de YOLO11s-OBB."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("start", help="Commence un nouveau run.")
    start.add_argument("--run-name", default=FINAL_RUN_NAME)
    start.add_argument("--hours", type=float, default=DEFAULT_HOURS)
    start.add_argument(
        "--smoke-minutes",
        type=float,
        help="Teste le moniteur puis quitte sans resultat scientifique.",
    )

    resume = commands.add_parser("resume", help="Reprend le dernier checkpoint.")
    resume.add_argument("--run-name", default=FINAL_RUN_NAME)

    commands.add_parser("check", help="Verifie les prerequis sans entrainer.")
    return parser.parse_args()


def validate_run_name(name):
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Le nom du run doit etre un simple nom de dossier.")


def check_environment(torch, yaml, weights):
    """Verifie les prerequis qui pourraient faire echouer un long run."""

    expected_python = PROJECT_DIR / ".venv_gpu" / "Scripts" / "python.exe"
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise RuntimeError("Utilise .\\.venv_gpu\\Scripts\\python.exe.")
    if metadata.version("ultralytics") != ULTRALYTICS_VERSION:
        raise RuntimeError(f"Ultralytics {ULTRALYTICS_VERSION} est requis.")
    if not DATASET_YAML.is_file() or not Path(weights).is_file():
        raise FileNotFoundError("Le YAML du dataset ou les poids sont introuvables.")

    with DATASET_YAML.open(encoding="utf-8") as file:
        dataset_config = yaml.safe_load(file)

    # Le YAML original contient le chemin absolu de cette machine. Apres une
    # copie sur un autre PC, sa racine devient automatiquement le dossier local.
    local_dataset_root = DATASET_YAML.parent.resolve()
    if Path(str(dataset_config.get("path", ""))).resolve() != local_dataset_root:
        dataset_config["path"] = str(local_dataset_root)
        temporary_yaml = DATASET_YAML.with_suffix(".yaml.tmp")
        with temporary_yaml.open("w", encoding="utf-8") as file:
            yaml.safe_dump(dataset_config, file, sort_keys=False)
        os.replace(temporary_yaml, DATASET_YAML)

    names = dataset_config.get("names", {})
    if len(names) != CLASS_COUNT:
        raise RuntimeError(f"Le dataset doit contenir {CLASS_COUNT} classes.")
    for split_name in ("train", "val"):
        split_path = local_dataset_root / dataset_config[split_name]
        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Le dossier du split {split_name} est introuvable : {split_path}"
            )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA n'est pas disponible.")

    gpu_name = torch.cuda.get_device_name(0)
    expected_gpu_name = os.environ.get("DOTA_EXPECTED_GPU", "").strip()
    if expected_gpu_name and expected_gpu_name.casefold() not in gpu_name.casefold():
        raise RuntimeError(
            f"GPU inattendu : {gpu_name}. GPU requis : {expected_gpu_name}."
        )

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gib = free_bytes / 1024**3
    total_gib = total_bytes / 1024**3
    # Le contexte CUDA de PyTorch utilise deja une petite partie de la VRAM.
    required_gib = 0.85 * total_gib
    if free_gib < required_gib:
        raise RuntimeError(
            f"VRAM libre insuffisante ({free_gib:.2f} Gio). "
            "Arrete le noyau Jupyter puis recommence."
        )

    free_disk_gib = shutil.disk_usage(PROJECT_DIR).free / 1024**3
    if free_disk_gib < MIN_FREE_DISK_GIB:
        raise RuntimeError(f"Espace disque insuffisant ({free_disk_gib:.2f} Gio).")

    print("Environnement valide.", flush=True)
    print("GPU :", gpu_name, flush=True)
    print(f"VRAM libre : {free_gib:.2f}/{total_gib:.2f} Gio", flush=True)
    print(f"Espace disque libre : {free_disk_gib:.2f} Gio", flush=True)


def load_checkpoint_info(path, torch, expected_config=None):
    """Retourne l'epoque seulement si le checkpoint permet une vraie reprise."""

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        train_args = checkpoint.get("train_args") or {}
        model = checkpoint.get("ema")
        if model is None:
            model = checkpoint.get("model")
        names = getattr(model, "names", {})
        valid = (
            int(checkpoint.get("epoch", -1)) >= 0
            and checkpoint.get("optimizer") is not None
            and checkpoint.get("scaler") is not None
            and checkpoint.get("ema") is not None
            and checkpoint.get("best_fitness") is not None
            and train_args.get("task") == "obb"
            and len(names) == CLASS_COUNT
        )
        if expected_config is not None:
            expected_args = expected_config["trainer_args"]
            expected_run_dir = (
                RUNS_DIR / expected_config["run_name"]
            ).resolve()
            valid = valid and (
                int(train_args.get("epochs", -1)) == int(expected_args["epochs"])
                and train_args.get("time") is None
                and train_args.get("name") == expected_config["run_name"]
                and stored_path_matches(train_args.get("data"), DATASET_YAML)
                and stored_path_matches(
                    train_args.get("save_dir"), expected_run_dir
                )
            )
        if not valid:
            return None
        return {
            "path": Path(path),
            "epoch": int(checkpoint["epoch"]),
            "mtime": Path(path).stat().st_mtime,
        }
    except Exception as error:
        print(f"Checkpoint ignore ({Path(path).name}) : {error}", flush=True)
        return None
    finally:
        if "checkpoint" in locals():
            del checkpoint
        gc.collect()


def find_resume_checkpoint(run_dir, torch, config=None):
    """Compare last, les copies tournantes et le dernier jalon periodique."""

    weights = run_dir / "weights"
    recovery = run_dir / "recovery"
    epoch_files = sorted(
        weights.glob("epoch*.pt"),
        key=lambda path: int(path.stem.removeprefix("epoch")),
        reverse=True,
    )
    first_candidates = [
        weights / "last.pt",
        recovery / "resume_new.pt",
        recovery / "resume_latest.pt",
        recovery / "resume_previous.pt",
    ]

    valid = []
    seen = set()
    for path in first_candidates:
        if path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            info = load_checkpoint_info(path, torch, config)
            if info:
                valid.append(info)

    # Un jalon periodique peut etre plus recent que les copies tournantes.
    best_known_epoch = max(
        (item["epoch"] for item in valid),
        default=-1,
    )
    for path in epoch_files:
        file_epoch = int(path.stem.removeprefix("epoch"))
        if file_epoch <= best_known_epoch:
            break
        info = load_checkpoint_info(path, torch, config)
        if info:
            valid.append(info)
            best_known_epoch = info["epoch"]
            break
    if not valid:
        raise FileNotFoundError("Aucun checkpoint complet et valide n'a ete trouve.")

    selected = max(valid, key=lambda item: (item["epoch"], item["mtime"]))
    print(
        "Checkpoint de reprise :",
        relative(selected["path"]),
        f"(epoque {selected['epoch'] + 1})",
        flush=True,
    )
    return selected


def read_results(path):
    """Lit les lignes valides de results.csv."""

    if not Path(path).is_file():
        return [], None
    with Path(path).open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames


def reconcile_results_csv(path, checkpoint_epoch):
    """Retire les lignes plus recentes que le checkpoint choisi."""

    rows, fieldnames = read_results(path)
    if not rows or not fieldnames:
        return []

    maximum_human_epoch = checkpoint_epoch + 1
    rows_by_epoch = {}
    for row in rows:
        try:
            human_epoch = int(float(row["epoch"]))
        except (KeyError, TypeError, ValueError):
            continue
        if human_epoch <= maximum_human_epoch:
            rows_by_epoch[human_epoch] = row
    clean_rows = [rows_by_epoch[key] for key in sorted(rows_by_epoch)]

    if clean_rows != rows:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = Path(path).with_name(f"results_before_resume_{timestamp}.csv")
        shutil.copy2(path, backup)
        temporary = Path(path).with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(clean_rows)
        os.replace(temporary, path)
        print("results.csv a ete aligne sur le checkpoint choisi.", flush=True)
    return clean_rows


def csv_summary(rows):
    """Reconstruit l'avancement et le meilleur score depuis le CSV."""

    summary = {
        "last_epoch": None,
        "last_validation": {},
        "best_epoch": None,
        "best_map50_95": None,
    }
    metric_columns = {
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "map50",
        "metrics/mAP50-95(B)": "map50_95",
    }
    for row in rows:
        try:
            epoch = int(float(row["epoch"]))
            score = float(row["metrics/mAP50-95(B)"])
        except (KeyError, TypeError, ValueError):
            continue

        if summary["last_epoch"] is None or epoch >= summary["last_epoch"]:
            validation = {"epoch": epoch}
            for source, destination in metric_columns.items():
                try:
                    validation[destination] = round(float(row[source]), 6)
                except (KeyError, TypeError, ValueError):
                    pass
            summary["last_epoch"] = epoch
            summary["last_validation"] = validation

        if summary["best_map50_95"] is None or score > summary["best_map50_95"]:
            summary["best_epoch"] = epoch
            summary["best_map50_95"] = score
    return summary


def copy_checkpoint(source, destination):
    """Copie un checkpoint complet avant de le publier atomiquement."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    try:
        with Path(source).open("rb") as source_file:
            with temporary.open("wb") as destination_file:
                shutil.copyfileobj(source_file, destination_file, 1024**2)
                destination_file.flush()
                os.fsync(destination_file.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_resume_checkpoint(source, run_dir, torch, config):
    """Cree une copie de reprise qui ne recharge pas les poids preentraines.

    Ultralytics restaure d'abord le checkpoint du run. Toutefois, si son
    ``train_args.pretrained`` contient encore un chemin de poids, son
    ``setup_model`` remplace ensuite ces poids par ce fichier externe avant de
    restaurer l'optimiseur et l'EMA. Le modele actif et l'optimiseur deviennent
    alors incoherents et les performances peuvent s'effondrer apres reprise.

    On ne modifie jamais le checkpoint source. Seule une copie atomique voit
    ``pretrained=False``; le modele, l'EMA, l'optimiseur, le scaler et l'epoque
    restent strictement identiques.
    """

    source = Path(source)
    recovery = Path(run_dir) / "recovery"
    destination = recovery / "resume_safe.pt"
    temporary = recovery / "resume_safe_new.pt"
    recovery.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    try:
        train_args = dict(checkpoint.get("train_args") or {})
        previous_pretrained = train_args.get("pretrained")
        train_args["pretrained"] = False
        checkpoint["train_args"] = train_args

        torch.save(checkpoint, temporary)
        # Relire completement la copie avant de la publier. Cette validation
        # detecte une sauvegarde incomplete avant qu'elle puisse etre choisie.
        safe_info = load_checkpoint_info(temporary, torch, config)
        if safe_info is None:
            raise RuntimeError("La copie de reprise securisee est invalide.")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
        del checkpoint
        gc.collect()

    print(
        "Checkpoint de reprise securise : pretrained=False",
        f"(ancienne valeur : {previous_pretrained!r})",
        flush=True,
    )
    return {
        "path": destination,
        "epoch": safe_info["epoch"],
        "mtime": destination.stat().st_mtime,
    }


def inspect_model_checkpoint(path, torch, expected_config=None):
    """Verifie qu'un fichier contient un modele OBB exploitable."""

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = checkpoint.get("ema")
        if model is None:
            model = checkpoint.get("model")
        names = getattr(model, "names", {})
        train_args = checkpoint.get("train_args") or {}
        metrics = checkpoint.get("train_metrics") or {}
        if model is None or train_args.get("task") != "obb" or len(names) != CLASS_COUNT:
            return None

        if expected_config is not None:
            expected_run_dir = (
                RUNS_DIR / expected_config["run_name"]
            ).resolve()
            if not (
                train_args.get("name") == expected_config["run_name"]
                and stored_path_matches(train_args.get("data"), DATASET_YAML)
                and stored_path_matches(
                    train_args.get("save_dir"), expected_run_dir
                )
            ):
                return None

        checkpoint_epoch = int(checkpoint.get("epoch", -1))
        effective_epoch = checkpoint_epoch
        if checkpoint_epoch < 0:
            saved_epochs = (checkpoint.get("train_results") or {}).get("epoch", [])
            if len(saved_epochs):
                effective_epoch = int(float(saved_epochs[-1])) - 1
        return {
            "path": Path(path),
            "epoch": checkpoint_epoch,
            "effective_epoch": effective_epoch,
            "best_fitness": checkpoint.get("best_fitness"),
            "current_fitness": metrics.get("fitness"),
        }
    except Exception:
        return None
    finally:
        if "checkpoint" in locals():
            del checkpoint
        gc.collect()


def repair_best_checkpoint(run_dir, resume_info, torch, config):
    """Restaure best.pt si une coupure l'a laisse incomplet."""

    best_path = run_dir / "weights" / "best.pt"
    best_was_present = best_path.exists()
    candidates = [best_path]
    selected_info = inspect_model_checkpoint(
        resume_info["path"], torch, config
    )
    if selected_info:
        current = selected_info["current_fitness"]
        best = selected_info["best_fitness"]
        if current is not None and best is not None and float(current) == float(best):
            candidates.append(resume_info["path"])

    recovery = run_dir / "recovery"
    candidates.extend(
        [
            recovery / "best_new.pt",
            recovery / "best_latest.pt",
            recovery / "best_previous.pt",
        ]
    )
    compatible = []
    existing_info = None
    for path in candidates:
        if not path.is_file():
            continue
        info = inspect_model_checkpoint(path, torch, config)
        if path == best_path:
            existing_info = info
        if (
            info is not None
            and info["effective_epoch"] >= 0
            and info["effective_epoch"] <= resume_info["epoch"]
        ):
            compatible.append(info)

    replacement = None
    if compatible:
        replacement = max(
            compatible,
            key=lambda info: (
                float(info["current_fitness"])
                if info["current_fitness"] is not None
                else float("-inf"),
                info["effective_epoch"],
            ),
        )["path"]

    if replacement == best_path:
        return

    if best_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = "after_rollback" if existing_info else "corrupt"
        quarantine = recovery / f"best_{label}_{timestamp}.pt"
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        os.replace(best_path, quarantine)

    if replacement is not None:
        copy_checkpoint(replacement, best_path)
        print("best.pt a ete restaure depuis une copie valide.", flush=True)
    elif existing_info is not None:
        print("best.pt plus recent a ete place dans recovery.", flush=True)
    elif best_was_present:
        print("best.pt corrompu a ete place dans recovery.", flush=True)


def build_config(run_name, hours, smoke_minutes=None):
    smoke = smoke_minutes is not None
    target_seconds = (smoke_minutes * 60) if smoke else (hours * 3600)
    return {
        "run_name": run_name,
        "created_at_utc": utc_now(),
        "initial_weights": relative(PILOT_BEST),
        "target_seconds": target_seconds,
        "smoke_test": smoke,
        "trainer_args": {
            "data": str(DATASET_YAML),
            "epochs": FINAL_EPOCHS,
            "time": None,
            "imgsz": 1024,
            "batch": 2,
            "nbs": 64,
            "device": 0,
            "workers": 4,
            "amp": True,
            "optimizer": "AdamW",
            "lr0": 0.0005,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "warmup_epochs": 1.0,
            "patience": 50,
            "save": True,
            "save_period": 25,
            "val": True,
            "plots": True,
            "cache": False,
            "seed": 42,
            "deterministic": True,
            "multi_scale": False,
            "degrees": 180.0,
            "flipud": 0.5,
            "fliplr": 0.5,
            "close_mosaic": 10,
            "project": str(RUNS_DIR),
            "name": run_name,
            "exist_ok": True,
        },
    }


def initial_status(config, mode, previous=None, csv_info=None):
    """Cree un statut neuf tout en conservant le temps des sessions precedentes."""

    previous = previous or {}
    csv_info = csv_info or csv_summary([])
    return {
        "run_name": config["run_name"],
        "mode": mode,
        "pid": os.getpid(),
        "session": int(previous.get("session", 0)) + 1,
        "status": "starting",
        "phase": "initialisation",
        "message": "Chargement du modele et des donnees",
        "heartbeat_at_utc": utc_now(),
        "last_progress_at_utc": None,
        "target_seconds": config["target_seconds"],
        "session_seconds": 0.0,
        "active_seconds": float(previous.get("active_seconds", 0.0)),
        "remaining_seconds": max(
            0.0,
            config["target_seconds"]
            - float(previous.get("active_seconds", 0.0)),
        ),
        "epoch": csv_info["last_epoch"],
        "epochs": config["trainer_args"]["epochs"],
        "last_completed_epoch": csv_info["last_epoch"],
        "batch": 0,
        "batches": None,
        "batch_percent": 0.0,
        "epoch_seconds": None,
        "epoch_eta_seconds": None,
        "losses": {},
        "learning_rates": {},
        "validation": (
            csv_info.get("last_validation") or previous.get("validation", {})
        ),
        "final_validation": previous.get("final_validation", {}),
        "validation_batch": 0,
        "validation_batches": None,
        "validation_percent": 0.0,
        "best": {
            "epoch": csv_info["best_epoch"],
            "map50_95": csv_info["best_map50_95"],
        },
        "checkpoint": previous.get("checkpoint", {}),
        "stop_reason": None,
        "error": None,
    }


class Tracker:
    """Met a jour le moniteur et deux checkpoints tournants."""

    def __init__(self, run_dir, config, status, resume_summary=None):
        self.run_dir = run_dir
        self.config = config
        self.status = status
        self.resume_summary = resume_summary or csv_summary([])
        self.status_file = run_dir / "training_status.json"
        self.stop_file = run_dir / "STOP_REQUESTED"
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.session_start = time.monotonic()
        self.epoch_start = None
        self.base_active = status["active_seconds"]
        self.batch_count = 0
        self.validation_batch_count = 0
        self.saved_epoch = None
        self.time_limit = False
        self.stop_requested = False

    def update_time(self):
        session_seconds = time.monotonic() - self.session_start
        active_seconds = self.base_active + session_seconds
        self.status["session_seconds"] = round(session_seconds, 2)
        self.status["active_seconds"] = round(active_seconds, 2)
        self.status["remaining_seconds"] = round(
            max(0.0, self.config["target_seconds"] - active_seconds), 2
        )
        if self.epoch_start is not None:
            epoch_seconds = time.monotonic() - self.epoch_start
            self.status["epoch_seconds"] = round(epoch_seconds, 2)
            progress = self.status["batch_percent"] / 100
            if progress > 0:
                self.status["epoch_eta_seconds"] = round(
                    epoch_seconds * (1 - progress) / progress, 2
                )

    def save(self):
        with self.lock:
            self.update_time()
            self.status["heartbeat_at_utc"] = utc_now()
            snapshot = deepcopy(self.status)
            # Garder aussi l'ecriture sous le verrou empeche deux sauvegardes
            # du statut de se chevaucher dans ce processus.
            write_json(self.status_file, snapshot)

    def set_state(self, status, phase, message, reason=None, error=None):
        with self.lock:
            self.status.update(
                status=status,
                phase=phase,
                message=message,
                stop_reason=reason,
                error=error,
            )
        self.save()

    def start(self):
        self.save()

        def heartbeat():
            while not self.stop_event.wait(HEARTBEAT_SECONDS):
                try:
                    self.save()
                except Exception as error:
                    # Une erreur temporaire ne doit jamais tuer definitivement
                    # le thread qui alimente le moniteur.
                    print(
                        "AVERTISSEMENT heartbeat : "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )

        self.thread = threading.Thread(target=heartbeat, daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=HEARTBEAT_SECONDS + 1)
        self.save()

    def progress(self):
        self.status["last_progress_at_utc"] = utc_now()

    def on_train_start(self, trainer):
        with self.lock:
            self.status["batches"] = len(trainer.train_loader)
            self.status["status"] = "running"
            self.status["phase"] = "training"
            self.status["message"] = "Entrainement en cours"

        best_score = self.resume_summary.get("best_map50_95")
        best_epoch = self.resume_summary.get("best_epoch")
        checkpoint_best = trainer.best_fitness
        if checkpoint_best is not None:
            trainer.stopper.best_fitness = float(checkpoint_best)
        elif best_score is not None:
            trainer.stopper.best_fitness = best_score
        if best_epoch is not None:
            trainer.stopper.best_epoch = best_epoch
        self.save()

    def on_train_epoch_start(self, trainer):
        with self.lock:
            self.epoch_start = time.monotonic()
            self.batch_count = 0
            self.status.update(
                phase="training",
                message="Entrainement des lots",
                epoch=int(trainer.epoch) + 1,
                batch=0,
                batches=len(trainer.train_loader),
                batch_percent=0.0,
            )
            self.progress()
        self.save()

    def on_train_batch_end(self, trainer):
        with self.lock:
            self.batch_count += 1
            total = max(1, len(trainer.train_loader))
            self.status["batch"] = self.batch_count
            self.status["batches"] = total
            self.status["batch_percent"] = round(100 * self.batch_count / total, 2)

            losses = {}
            if isinstance(trainer.tloss, dict):
                for name, value in trainer.tloss.items():
                    if hasattr(value, "detach"):
                        value = value.detach().item()
                    losses[str(name)] = round(float(value), 6)
            self.status["losses"] = losses
            self.status["learning_rates"] = {
                f"pg{index}": float(group["lr"])
                for index, group in enumerate(trainer.optimizer.param_groups)
            }
            self.progress()

            smoke_finished = (
                self.config["smoke_test"]
                and self.base_active + time.monotonic() - self.session_start
                >= self.config["target_seconds"]
            )
            if self.stop_file.exists():
                self.stop_requested = True
                self.status["phase"] = "stop_requested"
                self.status["message"] = "Arret demande apres cette epoque"

        if smoke_finished:
            self.set_state(
                "stopping", "smoke_test", "Fin du test de cinq minutes",
                reason="smoke_time_limit",
            )
            raise SmokeTestFinished()

    def on_train_epoch_end(self, trainer):
        self.set_state("running", "validation", "Validation de l'epoque")

    def on_val_start(self, validator):
        with self.lock:
            self.validation_batch_count = 0
            dataloader = getattr(validator, "dataloader", None)
            total = len(dataloader) if dataloader is not None else None
            self.status.update(
                status="running",
                phase="validation",
                message="Validation en cours",
                validation_batch=0,
                validation_batches=total,
                validation_percent=0.0,
            )
            self.progress()
        self.save()

    def on_val_batch_end(self, validator):
        with self.lock:
            self.validation_batch_count += 1
            total = self.status.get("validation_batches")
            self.status["validation_batch"] = self.validation_batch_count
            if total:
                self.status["validation_percent"] = round(
                    100 * self.validation_batch_count / total, 2
                )
            self.progress()

    def rotate_copy(self, source, latest_name, previous_name):
        recovery = self.run_dir / "recovery"
        latest = recovery / latest_name
        previous = recovery / previous_name
        new = recovery / latest_name.replace("_latest", "_new")
        copy_checkpoint(Path(source), new)
        if latest.exists():
            os.replace(latest, previous)
        os.replace(new, latest)
        return latest

    def on_model_save(self, trainer):
        self.set_state("running", "saving", "Sauvegarde du checkpoint")
        latest = self.rotate_copy(
            trainer.last,
            "resume_latest.pt",
            "resume_previous.pt",
        )
        if (
            trainer.fitness is not None
            and trainer.best_fitness == trainer.fitness
            and Path(trainer.best).is_file()
        ):
            self.rotate_copy(
                trainer.best,
                "best_latest.pt",
                "best_previous.pt",
            )
        with self.lock:
            self.saved_epoch = int(trainer.epoch)
            self.status["checkpoint"] = {
                "path": relative(latest),
                "epoch": self.saved_epoch + 1,
                "saved_at_utc": utc_now(),
                "size_mib": round(latest.stat().st_size / 1024**2, 2),
            }
            self.progress()
        self.save()

    def update_metrics(self, trainer, epoch, table_name="validation"):
        metrics = trainer.metrics or {}
        mapping = {
            "metrics/precision(B)": "precision",
            "metrics/recall(B)": "recall",
            "metrics/mAP50(B)": "map50",
            "metrics/mAP50-95(B)": "map50_95",
        }
        values = {"epoch": epoch}
        for source, metric_name in mapping.items():
            if source in metrics:
                values[metric_name] = round(float(metrics[source]), 6)
        if len(values) > 1:
            self.status[table_name] = values
        return values

    def update_best(self, values, trainer):
        """Met a jour le meilleur score affiche par le moniteur."""

        if len(values) > 1:
            score = values.get("map50_95")
            old_score = self.status["best"].get("map50_95")
            is_checkpoint_best = (
                trainer.fitness is not None
                and trainer.best_fitness == trainer.fitness
            )
            if score is not None and (
                is_checkpoint_best or old_score is None or score > old_score
            ):
                self.status["best"] = {
                    "epoch": values.get("epoch"),
                    "map50_95": score,
                }

    def on_fit_epoch_end(self, trainer):
        with self.lock:
            final_evaluation = self.saved_epoch != int(trainer.epoch)
            epoch = (
                self.saved_epoch + 1
                if self.saved_epoch is not None
                else int(trainer.epoch) + 1
            )
            if final_evaluation:
                best_epoch = self.status["best"].get("epoch")
                self.update_metrics(
                    trainer,
                    best_epoch,
                    table_name="final_validation",
                )
                self.save()
                return

            values = self.update_metrics(trainer, epoch)
            self.update_best(values, trainer)
            self.status["last_completed_epoch"] = epoch
            self.stop_requested |= self.stop_file.exists()
            active = self.base_active + time.monotonic() - self.session_start
            if active >= self.config["target_seconds"]:
                self.time_limit = True
                trainer.stop = True
                self.status.update(
                    status="stopping",
                    phase="time_limit",
                    message="Budget de temps atteint",
                    stop_reason="time_limit",
                )
            elif self.stop_requested:
                trainer.stop = True
                self.status.update(
                    status="stopping",
                    phase="user_request",
                    message="Arret propre demande",
                    stop_reason="user_request",
                )
            else:
                self.status.update(phase="between_epochs", message="Epoque terminee")
            self.progress()
        self.save()

    def on_train_end(self, trainer):
        with self.lock:
            if self.time_limit:
                reason = "time_limit"
            elif self.stop_requested:
                reason = "user_request"
            elif int(trainer.epoch) + 1 >= int(trainer.epochs):
                reason = "epoch_limit"
            else:
                reason = "early_stopping"
        self.set_state(
            "completed", "finished", "Entrainement termine proprement", reason
        )


def attach_callbacks(model, tracker):
    callbacks = {
        "on_train_start": tracker.on_train_start,
        "on_train_epoch_start": tracker.on_train_epoch_start,
        "on_train_batch_end": tracker.on_train_batch_end,
        "on_train_epoch_end": tracker.on_train_epoch_end,
        "on_val_start": tracker.on_val_start,
        "on_val_batch_end": tracker.on_val_batch_end,
        "on_model_save": tracker.on_model_save,
        "on_fit_epoch_end": tracker.on_fit_epoch_end,
        "on_train_end": tracker.on_train_end,
    }
    for event, callback in callbacks.items():
        model.add_callback(event, callback)


def validate_model(model):
    if model.task != "obb" or len(model.names) != CLASS_COUNT:
        raise RuntimeError("Le checkpoint doit etre un modele OBB de 15 classes.")


def ensure_new_run(run_dir):
    protected = [
        run_dir / "training_config.json",
        run_dir / "training_status.json",
        run_dir / "results.csv",
        run_dir / "weights",
        run_dir / "recovery",
    ]
    if any(path.exists() for path in protected):
        raise FileExistsError("Ce run existe deja; utilise resume ou un autre nom.")


def run_check(torch, yaml, YOLO):
    check_environment(torch, yaml, PILOT_BEST)
    model = YOLO(str(PILOT_BEST), task="obb", verbose=False)
    validate_model(model)
    print("Checkpoint pilote OBB valide : 15 classes", flush=True)
    print("Aucun entrainement n'a ete lance.", flush=True)


def run_training(args, torch, yaml, YOLO, session_log):
    validate_run_name(args.run_name)
    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "start":
        if args.hours <= 0 or (
            args.smoke_minutes is not None and args.smoke_minutes <= 0
        ):
            raise ValueError("La duree doit etre strictement positive.")
        if args.smoke_minutes and args.run_name == FINAL_RUN_NAME:
            raise ValueError("Donne un nom distinct au smoke test.")
        ensure_new_run(run_dir)
        config = build_config(args.run_name, args.hours, args.smoke_minutes)
        weights = PILOT_BEST
        check_environment(torch, yaml, weights)
        write_json(run_dir / "training_config.json", config)
        previous = {}
        resume_info = None
        resume_rows = []
        exact_resume = False
    else:
        config_file = run_dir / "training_config.json"
        if not config_file.is_file():
            raise FileNotFoundError("training_config.json est introuvable.")
        config = read_json(config_file)
        if config.get("run_name") != args.run_name:
            raise RuntimeError("Le nom du run ne correspond pas a sa configuration.")
        if relocate_config_paths(config):
            write_json(config_file, config)
            print("Chemins du run adaptes a cette machine.", flush=True)
        if config["smoke_test"]:
            raise RuntimeError("Un smoke test ne doit pas etre repris.")

        status_file = run_dir / "training_status.json"
        previous = read_json(status_file) if status_file.is_file() else {}
        if (
            previous.get("status") == "completed"
            and previous.get("stop_reason") != "user_request"
        ):
            raise RuntimeError("Ce run est deja termine.")
        if previous.get("active_seconds", 0) >= config["target_seconds"]:
            raise RuntimeError("Le budget de temps de ce run est deja atteint.")
        stop_file = run_dir / "STOP_REQUESTED"
        if stop_file.exists():
            stop_file.unlink()

        try:
            resume_info = find_resume_checkpoint(run_dir, torch, config)
            weights = resume_info["path"]
            exact_resume = True
        except FileNotFoundError:
            if previous.get("last_completed_epoch") is not None:
                raise
            resume_info = None
            resume_rows = []
            weights = absolute(config["initial_weights"])
            exact_resume = False
            print("Aucune epoque complete: retour aux poids initiaux.", flush=True)
        check_environment(torch, yaml, weights)
        if exact_resume:
            repair_best_checkpoint(run_dir, resume_info, torch, config)
            resume_rows = reconcile_results_csv(
                run_dir / "results.csv", resume_info["epoch"]
            )
            # Important : faire cette copie seulement apres les controles et
            # la reconciliation fondes sur le checkpoint original.
            resume_info = prepare_resume_checkpoint(
                resume_info["path"], run_dir, torch, config
            )
            weights = resume_info["path"]

    summary = csv_summary(resume_rows)
    status = initial_status(config, args.command, previous, summary)
    status["session_log"] = relative(session_log)
    if resume_info is not None:
        status["checkpoint"] = {
            "path": relative(resume_info["path"]),
            "epoch": resume_info["epoch"] + 1,
            "saved_at_utc": datetime.fromtimestamp(
                resume_info["mtime"], timezone.utc
            ).isoformat(timespec="seconds"),
            "size_mib": round(resume_info["path"].stat().st_size / 1024**2, 2),
        }
    write_json(run_dir / "training_status.json", status)
    write_json(
        CURRENT_RUN_FILE,
        {
            "run_name": args.run_name,
            "status_path": relative(run_dir / "training_status.json"),
        },
    )

    tracker = Tracker(run_dir, config, status, summary)
    tracker.start()
    try:
        model = YOLO(str(weights), task="obb", verbose=False)
        validate_model(model)
        attach_callbacks(model, tracker)
        if exact_resume:
            print("Reprise de l'etat sauvegarde :", relative(weights), flush=True)
            model.train(
                resume=str(weights),
                data=str(DATASET_YAML),
                save_dir=str(run_dir),
                device=0,
                workers=config["trainer_args"]["workers"],
                save_period=config["trainer_args"]["save_period"],
                patience=config["trainer_args"]["patience"],
                val=True,
                plots=True,
                time=None,
            )
        else:
            print("Nouvelle phase depuis :", relative(weights), flush=True)
            model.train(**config["trainer_args"])
        return 0
    except SmokeTestFinished:
        tracker.set_state(
            "smoke_completed",
            "finished",
            "Test du moniteur termine",
            "smoke_time_limit",
        )
        print("Test technique termine; aucune validation finale.", flush=True)
        return 0
    except KeyboardInterrupt:
        tracker.set_state(
            "interrupted",
            "stopped",
            "Interruption clavier; reprise disponible",
            "keyboard_interrupt",
        )
        return 130
    except Exception as error:
        tracker.set_state(
            "failed",
            "error",
            "Echec de l'entrainement",
            "exception",
            f"{type(error).__name__}: {error}",
        )
        traceback.print_exc()
        return 1
    finally:
        tracker.close()


def main():
    args = parse_arguments()
    if args.command == "check":
        import torch
        import yaml
        from ultralytics import YOLO

        with GpuLock():
            run_check(torch, yaml, YOLO)
            return 0

    validate_run_name(args.run_name)
    run_dir = RUNS_DIR / args.run_name
    with SessionLogger(run_dir) as session:
        import torch
        import yaml
        from ultralytics import YOLO

        with GpuLock():
            return run_training(
                args,
                torch,
                yaml,
                YOLO,
                session.path,
            )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERREUR : {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
