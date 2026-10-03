"""Affiche l'avancement du script train_yolo_obb_v15_clean.py en lecture seule."""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_DIR / "outputs" / "yolo_runs"
CURRENT_RUN_FILE = RUNS_DIR / "current_training.json"
TERMINAL_STATES = {"completed", "failed", "interrupted", "smoke_completed"}
SUPERVISOR_TERMINAL_STATES = {"completed", "stopped", "blocked"}
SUPERVISOR_RESTART_GRACE_SECONDS = 180


def parse_arguments():
    """Lit les trois options utiles du moniteur."""

    parser = argparse.ArgumentParser(
        description="Suit YOLO-OBB sans interrompre ni modifier l'entrainement."
    )
    parser.add_argument(
        "--run-name",
        help="Nom du run. Sans cette option, utilise current_training.json.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Affiche une seule fois le statut puis quitte.",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=2.0,
        help="Secondes entre deux affichages (defaut: 2).",
    )
    args = parser.parse_args()

    if args.refresh <= 0:
        parser.error("--refresh doit etre strictement positif.")
    if args.run_name and (
        Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}
    ):
        parser.error("--run-name doit etre un simple nom de dossier.")
    return args


def read_json(path):
    """Lit un JSON atomique et tolere son absence pendant le demarrage."""

    try:
        with Path(path).open(encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else None
    except (FileNotFoundError, PermissionError, OSError, json.JSONDecodeError):
        return None


def resolve_status_file(run_name):
    """Trouve training_status.json par le nom du run ou par le pointeur."""

    if run_name:
        return RUNS_DIR / run_name / "training_status.json", None

    pointer = read_json(CURRENT_RUN_FILE)
    if pointer is None:
        return None, f"Pointeur introuvable : {CURRENT_RUN_FILE}"

    status_path = pointer.get("status_path")
    if status_path:
        status_file = Path(status_path)
        if not status_file.is_absolute():
            status_file = PROJECT_DIR / status_file
        return status_file, None

    pointed_name = pointer.get("run_name")
    if pointed_name:
        return RUNS_DIR / str(pointed_name) / "training_status.json", None

    return None, "current_training.json ne contient ni status_path ni run_name."


def number(value):
    """Convertit une valeur en float, ou retourne None si elle est absente."""

    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def integer(value):
    """Convertit une valeur numerique en entier."""

    value = number(value)
    return int(value) if value is not None else None


def first(*values):
    """Retourne la premiere valeur disponible, y compris zero."""

    for value in values:
        if value is not None:
            return value
    return None


def read_results(path):
    """Lit les lignes completes de results.csv pendant que YOLO peut l'ecrire."""

    try:
        with Path(path).open(encoding="utf-8", newline="") as file:
            rows = []
            for raw_row in csv.DictReader(file):
                row = {
                    str(key).strip(): str(value).strip()
                    for key, value in raw_row.items()
                    if key is not None and value is not None
                }
                if number(row.get("epoch")) is not None:
                    rows.append(row)
            return rows
    except (FileNotFoundError, PermissionError, OSError, csv.Error, UnicodeError):
        return []


def csv_number(row, column):
    """Lit une colonne numerique Ultralytics."""

    return number(row.get(column)) if row else None


def parse_time(value):
    """Transforme un horodatage ISO UTC en objet datetime."""

    if not isinstance(value, str) or not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.astimezone() if parsed.tzinfo else parsed.astimezone()


def age_seconds(value):
    """Calcule l'age d'un heartbeat ou d'une progression."""

    parsed = parse_time(value)
    if parsed is None:
        return None
    return max(0.0, (datetime.now().astimezone() - parsed).total_seconds())


def duration(value):
    """Affiche une duree en jours et en heures, minutes, secondes."""

    value = number(value)
    if value is None:
        return "--"
    total = max(0, int(round(value)))
    days, rest = divmod(total, 86_400)
    hours, rest = divmod(rest, 3_600)
    minutes, seconds = divmod(rest, 60)
    prefix = f"{days} j " if days else ""
    return f"{prefix}{hours:02d}:{minutes:02d}:{seconds:02d}"


def age(value):
    """Affiche depuis combien de temps une date a ete mise a jour."""

    seconds = age_seconds(value)
    return "--" if seconds is None else f"il y a {duration(seconds)}"


def time_until(value):
    """Affiche le temps restant avant une date future."""

    parsed = parse_time(value)
    if parsed is None:
        return "--"
    seconds = (parsed - datetime.now().astimezone()).total_seconds()
    return "maintenant" if seconds <= 0 else f"dans {duration(seconds)}"


def percent(value):
    """Affiche une proportion comprise entre zero et un en pourcentage."""

    value = number(value)
    return "--" if value is None else f"{100 * value:.2f} %"


def decimal(value, digits=4):
    """Affiche un nombre avec une precision courte."""

    value = number(value)
    return "--" if value is None else f"{value:.{digits}f}"


def windows_process_alive(pid):
    """Verifie le code de sortie du PID sans lui envoyer de signal."""

    process_query = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]

    handle = kernel32.OpenProcess(process_query, False, pid)
    if not handle:
        return None if ctypes.get_last_error() == 5 else False

    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return None
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def process_alive(pid):
    """Verifie si le processus d'entrainement existe encore."""

    pid = integer(pid)
    if pid is None or pid <= 0:
        return None
    if os.name == "nt":
        return windows_process_alive(pid)

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    except OSError:
        return False


def gpu_status():
    """Interroge nvidia-smi sans importer PyTorch ni occuper de VRAM."""

    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,temperature.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    expected_gpu = os.environ.get("DOTA_EXPECTED_GPU", "").strip().casefold()
    if expected_gpu:
        line = next(
            (line for line in lines if expected_gpu in line.casefold()),
            "",
        )
    else:
        line = lines[0] if lines else ""
    values = [item.strip() for item in line.split(",")]
    if len(values) != 5:
        return None
    return {
        "name": values[0],
        "usage": values[1],
        "temperature": values[2],
        "used": values[3],
        "total": values[4],
    }


def best_from_results(rows):
    """Trouve la meilleure mAP50-95 en secours si le statut ne la contient pas."""

    candidates = []
    for row in rows:
        epoch = integer(row.get("epoch"))
        score = csv_number(row, "metrics/mAP50-95(B)")
        if epoch is not None and score is not None:
            candidates.append((score, epoch))
    return max(candidates) if candidates else (None, None)


def recent_epoch_duration_baseline(rows):
    """Estime la duree habituelle avec les dernieres epoques du processus."""

    durations = []
    previous_cumulative_time = None
    for row in rows:
        cumulative_time = number(row.get("time"))
        if cumulative_time is None:
            continue

        # La colonne time repart de zero apres une reprise. Une diminution
        # indique donc une nouvelle session et non une duree negative.
        if (
            previous_cumulative_time is None
            or cumulative_time <= previous_cumulative_time
        ):
            epoch_duration = cumulative_time
        else:
            epoch_duration = cumulative_time - previous_cumulative_time

        if epoch_duration > 0:
            durations.append(epoch_duration)
        previous_cumulative_time = cumulative_time

    return statistics.median(durations[-5:]) if durations else None


def loss_value(losses, latest_row, short_name):
    """Prefere la perte en direct, puis celle de la derniere epoque terminee."""

    return first(
        number(losses.get(f"{short_name}_loss")),
        number(losses.get(short_name)),
        csv_number(latest_row, f"train/{short_name}_loss"),
    )


def learning_rate_text(values, latest_row):
    """Formate les learning rates des groupes de parametres."""

    if isinstance(values, dict):
        rates = [number(value) for value in values.values()]
    else:
        rates = []
    if not any(rate is not None for rate in rates):
        rates = [csv_number(latest_row, f"lr/pg{index}") for index in range(3)]
    rates = [rate for rate in rates if rate is not None]
    return ", ".join(f"{rate:.6g}" for rate in rates) if rates else "--"


def checkpoint_text(checkpoint):
    """Resume le dernier checkpoint sans ouvrir le fichier de poids."""

    if not isinstance(checkpoint, dict) or not checkpoint:
        return "aucun"
    path = checkpoint.get("path", "chemin inconnu")
    epoch = checkpoint.get("epoch", "--")
    size = checkpoint.get("size_mib")
    size_text = f" | {size} Mio" if size is not None else ""
    saved = age(checkpoint.get("saved_at_utc"))
    return f"{path} | epoque {epoch}{size_text} | {saved}"


def print_gpu(gpu):
    """Affiche une ligne GPU courte."""

    if gpu is None:
        print("GPU : nvidia-smi indisponible")
        return
    print(
        f"GPU : {gpu['usage']} % | {gpu['temperature']} C | "
        f"VRAM {gpu['used']}/{gpu['total']} Mio | {gpu['name']}"
    )


def render(status_file, status, rows, gpu, supervisor):
    """Affiche un instantane et retourne l'etat et la vie du processus."""

    print("SUIVI YOLO-OBB")
    print("=" * 76)
    print(f"Run : {status.get('run_name', status_file.parent.name)}")
    print(f"Statut : {status_file}")

    state = str(status.get("status", "inconnu")).lower()
    phase = status.get("phase", "--")
    message = status.get("message")
    pid = integer(status.get("pid"))
    alive = process_alive(pid)
    alive_text = "actif" if alive is True else "arrete" if alive is False else "inconnu"

    heartbeat_age = age_seconds(status.get("heartbeat_at_utc"))
    progress_age = age_seconds(status.get("last_progress_at_utc"))
    print(f"Etat : {state} | phase : {phase}")
    if message:
        print(f"Message : {message}")
    if status.get("session_log"):
        print(f"Journal : {status['session_log']}")
    print(f"Processus : PID {pid if pid is not None else '--'} | {alive_text}")
    print(
        f"Heartbeat : {age(status.get('heartbeat_at_utc'))} | "
        f"derniere progression : {age(status.get('last_progress_at_utc'))}"
    )

    supervisor_state = None
    supervisor_alive = None
    supervisor_heartbeat_age = None
    if supervisor:
        supervisor_state = str(supervisor.get("state", "inconnu")).casefold()
        supervisor_pid = integer(supervisor.get("supervisor_pid"))
        supervisor_alive = process_alive(supervisor_pid)
        supervisor_alive_text = (
            "actif"
            if supervisor_alive is True
            else "arrete"
            if supervisor_alive is False
            else "inconnu"
        )
        supervisor_heartbeat_age = age_seconds(
            supervisor.get("heartbeat_at_utc")
        )
        print(
            f"Superviseur : PID {supervisor_pid or '--'} | "
            f"{supervisor_alive_text} | etat : {supervisor_state}"
        )
        print(
            "Reprises : "
            f"{integer(supervisor.get('crash_count')) or 0} crash(s) | "
            f"{integer(supervisor.get('restart_count')) or 0} tentative(s) | "
            f"{integer(supervisor.get('successful_restart_count')) or 0} "
            "confirmee(s)"
        )
        print(
            "Superviseur externe : "
            f"{integer(supervisor.get('supervisor_recovery_count')) or 0} "
            "recuperation(s) par Windows | "
            f"echecs consecutifs "
            f"{integer(supervisor.get('consecutive_failures')) or 0}/"
            f"{integer(supervisor.get('max_consecutive_failures')) or '--'}"
        )
        if supervisor.get("next_retry_at_utc"):
            print(
                "Prochaine tentative : "
                f"{time_until(supervisor.get('next_retry_at_utc'))}"
            )
        if supervisor.get("message"):
            print(f"Supervision : {supervisor['message']}")

    if alive is False and state not in TERMINAL_STATES:
        if (
            supervisor_alive is True
            and supervisor_state not in SUPERVISOR_TERMINAL_STATES
        ):
            print("INFO : trainer arrete; la reprise automatique est prise en charge.")
        else:
            print("ALERTE : trainer et superviseur ne sont pas actifs.")
    elif alive is True and heartbeat_age is not None and heartbeat_age > 30:
        print("ALERTE : le heartbeat ne se met plus a jour.")
    elif (
        alive is True
        and phase == "training"
        and progress_age is not None
        and progress_age > 120
    ):
        print("ALERTE : aucun nouveau batch depuis plus de deux minutes.")

    if (
        supervisor
        and supervisor_alive is False
        and supervisor_state not in SUPERVISOR_TERMINAL_STATES
    ):
        print(
            "ALERTE : le superviseur est arrete; la tache Windows doit le "
            "relancer dans environ une minute."
        )

    epoch = first(status.get("epoch"), rows[-1].get("epoch") if rows else None)
    epochs = status.get("epochs")
    batch = status.get("batch")
    batches = status.get("batches")
    batch_percent = number(status.get("batch_percent"))

    print("-" * 76)
    if phase == "validation":
        validation_batch = status.get("validation_batch")
        validation_batches = status.get("validation_batches")
        validation_percent = number(status.get("validation_percent"))
        validation_progress = (
            f"{validation_percent:.2f} %"
            if validation_percent is not None
            else "--"
        )
        validation_batch_text = (
            validation_batch if validation_batch is not None else "--"
        )
        validation_batches_text = (
            validation_batches if validation_batches is not None else "?"
        )
        print(
            f"Epoque {epoch or '--'}/{epochs or '?'} | validation en cours | "
            f"batch {validation_batch_text}/{validation_batches_text} | "
            f"{validation_progress}"
        )
    else:
        progress_text = f"{batch_percent:.2f} %" if batch_percent is not None else "--"
        print(
            f"Epoque {epoch or '--'}/{epochs or '?'} | "
            f"batch {batch or '--'}/{batches or '?'} | {progress_text}"
        )
    print(f"Derniere epoque terminee : {status.get('last_completed_epoch') or '--'}")
    print(
        f"Temps : session {duration(status.get('session_seconds'))} | "
        f"cumule {duration(status.get('active_seconds'))}"
        f"/{duration(status.get('target_seconds'))} | "
        f"restant {duration(status.get('remaining_seconds'))}"
    )
    print(
        f"Epoque : ecoulee {duration(status.get('epoch_seconds'))} | "
        f"ETA {duration(status.get('epoch_eta_seconds'))}"
    )

    # Comparer la duree projetee a l'historique permet de reperer un batch
    # trop ambitieux, une saturation de RAM ou une forte contention disque.
    epoch_seconds = number(status.get("epoch_seconds"))
    baseline_seconds = recent_epoch_duration_baseline(rows)
    if (
        phase == "training"
        and epoch_seconds is not None
        and epoch_seconds >= 300
        and batch_percent is not None
        and batch_percent >= 10
        and baseline_seconds is not None
    ):
        projected_epoch_seconds = epoch_seconds / (batch_percent / 100)
        if projected_epoch_seconds > baseline_seconds * 1.8:
            print(
                "ALERTE : cette epoque est projetee a "
                f"{duration(projected_epoch_seconds)}, contre environ "
                f"{duration(baseline_seconds)} habituellement."
            )

    latest_row = rows[-1] if rows else {}
    losses = status.get("losses") if isinstance(status.get("losses"), dict) else {}
    print("-" * 76)
    print(
        "Pertes : "
        f"box {decimal(loss_value(losses, latest_row, 'box'))} | "
        f"cls {decimal(loss_value(losses, latest_row, 'cls'))} | "
        f"dfl {decimal(loss_value(losses, latest_row, 'dfl'))} | "
        f"angle {decimal(loss_value(losses, latest_row, 'angle'))}"
    )
    print(
        "Learning rate : "
        f"{learning_rate_text(status.get('learning_rates'), latest_row)}"
    )

    validation = status.get("validation")
    validation = validation if isinstance(validation, dict) else {}
    precision = first(validation.get("precision"), csv_number(latest_row, "metrics/precision(B)"))
    recall = first(validation.get("recall"), csv_number(latest_row, "metrics/recall(B)"))
    map50 = first(validation.get("map50"), csv_number(latest_row, "metrics/mAP50(B)"))
    map50_95 = first(validation.get("map50_95"), csv_number(latest_row, "metrics/mAP50-95(B)"))
    validation_epoch = first(validation.get("epoch"), latest_row.get("epoch"))
    print(
        f"Validation epoque {validation_epoch or '--'} : "
        f"P {percent(precision)} | R {percent(recall)} | "
        f"mAP50 {percent(map50)} | mAP50-95 {percent(map50_95)}"
    )

    final_validation = status.get("final_validation")
    if isinstance(final_validation, dict) and final_validation:
        print(
            f"Validation finale du meilleur modele (epoque "
            f"{final_validation.get('epoch') or '--'}) : "
            f"P {percent(final_validation.get('precision'))} | "
            f"R {percent(final_validation.get('recall'))} | "
            f"mAP50 {percent(final_validation.get('map50'))} | "
            f"mAP50-95 {percent(final_validation.get('map50_95'))}"
        )

    best = status.get("best") if isinstance(status.get("best"), dict) else {}
    csv_best_score, csv_best_epoch = best_from_results(rows)
    best_score = first(best.get("map50_95"), csv_best_score)
    best_epoch = first(best.get("epoch"), csv_best_epoch)
    print(f"Meilleur mAP50-95 : {percent(best_score)} | epoque {best_epoch or '--'}")
    print(f"Checkpoint : {checkpoint_text(status.get('checkpoint'))}")
    print_gpu(gpu)

    if status.get("error"):
        print(f"Erreur : {status['error']}")
    print("-" * 76)
    print(f"Rafraichi : {datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(
        "Ctrl+C ferme uniquement ce moniteur; il ne modifie ni le trainer "
        "ni le superviseur."
    )
    return (
        state,
        alive,
        heartbeat_age,
        supervisor_state,
        supervisor_alive,
        supervisor_heartbeat_age,
    )


def clear_screen():
    """Nettoie seulement le terminal interactif du moniteur."""

    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def main():
    args = parse_arguments()
    cached_gpu = None
    last_gpu_query = float("-inf")

    try:
        while True:
            status_file, error = resolve_status_file(args.run_name)
            status = read_json(status_file) if status_file else None

            now = time.monotonic()
            if now - last_gpu_query >= 5:
                cached_gpu = gpu_status()
                last_gpu_query = now

            if not args.once:
                clear_screen()
            if status_file is None or status is None:
                print("SUIVI YOLO-OBB")
                print("=" * 76)
                print(error or f"Statut en attente : {status_file}")
                print_gpu(cached_gpu)
                state = None
                alive = None
                heartbeat_age = None
                supervisor_state = None
                supervisor_alive = None
                supervisor_heartbeat_age = None
            else:
                rows = read_results(status_file.parent / "results.csv")
                supervisor = read_json(
                    status_file.parent / "supervisor_status.json"
                )
                (
                    state,
                    alive,
                    heartbeat_age,
                    supervisor_state,
                    supervisor_alive,
                    supervisor_heartbeat_age,
                ) = render(
                    status_file, status, rows, cached_gpu, supervisor
                )

            if args.once:
                return 0 if status is not None else 1
            supervisor_may_restart = (
                supervisor_state not in {None, *SUPERVISOR_TERMINAL_STATES}
                and (
                    supervisor_alive is True
                    or supervisor_heartbeat_age is None
                    or supervisor_heartbeat_age
                    <= SUPERVISOR_RESTART_GRACE_SECONDS
                )
            )
            if (
                state in TERMINAL_STATES
                and not supervisor_may_restart
                and (
                    alive is not True
                    or (heartbeat_age is not None and heartbeat_age > 30)
                )
            ):
                return 0
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\nMoniteur ferme. L'entrainement continue en arriere-plan.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
