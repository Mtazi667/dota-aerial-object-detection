"""Supervise un entrainement YOLO et le reprend apres un arret anormal.

Ce processus n'importe jamais PyTorch dans sa propre memoire. Il reste donc
independant du contexte CUDA susceptible de faire tomber l'entrainement. Le
Planificateur de taches Windows peut, a son tour, redemarrer ce superviseur.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import msvcrt
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_DIR / "outputs" / "yolo_runs"
DEFAULT_RUN_NAME = "yolo11s_obb_v15_clean_coco_55h"
DEFAULT_TRAINER = "train_yolo_obb_v15_clean.py"
DEFAULT_EXPECTED_GPU = "RTX 5080"

POLL_SECONDS = 5.0
GPU_RECHECK_SECONDS = 30.0
MINIMUM_FREE_VRAM_FRACTION = 0.85
BASE_RESTART_DELAY_SECONDS = 60.0
MAX_RESTART_DELAY_SECONDS = 900.0
MAX_CONSECUTIVE_FAILURES = 5
MAX_TOTAL_RESTARTS = 25

NORMAL_STOP_REASONS = {
    "time_limit",
    "epoch_limit",
    "early_stopping",
    "smoke_time_limit",
}
MANUAL_STOP_REASONS = {"user_request", "keyboard_interrupt"}
NORMAL_TERMINAL_STATES = {"completed", "smoke_completed"}


class SupervisorAlreadyRunning(RuntimeError):
    """Indique qu'un autre superviseur possede deja le verrou de ce run."""


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Relance automatiquement YOLO apres un crash, sans relancer "
            "une fin normale ou un arret demande."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add_common(command_parser):
        command_parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
        command_parser.add_argument("--trainer", default=DEFAULT_TRAINER)
        command_parser.add_argument(
            "--expected-gpu", default=DEFAULT_EXPECTED_GPU
        )

    check = commands.add_parser(
        "check", help="Verifie les fichiers et le GPU sans rien lancer."
    )
    add_common(check)

    start = commands.add_parser(
        "start", help="Cree un nouveau run sous supervision."
    )
    add_common(start)
    start.add_argument("--hours", type=float, default=55.0)

    resume = commands.add_parser(
        "resume", help="Reprend et supervise un run existant."
    )
    add_common(resume)
    resume.add_argument(
        "--resume-after-user-stop",
        action="store_true",
        help="Autorise explicitement une reprise apres un ancien arret manuel.",
    )

    for command_parser in (start, resume):
        command_parser.add_argument(
            "--max-consecutive-failures",
            type=int,
            default=MAX_CONSECUTIVE_FAILURES,
        )
        command_parser.add_argument(
            "--max-total-restarts",
            type=int,
            default=MAX_TOTAL_RESTARTS,
        )
        command_parser.add_argument(
            "--base-delay",
            type=float,
            default=BASE_RESTART_DELAY_SECONDS,
        )
        command_parser.add_argument(
            "--max-delay",
            type=float,
            default=MAX_RESTART_DELAY_SECONDS,
        )

    args = parser.parse_args()
    validate_run_name(args.run_name)
    if Path(args.trainer).name != args.trainer or not args.trainer.endswith(".py"):
        parser.error("--trainer doit etre un simple nom de fichier Python.")
    if args.command == "start" and args.hours <= 0:
        parser.error("--hours doit etre strictement positif.")
    if args.command in {"start", "resume"}:
        if args.max_consecutive_failures <= 0 or args.max_total_restarts <= 0:
            parser.error("Les limites de relance doivent etre positives.")
        if args.base_delay < 0 or args.max_delay < args.base_delay:
            parser.error("Les delais de relance sont invalides.")
    return args


def validate_run_name(name):
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Le nom du run doit etre un simple nom de dossier.")


def read_json(path):
    try:
        with Path(path).open(encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, PermissionError, OSError, json.JSONDecodeError):
        return {}


def write_json(path, value):
    """Publie un JSON atomiquement, meme si le moniteur le lit en parallele."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
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
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def number(value, default=None):
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value, default=None):
    converted = number(value)
    return int(converted) if converted is not None else default


def windows_process_alive(pid):
    """Interroge Windows sans envoyer de signal au processus cible."""

    process_query = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]

    handle = kernel32.OpenProcess(process_query, False, int(pid))
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
    pid = integer(pid)
    if pid is None or pid <= 0:
        return False
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


class SupervisorLock:
    """Empeche deux watchdogs de relancer simultanement le meme run."""

    def __init__(self, path):
        self.path = Path(path)
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+b")
        self.file.seek(0, os.SEEK_END)
        if self.file.tell() == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            self.file.close()
            raise SupervisorAlreadyRunning(
                "Un superviseur est deja actif pour ce run."
            ) from error
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.file is not None:
            self.file.seek(0)
            msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            self.file.close()


class SupervisorStatus:
    """Conserve les compteurs meme si Windows relance le superviseur."""

    def __init__(self, run_dir, args, log_path):
        self.path = Path(run_dir) / "supervisor_status.json"
        previous = read_json(self.path)
        same_run = previous.get("run_name") == args.run_name
        previous = previous if same_run else {}

        # Si la session precedente n'avait pas declare une fin volontaire et
        # que son PID a disparu, la nouvelle instance a bien recupere la mort
        # du superviseur lui-meme (ou un redemarrage de Windows). Ce compteur
        # est distinct des crashs du processus d'entrainement.
        previous_supervisor_pid = integer(previous.get("supervisor_pid"))
        previous_supervisor_state = str(previous.get("state", "")).casefold()
        previous_supervisor_alive = process_alive(previous_supervisor_pid)
        previous_supervisor_was_interrupted = bool(previous) and (
            previous_supervisor_alive is False
            and previous_supervisor_state
            not in {"completed", "stopped", "blocked"}
        )
        supervisor_start_count = int(
            previous.get("supervisor_start_count", 0)
        ) + 1
        supervisor_recovery_count = int(
            previous.get("supervisor_recovery_count", 0)
        ) + int(previous_supervisor_was_interrupted)

        self.value = {
            "schema_version": 1,
            "run_name": args.run_name,
            "supervisor_pid": os.getpid(),
            "child_pid": None,
            "state": "starting",
            "message": "Initialisation du superviseur",
            "session_started_at_utc": utc_now(),
            "first_started_at_utc": previous.get(
                "first_started_at_utc", utc_now()
            ),
            "heartbeat_at_utc": utc_now(),
            "log_path": relative(log_path),
            "trainer": args.trainer,
            "supervisor_start_count": supervisor_start_count,
            "supervisor_recovery_count": supervisor_recovery_count,
            "crash_count": int(previous.get("crash_count", 0)),
            "restart_count": int(previous.get("restart_count", 0)),
            "successful_restart_count": int(
                previous.get("successful_restart_count", 0)
            ),
            "consecutive_failures": int(
                previous.get("consecutive_failures", 0)
            ),
            "max_consecutive_failures": int(
                getattr(args, "max_consecutive_failures", 0)
            ),
            "max_total_restarts": int(
                getattr(args, "max_total_restarts", 0)
            ),
            "last_exit_code": previous.get("last_exit_code"),
            "last_completed_epoch": previous.get("last_completed_epoch"),
            "last_counted_training_pid": previous.get(
                "last_counted_training_pid"
            ),
            "last_confirmed_training_pid": previous.get(
                "last_confirmed_training_pid"
            ),
            "next_retry_at_utc": None,
            "gpu": previous.get("gpu"),
            "error": None,
        }
        self.save()

    def update(self, **values):
        self.value.update(values)
        self.save()

    def save(self):
        self.value["heartbeat_at_utc"] = utc_now()
        write_json(self.path, self.value)


def relative(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)


def configure_logger(run_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(run_dir) / "logs" / (
        f"supervisor_{timestamp}_{os.getpid()}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"yolo_supervisor_{os.getpid()}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger, log_path


def validate_layout(args):
    run_dir = RUNS_DIR / args.run_name
    python_path = PROJECT_DIR / ".venv_gpu" / "Scripts" / "python.exe"
    trainer_path = PROJECT_DIR / "src" / args.trainer
    if not python_path.is_file():
        raise FileNotFoundError(f"Python GPU introuvable : {python_path}")
    if not trainer_path.is_file():
        raise FileNotFoundError(f"Trainer introuvable : {trainer_path}")

    config_path = run_dir / "training_config.json"
    if args.command in {"resume", "check"} and not config_path.is_file():
        raise FileNotFoundError(f"Configuration du run introuvable : {config_path}")
    if args.command == "start" and config_path.exists():
        raise FileExistsError(
            "Ce run existe deja; utilise la commande resume du superviseur."
        )
    if config_path.is_file():
        config = read_json(config_path)
        if config.get("run_name") != args.run_name:
            raise RuntimeError("Le nom du run et sa configuration different.")
    return run_dir, python_path, trainer_path


def gpu_snapshot(expected_gpu):
    command = [
        "nvidia-smi.exe" if os.name == "nt" else "nvidia-smi",
        "--query-gpu=index,name,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as error:
        return None, f"nvidia-smi indisponible : {error}"
    if result.returncode != 0:
        return None, f"nvidia-smi retourne {result.returncode}."

    matches = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        index, name, free_text, total_text = fields
        if expected_gpu.casefold() not in name.casefold():
            continue
        try:
            free_mib = float(free_text)
            total_mib = float(total_text)
        except ValueError:
            continue
        matches.append({
            "physical_index": index,
            "name": name,
            "free_mib": free_mib,
            "total_mib": total_mib,
            "free_fraction": free_mib / max(total_mib, 1.0),
        })
    if len(matches) != 1:
        return None, (
            f"Une seule {expected_gpu} est requise; trouve : {len(matches)}."
        )
    return matches[0], None


def child_environment(gpu, expected_gpu):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu["physical_index"])
    environment["DOTA_EXPECTED_GPU"] = expected_gpu
    return environment


def torch_cuda_probe(python_path, environment, expected_gpu):
    """Verifie CUDA dans un processus jetable qui libere ensuite sa VRAM."""

    code = (
        "import torch; "
        "assert torch.cuda.is_available(), 'CUDA indisponible'; "
        "name=torch.cuda.get_device_name(0); "
        f"assert {expected_gpu!r}.casefold() in name.casefold(), name; "
        "x=torch.ones(1, device='cuda'); torch.cuda.synchronize(); "
        "print(name)"
    )
    try:
        result = subprocess.run(
            [str(python_path), "-c", code],
            cwd=PROJECT_DIR,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=90,
            check=False,
            creationflags=(
                subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            ),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def completed_epoch(training_status):
    return integer(training_status.get("last_completed_epoch"), -1)


def terminal_reason(run_dir, training_status, allow_user_resume=False):
    """Retourne la raison qui interdit une reprise, sinon None."""

    if (
        (Path(run_dir) / "STOP_REQUESTED").exists()
        and not allow_user_resume
    ):
        return "arret manuel demande par STOP_REQUESTED"

    state = str(training_status.get("status", "")).casefold()
    reason = str(training_status.get("stop_reason", "")).casefold()
    if state in NORMAL_TERMINAL_STATES:
        # STOP_REQUESTED est la preuve persistante d'un arret voulu. Le
        # retirer manuellement constitue donc une autorisation explicite de
        # reprendre un ancien statut user_request.
        if reason == "user_request" and (
            allow_user_resume
            or not (Path(run_dir) / "STOP_REQUESTED").exists()
        ):
            return None
        return reason or f"etat terminal {state}"
    if reason in NORMAL_STOP_REASONS:
        return reason
    if reason in MANUAL_STOP_REASONS and not allow_user_resume:
        return reason

    active = number(training_status.get("active_seconds"), 0.0)
    target = number(training_status.get("target_seconds"))
    if target is not None and active >= target:
        return "budget de temps atteint"
    return None


def record_crash(supervisor, training_status, training_pid, baseline_epoch):
    """Compte une mort une seule fois et gere le circuit breaker."""

    current_epoch = completed_epoch(training_status)
    previous_counted_pid = integer(
        supervisor.value.get("last_counted_training_pid")
    )
    # Une nouvelle instance du superviseur peut retrouver le meme trainer
    # deja mort que l'instance precedente avait compte. Ne jamais compter
    # deux fois le meme PID ni faire avancer artificiellement le coupe-circuit.
    if training_pid is not None and training_pid == previous_counted_pid:
        return (
            current_epoch,
            int(supervisor.value.get("consecutive_failures", 0)),
        )

    crash_count = int(supervisor.value.get("crash_count", 0))
    if training_pid is None or training_pid != previous_counted_pid:
        crash_count += 1

    if current_epoch > baseline_epoch:
        consecutive = 1
    else:
        consecutive = int(supervisor.value.get("consecutive_failures", 0)) + 1

    supervisor.update(
        crash_count=crash_count,
        consecutive_failures=consecutive,
        last_completed_epoch=current_epoch,
        last_counted_training_pid=training_pid,
    )
    return current_epoch, consecutive


def restart_delay(args, consecutive_failures):
    exponent = max(0, consecutive_failures - 1)
    return min(args.max_delay, args.base_delay * (2**exponent))


def sleep_with_heartbeat(supervisor, run_dir, seconds, state, message):
    deadline = time.monotonic() + max(0.0, seconds)
    retry_at = datetime.now(timezone.utc) + timedelta(seconds=max(0.0, seconds))
    while True:
        if (Path(run_dir) / "STOP_REQUESTED").exists():
            supervisor.update(
                state="stopped",
                message="Arret manuel detecte pendant l'attente",
                child_pid=None,
                next_retry_at_utc=None,
            )
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        supervisor.update(
            state=state,
            message=message,
            next_retry_at_utc=retry_at.isoformat(timespec="seconds"),
        )
        time.sleep(min(POLL_SECONDS, remaining))


def wait_for_gpu(supervisor, run_dir, python_path, expected_gpu, logger):
    """Attend sans consommer une tentative tant que le GPU n'est pas sain."""

    while True:
        if (Path(run_dir) / "STOP_REQUESTED").exists():
            return None, None
        gpu, error = gpu_snapshot(expected_gpu)
        if gpu is not None:
            environment = child_environment(gpu, expected_gpu)
            enough_vram = gpu["free_fraction"] >= MINIMUM_FREE_VRAM_FRACTION
            if enough_vram and torch_cuda_probe(
                python_path, environment, expected_gpu
            ):
                supervisor.update(
                    state="gpu_ready",
                    message="GPU valide; lancement de la reprise",
                    gpu=gpu,
                    next_retry_at_utc=None,
                )
                return gpu, environment
            error = (
                f"GPU non pret : {gpu['free_mib']:.0f}/"
                f"{gpu['total_mib']:.0f} Mio libres"
            )

        logger.warning("Attente GPU : %s", error)
        supervisor.update(
            state="waiting_gpu",
            message=error or "GPU non pret",
            gpu=gpu,
        )
        if not sleep_with_heartbeat(
            supervisor,
            run_dir,
            GPU_RECHECK_SECONDS,
            "waiting_gpu",
            error or "GPU non pret",
        ):
            return None, None


def build_child_command(args, python_path, trainer_path, mode):
    command = [
        str(python_path),
        "-u",
        str(trainer_path),
        mode,
        "--run-name",
        args.run_name,
    ]
    if mode == "start":
        command.extend(["--hours", str(args.hours)])
    return command


def launch_child(args, python_path, trainer_path, mode, environment):
    command = build_child_command(
        args, python_path, trainer_path, mode
    )
    return subprocess.Popen(
        command,
        cwd=PROJECT_DIR,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
    )


def confirm_restart_if_ready(supervisor, training_status, child_pid):
    """Compte une relance reussie seulement lorsque le trainer est vraiment actif."""

    if integer(training_status.get("pid")) != child_pid:
        return
    if str(training_status.get("status", "")).casefold() != "running":
        return
    if integer(supervisor.value.get("last_confirmed_training_pid")) == child_pid:
        return
    supervisor.update(
        successful_restart_count=int(
            supervisor.value.get("successful_restart_count", 0)
        ) + 1,
        last_confirmed_training_pid=child_pid,
        message="Relance confirmee; entrainement actif",
    )


def monitor_pid(supervisor, run_dir, pid, child_process=None, is_restart=False):
    """Suit un trainer lance par nous ou deja actif avant le superviseur."""

    while True:
        alive = (
            child_process.poll() is None
            if child_process is not None
            else process_alive(pid)
        )
        if not alive:
            return child_process.returncode if child_process is not None else None

        training_status = read_json(Path(run_dir) / "training_status.json")
        if is_restart:
            confirm_restart_if_ready(supervisor, training_status, pid)
        supervisor.update(
            state="monitoring",
            message="Entrainement actif sous supervision",
            child_pid=pid,
            last_completed_epoch=completed_epoch(training_status),
            next_retry_at_utc=None,
        )
        time.sleep(POLL_SECONDS)


def circuit_breaker_reached(supervisor, args):
    return (
        int(supervisor.value.get("consecutive_failures", 0))
        >= args.max_consecutive_failures
        or int(supervisor.value.get("restart_count", 0))
        >= args.max_total_restarts
    )


def run_check(args):
    run_dir, python_path, trainer_path = validate_layout(args)
    gpu, error = gpu_snapshot(args.expected_gpu)
    if gpu is None:
        raise RuntimeError(error or "GPU introuvable.")
    environment = child_environment(gpu, args.expected_gpu)
    if not torch_cuda_probe(python_path, environment, args.expected_gpu):
        raise RuntimeError("Le test CUDA jetable a echoue.")
    print("Superviseur pret.")
    print("Run :", run_dir)
    print("Trainer :", trainer_path.name)
    print("GPU :", gpu["name"])
    print(
        "VRAM libre :",
        f"{gpu['free_mib'] / 1024:.2f}/{gpu['total_mib'] / 1024:.2f} Gio",
    )
    print("Aucun entrainement n'a ete lance.")
    return 0


def run_supervisor(args):
    run_dir, python_path, trainer_path = validate_layout(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger, log_path = configure_logger(run_dir)

    with SupervisorLock(run_dir / "supervisor.lock"):
        supervisor = SupervisorStatus(run_dir, args, log_path)
        logger.info(
            "Superviseur lance: run=%s, commande=%s, pid=%s",
            args.run_name,
            args.command,
            os.getpid(),
        )

        mode = args.command
        training_status = read_json(run_dir / "training_status.json")
        allow_user_resume = bool(
            getattr(args, "resume_after_user_stop", False)
        )
        reason = terminal_reason(run_dir, training_status, allow_user_resume)
        if reason:
            supervisor.update(
                state="completed",
                message=f"Aucune relance : {reason}",
                child_pid=None,
            )
            logger.info("Aucune relance : %s", reason)
            return 0

        if allow_user_resume:
            # Le drapeau explicite est l'autorisation de retirer l'ancien
            # marqueur. Sans ce drapeau, le superviseur ne touche jamais a
            # STOP_REQUESTED.
            (run_dir / "STOP_REQUESTED").unlink(missing_ok=True)

        existing_pid = integer(training_status.get("pid"))
        existing_alive = process_alive(existing_pid)
        baseline_epoch = completed_epoch(training_status)
        restart_pending = mode == "resume"
        intentional_resume = (
            str(training_status.get("status", "")).casefold() == "completed"
            and str(training_status.get("stop_reason", "")).casefold()
            == "user_request"
            and not (run_dir / "STOP_REQUESTED").exists()
        )
        if intentional_resume:
            # L'utilisateur vient de retirer STOP_REQUESTED pour reprendre.
            # Ce depart volontaire n'est ni un crash ni une auto-relance.
            restart_pending = False

        if existing_alive:
            logger.info("Trainer existant adopte : PID %s", existing_pid)
            supervisor.update(
                state="monitoring",
                message="Trainer existant adopte",
                child_pid=existing_pid,
            )
            monitor_pid(supervisor, run_dir, existing_pid)
            training_status = read_json(run_dir / "training_status.json")
            reason = terminal_reason(
                run_dir,
                training_status,
                allow_user_resume=allow_user_resume,
            )
            if reason:
                supervisor.update(
                    state="completed",
                    message=f"Fin sans relance : {reason}",
                    child_pid=None,
                )
                return 0
            baseline_epoch, _ = record_crash(
                supervisor, training_status, existing_pid, baseline_epoch
            )
            restart_pending = True
        elif mode == "resume" and not intentional_resume:
            # Le statut non terminal et le PID mort de l'incident courant
            # constituent deja le premier crash a recuperer.
            baseline_epoch, _ = record_crash(
                supervisor, training_status, existing_pid, baseline_epoch
            )

        while True:
            training_status = read_json(run_dir / "training_status.json")
            reason = terminal_reason(
                run_dir,
                training_status,
                allow_user_resume=allow_user_resume,
            )
            if reason:
                supervisor.update(
                    state="completed",
                    message=f"Fin sans relance : {reason}",
                    child_pid=None,
                    next_retry_at_utc=None,
                )
                logger.info("Supervision terminee : %s", reason)
                return 0

            if circuit_breaker_reached(supervisor, args):
                message = (
                    "Circuit breaker atteint; intervention manuelle requise"
                )
                supervisor.update(
                    state="blocked",
                    message=message,
                    child_pid=None,
                    next_retry_at_utc=None,
                )
                logger.error(message)
                # Une sortie zero empeche le Planificateur de contourner
                # volontairement ce garde-fou en relancant le superviseur.
                return 0

            if restart_pending:
                delay = restart_delay(
                    args,
                    int(supervisor.value.get("consecutive_failures", 1)),
                )
                logger.warning("Relance dans %.0f secondes.", delay)
                if not sleep_with_heartbeat(
                    supervisor,
                    run_dir,
                    delay,
                    "backoff",
                    "Arret anormal detecte; relance automatique en attente",
                ):
                    return 0

            gpu, environment = wait_for_gpu(
                supervisor,
                run_dir,
                python_path,
                args.expected_gpu,
                logger,
            )
            if gpu is None:
                return 0

            child_mode = "resume" if restart_pending else mode
            if restart_pending:
                supervisor.value["restart_count"] = int(
                    supervisor.value.get("restart_count", 0)
                ) + 1
            supervisor.update(
                state="launching",
                message=f"Lancement du trainer en mode {child_mode}",
                gpu=gpu,
            )
            try:
                child = launch_child(
                    args,
                    python_path,
                    trainer_path,
                    child_mode,
                    environment,
                )
                # L'autorisation ne vaut que pour franchir l'ancien statut
                # user_request et lancer une seule reprise. Toute nouvelle
                # demande d'arret sera ensuite respectee normalement.
                allow_user_resume = False
            except OSError as error:
                logger.exception("Echec de lancement du trainer")
                training_status = read_json(run_dir / "training_status.json")
                baseline_epoch, _ = record_crash(
                    supervisor, training_status, None, baseline_epoch
                )
                supervisor.update(
                    state="restart_pending",
                    message=f"Echec de lancement : {error}",
                    error=f"{type(error).__name__}: {error}",
                )
                restart_pending = True
                continue

            logger.info(
                "Trainer lance : PID %s, mode=%s", child.pid, child_mode
            )
            supervisor.update(
                state="monitoring",
                message="Trainer lance; attente de confirmation",
                child_pid=child.pid,
                error=None,
            )
            exit_code = monitor_pid(
                supervisor,
                run_dir,
                child.pid,
                child_process=child,
                is_restart=restart_pending,
            )
            supervisor.update(
                child_pid=None,
                last_exit_code=exit_code,
            )
            logger.warning(
                "Trainer termine : PID %s, code=%s", child.pid, exit_code
            )

            training_status = read_json(run_dir / "training_status.json")
            reason = terminal_reason(run_dir, training_status)
            if reason:
                supervisor.update(
                    state="completed",
                    message=f"Fin sans relance : {reason}",
                    child_pid=None,
                    next_retry_at_utc=None,
                )
                return 0

            baseline_epoch, _ = record_crash(
                supervisor,
                training_status,
                child.pid,
                baseline_epoch,
            )
            supervisor.update(
                state="restart_pending",
                message="Trainer arrete anormalement; reprise programmee",
                error=training_status.get("error"),
            )
            restart_pending = True


def main():
    args = parse_arguments()
    if args.command == "check":
        return run_check(args)
    try:
        return run_supervisor(args)
    except SupervisorAlreadyRunning as error:
        print(str(error), flush=True)
        # Ce n'est pas une panne : l'instance deja active reste responsable.
        return 0
    except KeyboardInterrupt:
        print(
            "Superviseur ferme; le trainer deja lance n'est pas interrompu.",
            flush=True,
        )
        return 130
    except Exception as error:
        run_dir = RUNS_DIR / getattr(args, "run_name", DEFAULT_RUN_NAME)
        status_path = run_dir / "supervisor_status.json"
        status = read_json(status_path)
        status.update({
            "schema_version": 1,
            "run_name": getattr(args, "run_name", DEFAULT_RUN_NAME),
            "supervisor_pid": os.getpid(),
            "state": "failed",
            "message": "Le superviseur a rencontre une erreur inattendue",
            "heartbeat_at_utc": utc_now(),
            "error": f"{type(error).__name__}: {error}",
        })
        try:
            write_json(status_path, status)
        except Exception:
            pass
        print(f"ERREUR SUPERVISEUR : {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        # Le Planificateur de taches interprete ce code comme un echec et
        # redemarre cette couche externe apres son delai d'une minute.
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
