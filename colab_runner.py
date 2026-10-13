"""Run inside a Colab cell; download recoverable checkpoints and measured artifacts."""
import json
import subprocess
import sys
import threading
import zipfile
from pathlib import Path


def backup_checkpoint(message):
    directory = Path(message["directory"])
    bundle = Path("work/backups") / f"day10_{message['name'].lower()}_step{message['step']}.zip"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary = bundle.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in directory.rglob("*"):
            if path.is_file():
                archive.write(path, (Path("work/checkpoints") / directory.name / path.relative_to(directory)).as_posix())
    temporary.replace(bundle)
    return bundle


def run_colab(extra_args=(), auto_download=True):
    from google.colab import files
    from run_lab import package_submission
    command = [sys.executable, "-u", "run_lab.py", *map(str, extra_args)]
    Path("work/logs").mkdir(parents=True, exist_ok=True)
    with Path("work/logs/colab_supervisor.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
            if line.startswith("CHECKPOINT_READY "):
                message = json.loads(line.removeprefix("CHECKPOINT_READY "))
                bundle = backup_checkpoint(message)
                print(f"BACKUP_DOWNLOAD {bundle} ({bundle.stat().st_size / 1024**2:.1f} MiB)", flush=True)
                if auto_download:
                    files.download(str(bundle))
            elif "[test] n=" in line:
                bundle = package_submission()
                print("ARTIFACT_DOWNLOAD", bundle, flush=True)
                if auto_download:
                    files.download(str(bundle))
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    if auto_download:
        files.download(str(package_submission()))
    print("PIPELINE_COMPLETE", flush=True)


def start_colab(extra_args=()):
    """Release the notebook kernel so file download comms can run during training."""
    thread = threading.Thread(target=run_colab, args=(extra_args, False), daemon=True,
                              name="day10-lab-supervisor")
    thread.start()
    return thread


_requested_downloads = set()


def poll_colab():
    """Run in a short foreground cell so Colab can process file download messages."""
    from google.colab import files
    status = json.loads(Path("work/pipeline_status.json").read_text())
    print(json.dumps(status, ensure_ascii=False, indent=2))
    log = Path("work/logs/colab_supervisor.log")
    if log.exists():
        print(log.read_text(encoding="utf-8")[-1200:])
    bundles = sorted(Path("work/backups").glob("day10_*.zip"))
    artifact = Path("work/lab_submission.zip")
    if artifact.exists():
        bundles.append(artifact)
    for bundle in bundles:
        stamp = bundle.stat()
        key = (str(bundle), stamp.st_size, stamp.st_mtime_ns)
        if key not in _requested_downloads:
            files.download(str(bundle))
            _requested_downloads.add(key)
            print("DOWNLOAD_REQUESTED", bundle, stamp.st_size, flush=True)
    return status
