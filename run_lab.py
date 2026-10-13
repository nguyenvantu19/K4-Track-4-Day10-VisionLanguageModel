"""Run the required experiments, score with original tools, and build a report.

Every model run is isolated in a new Python process to release CUDA memory.
Complete checkpoints and prediction files can be reused after an interruption.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from scripts.labcommon import read_jsonl
from scripts.score import evaluate


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def run(command, log_path):
    print("\nRUN:", " ".join(map(str, command)), flush=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(log_path).open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        returncode = proc.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def complete_predictions(path, split, limit=None):
    path = Path(path)
    if not path.exists():
        return False
    rows = read_jsonl(f"data/splits/{split}.jsonl")
    if limit:
        rows = rows[:limit]
    try:
        predictions = read_jsonl(path)
    except (ValueError, KeyError):
        return False
    ids = [r["image_id"] for r in predictions]
    return len(ids) == len(set(ids)) and set(ids) == {r["image_id"] for r in rows}


def main(args):
    python = sys.executable
    root = Path(args.checkpoints)
    root.mkdir(parents=True, exist_ok=True)
    status_path = Path("work/pipeline_status.json")
    started = time.time()

    def status(stage, **extra):
        write(status_path, {"stage": stage, "started_unix": started,
                           "updated_unix": time.time(), **extra})

    def infer(name, checkpoint=None, split="test", limit=None):
        tag = name.lower() + ("_val" if split == "val" else "")
        out = f"work/{tag}.jsonl"
        if not complete_predictions(out, split, limit):
            status(f"infer {name} {split}")
            command = [python, "-u", "student_lab.py", "infer", "--name", name,
                       "--split", split, "--output", out, "--resume", "--batch-size", args.infer_batch]
            if checkpoint:
                command += ["--checkpoint", checkpoint]
            if limit:
                command += ["--limit", limit]
            run(command, f"work/logs/{tag}_infer.log")
        return out

    def score(name, pred, run_info=None, filename=None):
        command = [python, "scripts/score.py", "--pred", pred, "--name", name,
                   "--out", filename or f"report/evidence/{name.lower()}.json"]
        if run_info:
            for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
                command += ["--meta", f"{key}={run_info[key]}"]
        run(command, f"work/logs/{name.lower()}_score.log")

    def train(name, setup, fraction=1.0):
        checkpoint = root / name.lower()
        run_path = checkpoint / "run.json"
        current = load(run_path) if run_path.exists() else None
        if current and (current["setup"] != setup or current["train_fraction"] != fraction):
            raise ValueError("Existing checkpoint has a different experiment configuration")
        if not current or current["completed_steps"] < current["max_steps"]:
            status(f"train {name}")
            command = [python, "-u", "student_lab.py", "train", "--name", name,
                       "--setup", setup, "--output", checkpoint, "--fraction", fraction,
                       "--batch-size", args.train_batch, "--grad-accum", args.grad_accum]
            if current and (checkpoint / "training_state.pt").exists():
                command += ["--resume", checkpoint]
            run(command, f"work/logs/{name.lower()}_train.log")
        current = load(run_path)
        # Keep compact run metadata/logs in the repository, but no weights.
        write(f"report/runs/{name.lower()}.json", current)
        source_log = checkpoint / "train_log.jsonl"
        Path(f"report/runs/{name.lower()}_train.jsonl").write_text(source_log.read_text(), encoding="utf-8")
        return checkpoint, current

    try:
        status("check data and model")
        import importlib.metadata
        environment = {"python": sys.version}
        for package in ("torch", "torchvision", "transformers", "peft", "accelerate", "bitsandbytes", "torchao"):
            try:
                environment[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                environment[package] = None
        write("report/environment.json", environment)
        run([python, "student_lab.py", "inspect-data"], "work/logs/checks.log")
        run([python, "student_lab.py", "verify-model"], "work/logs/checks.log")
        score("E0", infer("E0"))
        validation = {}
        for setup in ("E1", "E2", "E3"):
            checkpoint, info = train(setup, setup)
            score(setup, infer(setup, checkpoint), info)
            val_pred = infer(setup, checkpoint, "val", args.val_images)
            val_rows = read_jsonl("data/splits/val.jsonl")[:args.val_images]
            raw = {r["image_id"]: r["raw"] for r in read_jsonl(val_pred)}
            result = evaluate(val_rows, raw)
            result["split"] = "val_subset"
            result["n_selected"] = len(val_rows)
            write(f"report/validation/{setup.lower()}.json", result)
            validation[setup] = result["micro"]["f1"]
        best = max(validation, key=validation.get)
        write("report/selection.json", {"best_setup": best, "validation_images": args.val_images,
                                        "selection_metric": "micro_f1", "validation_scores": validation})
        print("Selected E4 baseline from validation:", best, validation, flush=True)
        checkpoint, info = train("E4_data_size_0.1", best, 0.1)
        score("E4_data_size_0.1", infer("E4_data_size_0.1", checkpoint), info)
        info = load(root / best.lower() / "run.json")
        score("E4_data_size_1.0", f"work/{best.lower()}.jsonl", info)
        status("build report")
        run([python, "make_report.py", "--name", args.name], "work/logs/report.log")
        status("complete")
        bundle = Path("work/lab_submission.zip")
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
            for item in ("student_lab.py", "run_lab.py", "make_report.py", "lab_day10.ipynb",
                         "build_notebook.py", "student_checks.py", "data/splits/meta.json",
                         "REPORT.md", "README.md", "GUIDE.md", "RUBRIC.md", "requirements-colab.txt",
                         "pytest.ini", ".gitignore", ".env.example"):
                if Path(item).exists():
                    archive.write(item, item)
            for directory in ("report", "configs", "scripts", "tests"):
                for path in Path(directory).rglob("*"):
                    if path.is_file() and "__pycache__" not in path.parts:
                        archive.write(path, path.as_posix())
            # Raw outputs support inspection/re-scoring; work/ remains ignored by Git.
            for path in list(Path("work").glob("*.jsonl")) + list(Path("work").glob("*.run.json")):
                archive.write(path, path.as_posix())
        print(f"All mandatory experiments completed. Artifact: {bundle.resolve()}", flush=True)
    except Exception as exc:
        status("failed", error=str(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", default="work/checkpoints")
    parser.add_argument("--train-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--infer-batch", type=int, default=4)
    parser.add_argument("--val-images", type=int, default=40)
    parser.add_argument("--name", default="Nguyễn Văn Tứ")
    main(parser.parse_args())
