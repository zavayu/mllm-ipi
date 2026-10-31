"""Run frozen clean/visible baselines on development images; holdout splits stay closed."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import threading
from time import perf_counter

import psutil

from src.baseline_report import write_report
from src.benchmark_data import load_manifest
from src.config import DatasetConfig, EvaluationConfig, OutputConfig, load_config
from src.model_clients import QwenVisionModelClient
from src.provenance import build_manifest, fingerprint, write_json_atomic, file_hash
from src.run_experiment import _client_from_config, run_experiment


class ResourceMonitor:
    """Sample this process's RSS; include model loading in the measured interval."""

    def __enter__(self):
        self.process = psutil.Process()
        self.start = perf_counter()
        cpu = self.process.cpu_times()
        self.start_cpu = cpu.user + cpu.system
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.wait(0.1):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=2)
        memory = self.process.memory_info()
        cpu = self.process.cpu_times()
        self.result = {
            "latest_invocation_wall_seconds": perf_counter() - self.start,
            "latest_invocation_cpu_seconds": cpu.user + cpu.system - self.start_cpu,
            "sampled_peak_rss_bytes": max(self.peak, memory.rss),
            "sampling_interval_seconds": 0.1,
            "os_process_lifetime_peak_working_set_bytes": getattr(
                memory, "peak_wset", None
            ),
            "logical_cpus": psutil.cpu_count(),
            "system_ram_bytes": psutil.virtual_memory().total,
            "platform": platform.platform(),
            "processor": platform.processor(),
        }


def run_baselines(
    config_path: Path,
    output: Path,
    manifest_override: Path | None = None,
    limit: int | None = None,
    allow_cpu=False,
    threads=4,
    client=None,
) -> tuple[Path, dict]:
    if limit is not None and limit < 1:
        raise ValueError("pilot limit must be positive")
    if threads < 1:
        raise ValueError("CPU thread count must be positive")
    config = load_config(config_path)
    manifest_path = manifest_override or config.dataset.manifest
    if manifest_path is None or not manifest_path.is_file():
        raise ValueError(
            "frozen dataset manifest is missing; run python -m src.benchmark_data coco first"
        )
    manifest, images = load_manifest(manifest_path, "development")
    if not images:
        raise ValueError("development split contains no images")
    if limit is not None:
        images = images[:limit]
    image_root = (manifest_path.parent / manifest["image_root"]).resolve()
    config = replace(
        config,
        dataset=DatasetConfig(
            image_root, max_images=limit, manifest=manifest_path, split="development"
        ),
        model=replace(config.model, require_gpu=not allow_cpu),
        evaluation=EvaluationConfig("paper_image_only", repetitions=1),
    )
    if config.model.do_sample:
        raise ValueError("baseline protocol requires deterministic decoding")
    if config.target.adaptive_prefix.enabled:
        raise ValueError("baseline protocol must not use adaptive prefixes")
    if (
        config.rendering.color_strategy != "static"
        or len(config.rendering.font_scales) != 1
        or len(config.rendering.placements) != 1
        or len(config.rendering.colors) != 1
    ):
        raise ValueError(
            "baseline requires a single fixed-placement visible-text configuration"
        )
    selected_hashes = {file_hash(path) for path in images}
    settings = asdict(config)
    settings.pop("output")
    protocol = {
        "schema_version": 1,
        "settings": json.loads(json.dumps(settings, default=str)),
        "config_sha256": file_hash(config_path),
        "dataset_manifest_id": manifest["manifest_id"],
        "dataset_source": manifest["source"],
        "scope": (
            "runtime_pilot"
            if limit or manifest["source"].get("purpose") != "benchmark"
            else "development_baseline"
        ),
        "images": [
            entry
            for entry in manifest["images"]
            if entry["sha256"] in selected_hashes and entry["split"] == "development"
        ],
        "conditions": ["clean", "paper_image_only"],
        "cpu_threads": threads,
        "metric": "case-sensitive exact response after surrounding whitespace trim",
    }
    protocol = {"protocol_id": fingerprint(protocol), **protocol}
    run_dir = output / protocol["protocol_id"][:16]
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = run_dir / "protocol.json"
    if (
        protocol_path.exists()
        and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol
    ):
        raise ValueError("protocol identity collision")
    write_json_atomic(protocol_path, protocol)
    experiment_ids = {}
    print(
        f"scope={protocol['scope']} images={len(images)} protocol={protocol['protocol_id']}",
        flush=True,
    )
    with ResourceMonitor() as resources:
        load_start = perf_counter()
        model_client = _client_from_config(config, client)
        torch = None
        if isinstance(model_client, QwenVisionModelClient):
            import torch

            torch.set_num_threads(threads)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            model_client.provenance()
        load_seconds = perf_counter() - load_start
        for condition in protocol["conditions"]:
            current = replace(
                config,
                evaluation=EvaluationConfig(condition),
                output=OutputConfig(run_dir / "images", run_dir / f"{condition}.jsonl"),
            )
            print(f"running={condition}", flush=True)

            def progress(row):
                print(
                    f"condition={condition} image={Path(row['original_image']).name} status={row['status']} query_seconds={row.get('query_seconds',0):.2f}",
                    flush=True,
                )

            rows = run_experiment(current, client=model_client, on_result=progress)
            if rows:
                experiment_ids[condition] = rows[0]["experiment_id"]
            else:
                experiment_ids[condition] = fingerprint(
                    build_manifest(current, images, model_client)
                )
            print(
                f"condition={condition} new_completed={sum(r['status']=='completed' for r in rows)} errors={sum(r['status']=='error' for r in rows)}",
                flush=True,
            )
    profile = {
        **resources.result,
        "model_load_seconds": load_seconds,
        "cpu_threads": threads,
        "torch_version": str(torch.__version__) if torch is not None else None,
        "gpu_available": torch.cuda.is_available() if torch is not None else None,
        "gpu_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated()
            if torch is not None and torch.cuda.is_available()
            else None
        ),
    }
    report = write_report(run_dir, protocol, experiment_ids, profile)
    if not report["complete"]:
        raise RuntimeError(
            f"baseline has missing image pairs; inspect {run_dir / 'summary.json'} and retry"
        )
    return run_dir, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/milestone2_baseline.yaml")
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results/milestone2"))
    parser.add_argument(
        "--pilot",
        type=int,
        help="Limit development images for runtime estimation; never opens holdouts.",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run_dir, report = run_baselines(
        args.config,
        args.out_dir,
        args.manifest,
        args.pilot,
        args.allow_cpu,
        args.threads,
    )
    print(
        json.dumps(
            {
                "report": str(run_dir / "summary.md"),
                "scope": report["scope"],
                "paired": report["paired"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
