"""
AI Model Optimization Benchmark Suite
Metrics Collection — Latency, Throughput, Memory, CPU, Warmup Time
"""

import json
import time
import threading
import numpy as np
import psutil
import os
from pathlib import Path
from datetime import datetime, timezone

import torch
import torchvision.models as models
import onnxruntime as ort
from openvino import Core

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "resnet18"
INPUT_SHAPE  = (1, 3, 224, 224)
WARMUP_RUNS  = 10
BENCH_RUNS   = 100
ONNX_PATH    = Path("resnet18.onnx")
RESULTS_PATH = Path("metrics_results.json")


# ── System sampler (background thread) ───────────────────────────────────────
class ResourceSampler:
    """Samples CPU % and RSS memory in a background thread during inference."""

    def __init__(self, interval: float = 0.05):
        self._interval = interval
        self._proc     = psutil.Process(os.getpid())
        self._cpu      : list[float] = []
        self._mem_mb   : list[float] = []
        self._running  = False
        self._thread   : threading.Thread | None = None

    def start(self):
        self._cpu.clear()
        self._mem_mb.clear()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        cpu = self._cpu or [0.0]
        mem = self._mem_mb or [0.0]
        return {
            "cpu_mean_pct":  round(float(np.mean(cpu)),  2),
            "cpu_peak_pct":  round(float(np.max(cpu)),   2),
            "mem_mean_mb":   round(float(np.mean(mem)),  2),
            "mem_peak_mb":   round(float(np.max(mem)),   2),
        }

    def _loop(self):
        # prime the CPU counter (first call always returns 0.0)
        self._proc.cpu_percent(interval=None)
        while self._running:
            self._cpu.append(self._proc.cpu_percent(interval=None))
            self._mem_mb.append(self._proc.memory_info().rss / 1e6)
            time.sleep(self._interval)


# ── Metric helpers ────────────────────────────────────────────────────────────
def latency_stats(times_sec: list[float]) -> dict:
    ms = np.array(times_sec) * 1000
    return {
        "avg_latency_ms":    round(float(ms.mean()),              3),
        "median_latency_ms": round(float(np.median(ms)),          3),
        "min_latency_ms":    round(float(ms.min()),               3),
        "max_latency_ms":    round(float(ms.max()),               3),
        "std_latency_ms":    round(float(ms.std()),               3),
        "p95_latency_ms":    round(float(np.percentile(ms, 95)),  3),
        "p99_latency_ms":    round(float(np.percentile(ms, 99)),  3),
    }


def throughput(times_sec: list[float], batch_size: int = 1) -> dict:
    total_samples = len(times_sec) * batch_size
    total_time    = sum(times_sec)
    return {
        "throughput_samples_per_sec": round(total_samples / total_time, 2),
        "throughput_batches_per_sec": round(len(times_sec) / total_time, 2),
    }


def warmup_stats(times_sec: list[float]) -> dict:
    ms = np.array(times_sec) * 1000
    return {
        "warmup_runs":          len(times_sec),
        "warmup_total_ms":      round(float(ms.sum()),   2),
        "warmup_first_ms":      round(float(ms[0]),      2),
        "warmup_last_ms":       round(float(ms[-1]),     2),
        "warmup_stabilized_ms": round(float(ms[-3:].mean()), 2),  # avg of last 3
    }


def make_dummy() -> np.ndarray:
    return np.random.rand(*INPUT_SHAPE).astype(np.float32)


# ── Backend runners ───────────────────────────────────────────────────────────
def collect_pytorch(model: torch.nn.Module, dummy: np.ndarray) -> dict:
    print("\n🔥  PyTorch CPU — collecting metrics …")
    tensor  = torch.from_numpy(dummy)
    sampler = ResourceSampler()

    # warmup
    warmup_times = []
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            t0 = time.perf_counter()
            model(tensor)
            warmup_times.append(time.perf_counter() - t0)

    # benchmark
    sampler.start()
    bench_times = []
    with torch.no_grad():
        for _ in range(BENCH_RUNS):
            t0 = time.perf_counter()
            model(tensor)
            bench_times.append(time.perf_counter() - t0)
    resources = sampler.stop()

    return {
        "backend": "PyTorch CPU",
        **latency_stats(bench_times),
        **throughput(bench_times),
        **warmup_stats(warmup_times),
        **resources,
        "bench_runs": BENCH_RUNS,
    }


def collect_onnxruntime(dummy: np.ndarray) -> dict:
    print("\n⚡  ONNX Runtime — collecting metrics …")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(ONNX_PATH), opts,
                                providers=["CPUExecutionProvider"])
    feed    = {sess.get_inputs()[0].name: dummy}
    sampler = ResourceSampler()

    warmup_times = []
    for _ in range(WARMUP_RUNS):
        t0 = time.perf_counter()
        sess.run(None, feed)
        warmup_times.append(time.perf_counter() - t0)

    sampler.start()
    bench_times = []
    for _ in range(BENCH_RUNS):
        t0 = time.perf_counter()
        sess.run(None, feed)
        bench_times.append(time.perf_counter() - t0)
    resources = sampler.stop()

    return {
        "backend": "ONNX Runtime",
        **latency_stats(bench_times),
        **throughput(bench_times),
        **warmup_stats(warmup_times),
        **resources,
        "bench_runs": BENCH_RUNS,
    }


def collect_openvino(dummy: np.ndarray) -> dict:
    print("\n🔷  OpenVINO CPU — collecting metrics …")
    ie        = Core()
    model_ov  = ie.read_model(str(ONNX_PATH))
    compiled  = ie.compile_model(model_ov, "CPU")
    req       = compiled.create_infer_request()
    sampler   = ResourceSampler()

    warmup_times = []
    for _ in range(WARMUP_RUNS):
        t0 = time.perf_counter()
        req.infer({"input": dummy})
        warmup_times.append(time.perf_counter() - t0)

    sampler.start()
    bench_times = []
    for _ in range(BENCH_RUNS):
        t0 = time.perf_counter()
        req.infer({"input": dummy})
        bench_times.append(time.perf_counter() - t0)
    resources = sampler.stop()

    return {
        "backend": "OpenVINO CPU",
        **latency_stats(bench_times),
        **throughput(bench_times),
        **warmup_stats(warmup_times),
        **resources,
        "bench_runs": BENCH_RUNS,
    }


# ── Pretty printer ────────────────────────────────────────────────────────────
def print_report(results: list[dict]):
    w = 22
    print("\n" + "=" * 70)
    print("  BENCHMARK METRICS REPORT")
    print("=" * 70)

    for r in results:
        print(f"\n  [{r['backend']}]")
        print(f"  {'─'*50}")

        print(f"  {'Avg Latency':{w}} {r['avg_latency_ms']:>8.2f} ms")
        print(f"  {'Median Latency':{w}} {r['median_latency_ms']:>8.2f} ms")
        print(f"  {'p95 Latency':{w}} {r['p95_latency_ms']:>8.2f} ms")
        print(f"  {'p99 Latency':{w}} {r['p99_latency_ms']:>8.2f} ms")
        print(f"  {'Min / Max':{w}} {r['min_latency_ms']:>6.2f} / {r['max_latency_ms']:.2f} ms")

        print(f"  {'Throughput':{w}} {r['throughput_samples_per_sec']:>8.1f} samples/sec")

        print(f"  {'CPU Mean / Peak':{w}} {r['cpu_mean_pct']:>5.1f}% / {r['cpu_peak_pct']:.1f}%")
        print(f"  {'Memory Mean / Peak':{w}} {r['mem_mean_mb']:>6.1f} / {r['mem_peak_mb']:.1f} MB")

        print(f"  {'Warmup (total)':{w}} {r['warmup_total_ms']:>8.1f} ms  ({r['warmup_runs']} runs)")
        print(f"  {'Warmup first run':{w}} {r['warmup_first_ms']:>8.1f} ms")
        print(f"  {'Warmup stabilized':{w}} {r['warmup_stabilized_ms']:>8.1f} ms  (avg last 3)")

    # speedup table
    baseline = next(r for r in results if r["backend"] == "PyTorch CPU")["avg_latency_ms"]
    print("\n" + "─" * 70)
    print(f"  {'Backend':<22} {'Avg Latency':>12}  {'Speedup':>8}  {'Throughput':>16}")
    print(f"  {'─'*22} {'─'*12}  {'─'*8}  {'─'*16}")
    for r in results:
        sx  = baseline / r["avg_latency_ms"]
        tag = " 🏆" if sx == max(baseline / r2["avg_latency_ms"] for r2 in results) else ""
        print(f"  {r['backend']:<22} {r['avg_latency_ms']:>10.2f}ms  {sx:>7.2f}×  "
              f"{r['throughput_samples_per_sec']:>12.1f}/s{tag}")
    print("=" * 70)


# ── Save ──────────────────────────────────────────────────────────────────────
def save(results: list[dict]):
    payload = {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "model":       MODEL_NAME,
        "input_shape": list(INPUT_SHAPE),
        "warmup_runs": WARMUP_RUNS,
        "bench_runs":  BENCH_RUNS,
        "metrics":     results,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\n💾  Saved → {RESULTS_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  AI Model Optimization Benchmark Suite — Metrics Collection")
    print(f"  Model: {MODEL_NAME}  |  Runs: {BENCH_RUNS}  |  Warmup: {WARMUP_RUNS}")
    print("=" * 70)

    if not ONNX_PATH.exists():
        print(f"\n⚠️  {ONNX_PATH} not found — run 'Benchmark engine.py' first to export it.")
        return

    dummy = make_dummy()
    model = models.resnet18(weights=None)
    model.eval()

    results = [
        collect_pytorch(model, dummy),
        collect_onnxruntime(dummy),
        collect_openvino(dummy),
    ]

    print_report(results)
    save(results)
    print("\n✅  Metrics collection complete.")


if __name__ == "__main__":
    main()