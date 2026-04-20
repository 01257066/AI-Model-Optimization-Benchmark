"""
AI Model Optimization Benchmark Suite
Core Engine — ResNet18 across PyTorch CPU, ONNX Runtime, OpenVINO
"""

import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torchvision.models as models
import onnx
import onnxruntime as ort
from openvino import Core

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "resnet18"
INPUT_SHAPE  = (1, 3, 224, 224)          # batch=1, RGB, 224×224
WARMUP_RUNS  = 10
BENCH_RUNS   = 100
ONNX_PATH    = Path("resnet18.onnx")
RESULTS_PATH = Path("benchmark_results.json")


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_dummy_input() -> np.ndarray:
    return np.random.rand(*INPUT_SHAPE).astype(np.float32)


def stats(times: list[float]) -> dict:
    arr = np.array(times) * 1000          # → ms
    return {
        "mean_ms":   round(float(arr.mean()), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "min_ms":    round(float(arr.min()), 3),
        "max_ms":    round(float(arr.max()), 3),
        "std_ms":    round(float(arr.std()), 3),
        "p95_ms":    round(float(np.percentile(arr, 95)), 3),
        "p99_ms":    round(float(np.percentile(arr, 99)), 3),
    }


# ── 1. Load baseline PyTorch model ────────────────────────────────────────────
def load_model() -> torch.nn.Module:
    print("📦  Loading ResNet18 …")
    model = models.resnet18(weights=None)   # random weights — fine for latency benchmarking
    model.eval()
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ── 2. PyTorch CPU inference ───────────────────────────────────────────────────
def bench_pytorch(model: torch.nn.Module, dummy: np.ndarray) -> dict:
    print("\n🔥  Benchmarking PyTorch CPU …")
    tensor = torch.from_numpy(dummy)

    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(tensor)

    times = []
    with torch.no_grad():
        for _ in range(BENCH_RUNS):
            t0 = time.perf_counter()
            out = model(tensor)
            times.append(time.perf_counter() - t0)

    result = {"backend": "PyTorch CPU", **stats(times), "runs": BENCH_RUNS}
    print(f"    mean {result['mean_ms']:.1f} ms  |  p95 {result['p95_ms']:.1f} ms")
    return result


# ── 3. Export to ONNX ─────────────────────────────────────────────────────────
def export_onnx(model: torch.nn.Module, dummy: np.ndarray) -> None:
    print(f"\n💾  Exporting to ONNX → {ONNX_PATH} …")
    tensor = torch.from_numpy(dummy)
    torch.onnx.export(
        model, tensor, str(ONNX_PATH),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    onnx.checker.check_model(str(ONNX_PATH))
    size_mb = ONNX_PATH.stat().st_size / 1e6
    print(f"    ✅  Valid ONNX model — {size_mb:.1f} MB")


# ── 4. ONNX Runtime inference ─────────────────────────────────────────────────
def bench_onnxruntime(dummy: np.ndarray) -> dict:
    print("\n⚡  Benchmarking ONNX Runtime …")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(ONNX_PATH), sess_options,
                                providers=["CPUExecutionProvider"])
    feed = {sess.get_inputs()[0].name: dummy}

    for _ in range(WARMUP_RUNS):
        sess.run(None, feed)

    times = []
    for _ in range(BENCH_RUNS):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append(time.perf_counter() - t0)

    result = {"backend": "ONNX Runtime", **stats(times), "runs": BENCH_RUNS}
    print(f"    mean {result['mean_ms']:.1f} ms  |  p95 {result['p95_ms']:.1f} ms")
    return result


# ── 5. OpenVINO inference ─────────────────────────────────────────────────────
def bench_openvino(dummy: np.ndarray) -> dict:
    print("\n🔷  Benchmarking OpenVINO …")
    ie = Core()
    model_ov = ie.read_model(str(ONNX_PATH))
    compiled  = ie.compile_model(model_ov, "CPU")
    infer_req = compiled.create_infer_request()

    for _ in range(WARMUP_RUNS):
        infer_req.infer({"input": dummy})

    times = []
    for _ in range(BENCH_RUNS):
        t0 = time.perf_counter()
        infer_req.infer({"input": dummy})
        times.append(time.perf_counter() - t0)

    result = {"backend": "OpenVINO CPU", **stats(times), "runs": BENCH_RUNS}
    print(f"    mean {result['mean_ms']:.1f} ms  |  p95 {result['p95_ms']:.1f} ms")
    return result


# ── 6. Compare & summarise ────────────────────────────────────────────────────
def compare(results: list[dict]) -> dict:
    baseline = next(r for r in results if r["backend"] == "PyTorch CPU")["mean_ms"]
    comparison = []
    for r in results:
        speedup = round(baseline / r["mean_ms"], 3)
        comparison.append({**r, "speedup_vs_pytorch": speedup})
        tag = "🏆 fastest" if speedup == max(
            baseline / r2["mean_ms"] for r2 in results) else ""
        print(f"  {r['backend']:20s}  {r['mean_ms']:7.2f} ms  "
              f"({speedup:.2f}×) {tag}")
    return comparison


# ── 7. Save results ───────────────────────────────────────────────────────────
def save_results(comparison: list[dict]) -> None:
    payload = {
        "timestamp":   datetime.utcnow().isoformat() + "Z",
        "model":       MODEL_NAME,
        "input_shape": list(INPUT_SHAPE),
        "warmup_runs": WARMUP_RUNS,
        "bench_runs":  BENCH_RUNS,
        "results":     comparison,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\n💾  Results saved → {RESULTS_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AI Model Optimization Benchmark Suite")
    print(f"  Model: {MODEL_NAME}  |  Runs: {BENCH_RUNS}  |  Warmup: {WARMUP_RUNS}")
    print("=" * 60)

    dummy = make_dummy_input()
    model = load_model()

    pt_result   = bench_pytorch(model, dummy)
    export_onnx(model, dummy)
    ort_result  = bench_onnxruntime(dummy)
    ov_result   = bench_openvino(dummy)

    print("\n📊  Summary (mean latency, speedup vs PyTorch baseline)")
    print("-" * 60)
    comparison = compare([pt_result, ort_result, ov_result])

    save_results(comparison)
    print("\n✅  Benchmark complete.")


if __name__ == "__main__":
    main()