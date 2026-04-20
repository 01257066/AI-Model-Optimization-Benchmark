"""
AI Model Optimization Benchmark Suite
FastAPI Benchmark API — accepts model, batch_size, runtime as parameters
"""

import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Literal

import torch
import torchvision.models as models
import onnxruntime as ort
from openvino import Core
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Config ────────────────────────────────────────────────────────────────────
WARMUP_RUNS = 10
BENCH_RUNS  = 100
ONNX_DIR    = Path(".")          # folder where .onnx files are stored
IMAGE_SIZE  = 224                # ResNet/EfficientNet standard input

SUPPORTED_MODELS = {
    "resnet18":       models.resnet18,
    "resnet50":       models.resnet50,
    "mobilenet_v2":   models.mobilenet_v2,
    "efficientnet_b0": models.efficientnet_b0,
}

SUPPORTED_RUNTIMES = ["pytorch", "onnxruntime", "openvino"]

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Model Optimization Benchmark API",
    description="Benchmark deep learning models across PyTorch, ONNX Runtime, and OpenVINO backends.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────
class BenchmarkRequest(BaseModel):
    model: Literal["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0"] = Field(
        default="resnet18",
        description="Model architecture to benchmark",
        examples=["resnet18"],
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        le=64,
        description="Number of images per inference batch (1–64)",
        examples=[1],
    )
    runtime: Literal["pytorch", "onnxruntime", "openvino"] = Field(
        default="pytorch",
        description="Inference backend to use",
        examples=["onnxruntime"],
    )
    warmup_runs: int = Field(default=WARMUP_RUNS, ge=1, le=50,  description="Warmup iterations before timing")
    bench_runs:  int = Field(default=BENCH_RUNS,  ge=1, le=500, description="Timed benchmark iterations")


class LatencyStats(BaseModel):
    avg_ms:    float
    median_ms: float
    min_ms:    float
    max_ms:    float
    std_ms:    float
    p95_ms:    float
    p99_ms:    float


class WarmupStats(BaseModel):
    runs:           int
    total_ms:       float
    first_ms:       float
    last_ms:        float
    stabilized_ms:  float


class BenchmarkResponse(BaseModel):
    # echo request params
    model:      str
    batch_size: int
    runtime:    str
    # core metrics
    latency:    LatencyStats
    throughput_samples_per_sec: float
    throughput_batches_per_sec: float
    warmup:     WarmupStats
    # metadata
    input_shape: list[int]
    bench_runs:  int
    timestamp:   str


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_input(batch_size: int) -> np.ndarray:
    return np.random.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)


def calc_latency(times: list[float]) -> LatencyStats:
    ms = np.array(times) * 1000
    return LatencyStats(
        avg_ms=    round(float(ms.mean()),             3),
        median_ms= round(float(np.median(ms)),         3),
        min_ms=    round(float(ms.min()),              3),
        max_ms=    round(float(ms.max()),              3),
        std_ms=    round(float(ms.std()),              3),
        p95_ms=    round(float(np.percentile(ms, 95)), 3),
        p99_ms=    round(float(np.percentile(ms, 99)), 3),
    )


def calc_warmup(times: list[float]) -> WarmupStats:
    ms = np.array(times) * 1000
    return WarmupStats(
        runs=          len(times),
        total_ms=      round(float(ms.sum()),         2),
        first_ms=      round(float(ms[0]),            2),
        last_ms=       round(float(ms[-1]),           2),
        stabilized_ms= round(float(ms[-3:].mean()),   2),
    )


def onnx_path_for(model_name: str) -> Path:
    return ONNX_DIR / f"{model_name}.onnx"


def export_to_onnx(model: torch.nn.Module, model_name: str, batch_size: int) -> Path:
    """Export model to ONNX if not already present."""
    path = onnx_path_for(model_name)
    if not path.exists():
        dummy = torch.from_numpy(make_input(batch_size))
        model.eval()
        torch.onnx.export(
            model, dummy, str(path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=18,
        )
    return path


# ── Runtime runners ───────────────────────────────────────────────────────────
def run_pytorch(model_name: str, batch_size: int, warmup_runs: int, bench_runs: int) -> dict:
    model_fn = SUPPORTED_MODELS[model_name]
    model    = model_fn(weights=None)
    model.eval()

    tensor = torch.from_numpy(make_input(batch_size))

    warmup_times = []
    with torch.no_grad():
        for _ in range(warmup_runs):
            t0 = time.perf_counter()
            model(tensor)
            warmup_times.append(time.perf_counter() - t0)

    bench_times = []
    with torch.no_grad():
        for _ in range(bench_runs):
            t0 = time.perf_counter()
            model(tensor)
            bench_times.append(time.perf_counter() - t0)

    return {"warmup_times": warmup_times, "bench_times": bench_times}


def run_onnxruntime(model_name: str, batch_size: int, warmup_runs: int, bench_runs: int) -> dict:
    model_fn = SUPPORTED_MODELS[model_name]
    model    = model_fn(weights=None)
    path     = export_to_onnx(model, model_name, batch_size)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    feed = {sess.get_inputs()[0].name: make_input(batch_size)}

    warmup_times = []
    for _ in range(warmup_runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        warmup_times.append(time.perf_counter() - t0)

    bench_times = []
    for _ in range(bench_runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        bench_times.append(time.perf_counter() - t0)

    return {"warmup_times": warmup_times, "bench_times": bench_times}


def run_openvino(model_name: str, batch_size: int, warmup_runs: int, bench_runs: int) -> dict:
    model_fn = SUPPORTED_MODELS[model_name]
    model    = model_fn(weights=None)
    path     = export_to_onnx(model, model_name, batch_size)

    ie       = Core()
    model_ov = ie.read_model(str(path))
    compiled = ie.compile_model(model_ov, "CPU")
    req      = compiled.create_infer_request()
    dummy    = make_input(batch_size)

    warmup_times = []
    for _ in range(warmup_runs):
        t0 = time.perf_counter()
        req.infer({"input": dummy})
        warmup_times.append(time.perf_counter() - t0)

    bench_times = []
    for _ in range(bench_runs):
        t0 = time.perf_counter()
        req.infer({"input": dummy})
        bench_times.append(time.perf_counter() - t0)

    return {"warmup_times": warmup_times, "bench_times": bench_times}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    """API info and available options."""
    return {
        "name":    "AI Model Optimization Benchmark API",
        "version": "1.0.0",
        "docs":    "/docs",
        "endpoints": {
            "POST /benchmark": "Run a benchmark with model, batch_size, runtime",
            "GET  /models":    "List supported models",
            "GET  /runtimes":  "List supported runtimes",
            "GET  /health":    "Health check",
        },
    }


@app.get("/health", tags=["Info"])
def health():
    """Health check."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/models", tags=["Info"])
def list_models():
    """List all supported model architectures."""
    return {
        "models": list(SUPPORTED_MODELS.keys()),
        "note":   "All models use random weights (inference latency benchmark only)",
    }


@app.get("/runtimes", tags=["Info"])
def list_runtimes():
    """List all supported inference runtimes."""
    return {"runtimes": SUPPORTED_RUNTIMES}


@app.post("/benchmark", response_model=BenchmarkResponse, tags=["Benchmark"])
def run_benchmark(req: BenchmarkRequest):
    """
    Run an inference benchmark.

    - **model**: Architecture to test (resnet18, resnet50, mobilenet_v2, efficientnet_b0)
    - **batch_size**: Images per forward pass (1–64)
    - **runtime**: Backend to use (pytorch, onnxruntime, openvino)
    """
    try:
        runners = {
            "pytorch":     run_pytorch,
            "onnxruntime": run_onnxruntime,
            "openvino":    run_openvino,
        }
        result = runners[req.runtime](
            req.model, req.batch_size, req.warmup_runs, req.bench_runs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")

    bench_times  = result["bench_times"]
    warmup_times = result["warmup_times"]
    total_time   = sum(bench_times)
    total_samples = len(bench_times) * req.batch_size

    return BenchmarkResponse(
        model=      req.model,
        batch_size= req.batch_size,
        runtime=    req.runtime,
        latency=    calc_latency(bench_times),
        throughput_samples_per_sec= round(total_samples / total_time, 2),
        throughput_batches_per_sec= round(len(bench_times) / total_time, 2),
        warmup=     calc_warmup(warmup_times),
        input_shape=[req.batch_size, 3, IMAGE_SIZE, IMAGE_SIZE],
        bench_runs= req.bench_runs,
        timestamp=  datetime.now(timezone.utc).isoformat(),
    )


@app.post("/benchmark/compare", tags=["Benchmark"])
def compare_all_runtimes(
    model:      Literal["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0"] = "resnet18",
    batch_size: int = 1,
):
    """
    Run the same model + batch_size across ALL three runtimes and return a comparison.
    """
    if batch_size < 1 or batch_size > 64:
        raise HTTPException(status_code=422, detail="batch_size must be between 1 and 64")

    comparison = []
    runners = {
        "pytorch":     run_pytorch,
        "onnxruntime": run_onnxruntime,
        "openvino":    run_openvino,
    }

    for runtime_name, runner_fn in runners.items():
        try:
            result      = runner_fn(model, batch_size, WARMUP_RUNS, BENCH_RUNS)
            bench_times = result["bench_times"]
            total_time  = sum(bench_times)
            total_samples = len(bench_times) * batch_size
            lat = calc_latency(bench_times)
            comparison.append({
                "runtime":   runtime_name,
                "avg_latency_ms": lat.avg_ms,
                "p95_latency_ms": lat.p95_ms,
                "throughput_samples_per_sec": round(total_samples / total_time, 2),
                "warmup_first_ms": round(result["warmup_times"][0] * 1000, 2),
            })
        except Exception as e:
            comparison.append({"runtime": runtime_name, "error": str(e)})

    # rank by avg latency
    valid = [r for r in comparison if "avg_latency_ms" in r]
    if valid:
        fastest = min(valid, key=lambda x: x["avg_latency_ms"])
        baseline = next((r for r in valid if r["runtime"] == "pytorch"), valid[0])
        for r in valid:
            r["speedup_vs_pytorch"] = round(baseline["avg_latency_ms"] / r["avg_latency_ms"], 3)
            r["fastest"] = (r["runtime"] == fastest["runtime"])

    return {
        "model":      model,
        "batch_size": batch_size,
        "bench_runs": BENCH_RUNS,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "results":    comparison,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("benchmark_api:app", host="0.0.0.0", port=8000, reload=False)
