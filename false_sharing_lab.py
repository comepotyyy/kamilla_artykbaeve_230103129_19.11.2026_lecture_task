import os
import sys
import json
import time
import hashlib
import platform
import threading
import subprocess
import numpy as np
from numba import njit, uint64

try:
    import resource
except ImportError:
    resource = None

# Replace with your exact Git handle before executing
GIT_USERNAME = "comepotyyy"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = {"A_contiguous": 1, "B_half_line": 4, "C_line_padded": 8, "D_prefetch_safe": 16}


def detect_cache_line_size() -> int:
    """Query L1 data cache coherency line size in bytes from OS sysfs/sysctl."""
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
        return int(out.strip())
    return 64


def cpu_model_string() -> str:
    """Best-effort CPU model name (used only for the seed)."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], encoding="utf-8").strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


# ==============================================================================
# TASK 1: Hardware Topology & Student Identity Context
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")

    line_size = detect_cache_line_size()
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{logical_cores}"

    # --- Task 1: deterministic 64-bit seed (SHA-256, first 16 hex digits) ---
    cpu_model = cpu_model_string()
    digest = hashlib.sha256(f"{git_user}::{cpu_sig}::{cpu_model}".encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    # --- Task 1: core affinity support ---
    affinity_supported = hasattr(os, "sched_setaffinity")
    if affinity_supported:
        available_cores = sorted(os.sched_getaffinity(0))
    else:
        available_cores = list(range(logical_cores))

    return {
        "git_username": git_user,
        "logical_cores": logical_cores,
        "cache_line_size": line_size,
        "cpu_sig": cpu_sig,
        "cpu_model": cpu_model,
        "instance_seed": instance_seed,
        "affinity_supported": affinity_supported,
        "available_cores": available_cores,
    }


def alloc_aligned(n_elems: int, align: int = 64) -> np.ndarray:
    """uint64 array whose first element sits on a 64-byte boundary."""
    raw = np.zeros(n_elems + align // 8, dtype=np.uint64)
    offset = (-raw.ctypes.data % align) // 8
    buf = raw[offset:offset + n_elems]
    assert buf.ctypes.data % align == 0
    return buf


# ==============================================================================
# TASK 2: GIL-free native counter kernel
# ==============================================================================
@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, seed):
    one = np.uint64(1)
    three = np.uint64(3)
    shift = np.uint64(63)
    for i in range(iters):
        v = buffer[slot]  # real load from memory every iteration
        # Address depends on the loaded value (v >> 63 is always 0 at runtime,
        # but the compiler cannot prove it), so buffer[slot] cannot be
        # promoted into a register across iterations.
        idx = slot + np.int64(v >> shift)
        inc = ((np.uint64(i) ^ seed) & three) + one
        buffer[idx] = v + inc  # real 64-bit store every iteration


# ==============================================================================
# TASK 4: thread-local register privatization baseline
# ==============================================================================
@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, seed):
    one = np.uint64(1)
    three = np.uint64(3)
    acc = np.uint64(0)  # lives in a register
    for i in range(iters):
        acc += ((np.uint64(i) ^ seed) & three) + one
    buffer[slot] = acc  # exactly one shared-memory write


# ==============================================================================
# TASK 3: multi-threaded stride sweep
# ==============================================================================
def run_stride_benchmark(kernel, stride: int, ctx: dict, iters: int = ITERATIONS_PER_THREAD) -> dict:
    buf = alloc_aligned(NUM_THREADS * stride + 16)
    cores = ctx["available_cores"]
    barrier = threading.Barrier(NUM_THREADS + 1)

    def target(tid):
        if ctx["affinity_supported"]:
            os.sched_setaffinity(0, {cores[tid % len(cores)]})  # pins this thread only
        seed = np.uint64((ctx["instance_seed"] + tid) & 0xFFFFFFFFFFFFFFFF)
        barrier.wait()
        kernel(buf, tid * stride, iters, seed)

    threads = [threading.Thread(target=target, args=(t,)) for t in range(NUM_THREADS)]
    for t in threads:
        t.start()

    ru0 = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    barrier.wait()
    t0 = time.perf_counter_ns()
    for t in threads:
        t.join()
    elapsed = time.perf_counter_ns() - t0
    ru1 = resource.getrusage(resource.RUSAGE_SELF) if resource else None

    finals = [int(buf[t * stride]) for t in range(NUM_THREADS)]
    return {
        "stride": stride,
        "elapsed_ns": elapsed,
        "ns_per_update": elapsed / iters,  # wall time / updates per thread
        "vol_ctx_switches": (ru1.ru_nvcsw - ru0.ru_nvcsw) if resource else None,
        "invol_ctx_switches": (ru1.ru_nivcsw - ru0.ru_nivcsw) if resource else None,
        "final_counters": finals,
    }


# ==============================================================================
# TASK 5: parity + LLVM IR audit + manifest
# ==============================================================================
def audit_llvm_and_parity(results: dict, private: dict, ctx: dict) -> dict:
    reference = private["final_counters"]
    parity_ok = all(r["final_counters"] == reference for r in results.values())

    ir_text = "\n".join(worker_strided_accumulate.inspect_llvm().values())
    has_i64_store = "store i64" in ir_text
    no_python_fallback = "PyObject" not in ir_text and "PyEval" not in ir_text

    verification_passed = bool(parity_ok and has_i64_store and no_python_fallback)

    manifest = {
        "git_username": ctx["git_username"],
        "instance_seed": ctx["instance_seed"],
        "cpu_model": ctx["cpu_model"],
        "cpu_sig": ctx["cpu_sig"],
        "logical_cores": ctx["logical_cores"],
        "cache_line_size": ctx["cache_line_size"],
        "threads": NUM_THREADS,
        "iterations_per_thread": ITERATIONS_PER_THREAD,
        "strides": {name: results[name] for name in results},
        "private_baseline": private,
        "slowdown_stride1_vs_stride8": results["A_contiguous"]["elapsed_ns"]
        / results["C_line_padded"]["elapsed_ns"],
        "parity_ok": parity_ok,
        "llvm_has_i64_store": has_i64_store,
        "llvm_no_python_fallback": no_python_fallback,
        "verification_passed": verification_passed,
    }
    with open("cache_lab_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ctx = init_system_context(GIT_USERNAME)
    print(f"Cores: {ctx['logical_cores']} | line size: {ctx['cache_line_size']} B | "
          f"CPU: {ctx['cpu_model']}")

    # Warm-up: compile both kernels so JIT time is not measured
    for k in (worker_strided_accumulate, worker_private_accumulate):
        run_stride_benchmark(k, 8, ctx, iters=1000)

    results = {}
    for name, stride in STRIDES.items():
        results[name] = run_stride_benchmark(worker_strided_accumulate, stride, ctx)
        r = results[name]
        print(f"{name:16s} stride={stride:2d}  {r['elapsed_ns']:>15,d} ns  "
              f"{r['ns_per_update']:.3f} ns/update")

    private = run_stride_benchmark(worker_private_accumulate, 8, ctx)
    print(f"{'private':16s} stride= 8  {private['elapsed_ns']:>15,d} ns  "
          f"{private['ns_per_update']:.3f} ns/update")

    manifest = audit_llvm_and_parity(results, private, ctx)
    print(f"Slowdown stride1/stride8: {manifest['slowdown_stride1_vs_stride8']:.2f}x")
    print(f"verification_passed = {manifest['verification_passed']}")


if __name__ == "__main__":
    main()