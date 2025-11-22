import statistics
from time import perf_counter

import numpy as np
import psutil

from dense.dense_ops import matrix_generation
from sparse.sparse_ops import sparse_matmul_csr


def used_memory_mb():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def benchmark_dense(method_name, matmul_fun, n, num_runs, block=None):
    times = []
    mem_peak = 0.0
    for _ in range(num_runs):
        A, B, C = matrix_generation(n)
        t0 = perf_counter()
        if block is None:
            matmul_fun(A, B, C)
        else:
            matmul_fun(A, B, C, block=block)
        dt = perf_counter() - t0
        times.append(dt)
        m = used_memory_mb()
        if m > mem_peak:
            mem_peak = m
    mean = statistics.mean(times)
    method_label = method_name if block is None else f"{method_name} (block={block})"
    return {
        "kind": "dense",
        "matrix_size": n,
        "method": method_label,
        "mean": mean,
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        "memory_peak": mem_peak,
        "block": block,
    }


def benchmark_sparse(label, A, B, num_runs):
    times = []
    mem_peak = 0.0
    for _ in range(num_runs):
        t0 = perf_counter()
        _ = sparse_matmul_csr(A, B)
        dt = perf_counter() - t0
        times.append(dt)
        m = used_memory_mb()
        if m > mem_peak:
            mem_peak = m
    mean = statistics.mean(times)
    n = A.shape[0]
    nnz = A.nnz
    total = float(n) * float(n) if n > 0 else 1.0
    sparsity = 1.0 - (nnz / total) if n > 0 else 0.0
    nnz_per_row = nnz / float(n) if n > 0 else 0.0
    return {
        "kind": "sparse",
        "matrix_size": n,
        "method": label,
        "mean": mean,
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        "memory_peak": mem_peak,
        "sparsity": sparsity,
        "nnz_per_row": nnz_per_row,
    }
