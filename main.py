import os
import numpy as np

from dense.dense_ops import matrix_generation, matmul_original, matmul_cache, matmul_tiling, matmul_numpy
from sparse.sparse_ops import generate_random_sparse_matrix, load_or_download_mc2depi, sparse_matmul_csr
from utils.benchmark import benchmark_dense, benchmark_sparse
from utils.results_saver import save_results
from config import DENSE_SIZES, BLOCK_SIZES, NUM_RUNS_DENSE, SPARSE_SIZES, NUM_RUNS_SPARSE, NUM_RUNS_MC2DEPI, CSV_RESULTS_DIR



def zero_matrix(n):
    return [[0.0 for _ in range(n)] for _ in range(n)]


def matrices_close(M1, M2, tol=1e-9):
    n = len(M1)
    for i in range(n):
        for j in range(n):
            if abs(M1[i][j] - M2[i][j]) > tol:
                return False
    return True


def run_correctness_tests():
    print("=== Correctness tests ===")

    n = 8
    A, B, _ = matrix_generation(n)

    C_orig = zero_matrix(n)
    C_cache = zero_matrix(n)
    C_tile = zero_matrix(n)

    matmul_original(A, B, C_orig)
    matmul_cache(A, B, C_cache)
    matmul_tiling(A, B, C_tile, block=3)

    A_np = np.array(A, dtype=float)
    B_np = np.array(B, dtype=float)
    C_np = (A_np @ B_np).tolist()

    ok_cache = matrices_close(C_orig, C_cache)
    ok_tile = matrices_close(C_orig, C_tile)
    ok_numpy = matrices_close(C_orig, C_np)

    print(f"Dense (n = {n}):")
    print("  original vs cache :", "OK" if ok_cache else "FAIL")
    print("  original vs tiling:", "OK" if ok_tile else "FAIL")
    print("  original vs numpy :", "OK" if ok_numpy else "FAIL")

    n_sparse = 20
    A_s, B_s = generate_random_sparse_matrix(n_sparse, sparsity=0.8)
    C_s = sparse_matmul_csr(A_s, B_s)

    C_s_dense = A_s.toarray() @ B_s.toarray()
    C_s_from_csr = C_s.toarray()

    ok_sparse = np.allclose(C_s_dense, C_s_from_csr, atol=1e-9)

    print(f"Sparse CSR (n = {n_sparse}):", "OK" if ok_sparse else "FAIL")
    print("=========================\n")


def run_dense_part():
    print("=== PART 1: Dense matrix multiplication ===\n")
    dense_results = []

    for n in DENSE_SIZES:
        print(f"Size {n} x {n}")

        r_orig = benchmark_dense("Original", matmul_original, n, NUM_RUNS_DENSE)
        r_orig["size"] = n
        r_orig["name"] = "Original"
        dense_results.append(r_orig)
        print(f"  Original: {r_orig['mean']:.4f} s  mem = {r_orig['memory_peak']:.2f} MB")

        r_cache = benchmark_dense("Cache", matmul_cache, n, NUM_RUNS_DENSE)
        r_cache["size"] = n
        r_cache["name"] = "Cache"
        dense_results.append(r_cache)
        print(f"  Cache   : {r_cache['mean']:.4f} s  mem = {r_cache['memory_peak']:.2f} MB")

        r_numpy = benchmark_dense("NumPy", matmul_numpy, n, NUM_RUNS_DENSE)
        r_numpy["size"] = n
        r_numpy["name"] = "NumPy"
        dense_results.append(r_numpy)
        print(f"  NumPy   : {r_numpy['mean']:.6f} s  mem = {r_numpy['memory_peak']:.2f} MB")

        for b in BLOCK_SIZES:
            r_tile = benchmark_dense("Tiling", matmul_tiling, n, NUM_RUNS_DENSE, block=b)
            r_tile["size"] = n
            r_tile["name"] = f"Tiling (block={b})"
            dense_results.append(r_tile)
            print(f"  Tiling (block = {b:3d}): {r_tile['mean']:.4f} s  mem = {r_tile['memory_peak']:.2f} MB")

        print()

    return dense_results


def run_sparse_part():
    print("=== PART 2: Sparse matrix multiplication (CSR) ===\n")
    sparse_results = []

    for n in SPARSE_SIZES:
        print(f"Size {n} x {n}")
        
        A_rand50, B_rand50 = generate_random_sparse_matrix(n, sparsity=0.5)
        res_50 = benchmark_sparse("Random 50% zeros", A_rand50, B_rand50, NUM_RUNS_SPARSE)
        res_50["size"] = n
        res_50["label"] = "Random 50% zeros"
        sparse_results.append(res_50)
        print(f"  Random (50% zeros): {res_50['mean']:.4f} s  mem = {res_50['memory_peak']:.2f} MB")

        A_rand75, B_rand75 = generate_random_sparse_matrix(n, sparsity=0.75)
        res_75 = benchmark_sparse("Random 75% zeros", A_rand75, B_rand75, NUM_RUNS_SPARSE)
        res_75["size"] = n
        res_75["label"] = "Random 75% zeros"
        sparse_results.append(res_75)
        print(f"  Random (75% zeros): {res_75['mean']:.4f} s  mem = {res_75['memory_peak']:.2f} MB")

        A_rand90, B_rand90 = generate_random_sparse_matrix(n, sparsity=0.9)
        res_90 = benchmark_sparse("Random 90% zeros", A_rand90, B_rand90, NUM_RUNS_SPARSE)
        res_90["size"] = n
        res_90["label"] = "Random 90% zeros"
        sparse_results.append(res_90)
        print(f"  Random (90% zeros): {res_90['mean']:.4f} s  mem = {res_90['memory_peak']:.2f} MB")

        A_rand95, B_rand95 = generate_random_sparse_matrix(n, sparsity=0.95)
        res_95 = benchmark_sparse("Random 95% zeros", A_rand95, B_rand95, NUM_RUNS_SPARSE)
        res_95["size"] = n
        res_95["label"] = "Random 95% zeros"
        sparse_results.append(res_95)
        print(f"  Random (95% zeros): {res_95['mean']:.4f} s  mem = {res_95['memory_peak']:.2f} MB")

        A_rand99, B_rand99 = generate_random_sparse_matrix(n, sparsity=0.99)
        res_99 = benchmark_sparse("Random 99% zeros", A_rand99, B_rand99, NUM_RUNS_SPARSE)
        res_99["size"] = n
        res_99["label"] = "Random 99% zeros"
        sparse_results.append(res_99)
        print(f"  Random (99% zeros): {res_99['mean']:.4f} s  mem = {res_99['memory_peak']:.2f} MB")

    return sparse_results


def run_mc2depi_part():
    mc2depi_results = []
    print("=== PART 3: Williams/mc2depi matrix ===\n")

    mc2 = load_or_download_mc2depi()
    if mc2 is None:
        print("mc2depi not available\n")
        return []

    full_n = mc2.shape[0]
    print(f"Full matrix: {full_n} x {full_n}, nnz = {mc2.nnz}")

    res_full = benchmark_sparse("mc2depi", mc2, mc2, NUM_RUNS_MC2DEPI)
    res_full["size"] = full_n
    res_full["method"] = "mc2depi"
    res_full["submatrix"] = full_n
    res_full["sparsity"] = 1 - (mc2.nnz / (full_n * full_n))
    res_full["nnz"] = mc2.nnz
    res_full["nnz_per_row"] = mc2.nnz / full_n

    mc2depi_results.append(res_full)

    print(f"time = {res_full['mean']:.6f} s  mem = {res_full['memory_peak']:.2f} MB\n")
    return mc2depi_results



def run_benchmark():
    print("======================================================================")
    print("TASK 2: OPTIMIZED MATRIX MULTIPLICATION AND SPARSE MATRICES")
    print("======================================================================")

    run_correctness_tests()
    dense_results = run_dense_part()
    sparse_results = run_sparse_part()
    mc2depi_results = run_mc2depi_part()

    save_results(dense_results, sparse_results, mc2depi_results, CSV_RESULTS_DIR)

    print("=== Task 2 benchmark finished ===")


if __name__ == "__main__":
    run_benchmark()
