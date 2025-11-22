import numpy as np

def matrix_generation(n):
    A = [[np.random.random() for _ in range(n)] for _ in range(n)]
    B = [[np.random.random() for _ in range(n)] for _ in range(n)]
    C = [[0.0 for _ in range(n)] for _ in range(n)]
    return A, B, C

def matmul_original(A, B, C):
    n = len(A)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s

def matmul_cache(A, B, C):
    n = len(A)
    for i in range(n):
        for k in range(n):
            aik = A[i][k]
            for j in range(n):
                C[i][j] += aik * B[k][j]

def matmul_tiling(A, B, C, block=64):
    n = len(A)
    for ii in range(0, n, block):
        for kk in range(0, n, block):
            for jj in range(0, n, block):
                i_max = min(ii + block, n)
                k_max = min(kk + block, n)
                j_max = min(jj + block, n)
                for i in range(ii, i_max):
                    for k in range(kk, k_max):
                        aik = A[i][k]
                        for j in range(jj, j_max):
                            C[i][j] += aik * B[k][j]

def matmul_numpy(A, B):
    A_np = np.array(A, dtype=float)
    B_np = np.array(B, dtype=float)
    _ = A_np @ B_np
