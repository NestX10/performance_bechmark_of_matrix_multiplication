import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.io import mmread
import urllib.request
import tarfile
import io
import os

def generate_random_sparse_matrix(n, sparsity=0.9, seed=42):
    np.random.seed(seed)
    density = 1.0 - sparsity
    A = sp.random(n, n, density=density, format="csr", random_state=seed)
    B = sp.random(n, n, density=density, format="csr", random_state=seed + 1)
    return A, B

def sparse_matmul_csr(A, B):
    A_csr = A.tocsr() if not isinstance(A, csr_matrix) else A
    B_csr = B.tocsr() if not isinstance(B, csr_matrix) else B
    return A_csr.dot(B_csr)

def load_or_download_mc2depi(local_mtx_path="mc2depi.mtx"):
    if os.path.exists(local_mtx_path):
        A = mmread(local_mtx_path)
        return csr_matrix(A)
    url = "https://sparse.tamu.edu/MM/Williams/mc2depi.tar.gz"
    try:
        with urllib.request.urlopen(url) as response:
            compressed_data = response.read()
        tar_buf = io.BytesIO(compressed_data)
        with tarfile.open(fileobj=tar_buf, mode="r:gz") as tar:
            mtx_member = None
            for member in tar.getmembers():
                if member.name.endswith(".mtx"):
                    mtx_member = member
                    break
            if mtx_member is None:
                return None
            mtx_file = tar.extractfile(mtx_member)
            A = mmread(mtx_file)
            return csr_matrix(A)
    except Exception:
        return None
