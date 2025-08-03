
# iclt_fit.py
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import os

def fit_iclt(x_path, y_path, save_dir, K=4, rank=256, dtype=torch.bfloat16):
    os.makedirs(save_dir, exist_ok=True)

    # Load hidden states
    x = torch.load(x_path).float()  # [N, d]
    y = torch.load(y_path).float()  # [N, d]

    # KMeans clustering
    print("Running KMeans...")
    kmeans = KMeans(n_clusters=K, n_init='auto', random_state=42)
    cluster_ids = kmeans.fit_predict(x.numpy())  # [N]

    for k in range(K):
        x_k = x[cluster_ids == k]
        y_k = y[cluster_ids == k]

        if len(x_k) < rank:
            print(f"[Warning] Cluster {k} has only {len(x_k)} samples, reducing rank.")
            actual_rank = min(rank, len(x_k))
        else:
            actual_rank = rank

        # Least squares: X_k @ T = Y_k
        T_k, _ = torch.lstsq(y_k, x_k)[:2]
        T_k = T_k[:x_k.shape[1]]

        # Low-rank SVD
        U, S, Vh = torch.linalg.svd(T_k, full_matrices=False)
        U_r = U[:, :actual_rank].contiguous()
        V_r = torch.mm(torch.diag(S[:actual_rank]), Vh[:actual_rank, :]).contiguous()

        torch.save(U_r.to(dtype), os.path.join(save_dir, f"U_cluster_{k}.pt"))
        torch.save(V_r.to(dtype), os.path.join(save_dir, f"V_cluster_{k}.pt"))

        print(f"[ICLT] Saved cluster {k}: U {U_r.shape}, V {V_r.shape}")
