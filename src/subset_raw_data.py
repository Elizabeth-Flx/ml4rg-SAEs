import numpy as np
import os
from typing import Tuple

def load_raw_data() -> Tuple[np.ndarray, np.ndarray]:
    data_dir = "/s/project/ml4rg_students/2025/project02/raw"
    embeddings_filename = "layer_11_embeddings.npy"
    groundtruth_filename = "chip_exo_57_TF_binding_sites.npy"

    embeddings_path = os.path.join(data_dir, embeddings_filename)
    groundtruth_path = os.path.join(data_dir, groundtruth_filename)

    embeddings = np.load(embeddings_path, mmap_mode="r")
    groundtruth = np.load(groundtruth_path, mmap_mode="r")

    return embeddings, groundtruth


def subset_data(
    embeddings: np.ndarray,
    groundtruth: np.ndarray,
    subset_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    n_seqs = embeddings.shape[0]
    n_seqs_subset = int(subset_size * n_seqs)

    idx = np.random.choice(np.arange(n_seqs), n_seqs_subset, replace=False)

    embeddings_subset = embeddings[idx]
    groundtruth_subset = groundtruth[idx]

    return embeddings_subset, groundtruth_subset


if __name__ == "__main__":
    np.random.seed(777)

    embeddings, groundtruth = load_raw_data()

    assert embeddings.shape[:2] == groundtruth.shape[:2]

    subset_size = 0.3
    embeddings_subset, groundtruth_subset = subset_data(
        embeddings, groundtruth, subset_size
    )

    subset_size_pct = int(subset_size * 100)
    group_dir = "/s/project/ml4rg_students/2025/project02/group-1"
    embeddings_subset_path = os.path.join(
        group_dir,
        f"layer_11_embeddings_{subset_size_pct}subset.npy"
    )
    groundtruth_subset_path = os.path.join(
        group_dir,
        f"chip_exo_57_TF_binding_sites_{subset_size_pct}subset.npy"
    )
    np.save(embeddings_subset_path, embeddings_subset)
    np.save(groundtruth_subset_path, groundtruth_subset)