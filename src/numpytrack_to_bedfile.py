"""
There are two formats we need:
 - .bed for binary data, i.e. groundtruth
 - .bedgraph for continuous data, i.e. SAE activations
"""

import os

import numpy as np


def track_to_bed(
    track: np.ndarray,
    filepath: str,
    track_name: str = "Ground Truth",
    chrom: str = "chr1",
):
    header = f'track name="{track_name}"'
    positions = np.arange(track.shape[0])
    track_positions = positions[track]

    # Taken from:
    # https://stackoverflow.com/questions/71168324/convert-numpy-array-to-a-range
    track_ranges = [
        (a[0], a[-1] + 1)
        for a in np.split(
            track_positions, np.where(np.diff(track_positions) != 1)[0] + 1
        )
    ]

    with open(filepath, "w") as file:
        file.write(f"{header}\n")
        for track_range in track_ranges:
            start, end = track_range
            file.write(f"{chrom}\t{start}\t{end}\n")


def track_to_bedgraph(
    track: np.ndarray,
    filepath: str,
    track_name: str = "SAE Activations",
    chrom: str = "chr1",
):
    header = f'track type=bedGraph name="{track_name}"'
    with open(filepath, "w") as file:
        file.write(f"{header}\n")
        for pos, value in enumerate(track):
            if value > 0:
                start, end = pos, pos + 1
                file.write(f"{chrom}\t{start}\t{end}\t{value}\n")


if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
    SEQ_LEN = 1003

    chrom = "chr1"
    # Create a dummy sequence of specified length
    fasta_path = os.path.join(DATA_DIR, f"{chrom}.fasta")
    with open(fasta_path, "w") as file:
        file.write(f">{chrom}\n")
        file.write("A" * SEQ_LEN)

    continuous_track = np.random.rand(SEQ_LEN)
    idx = np.random.choice(SEQ_LEN, int(0.3 * SEQ_LEN))
    continuous_track[idx] = 0.0
    bedgraph_path = os.path.join(DATA_DIR, f"{chrom}.bedgraph")
    track_to_bedgraph(continuous_track, bedgraph_path)

    binary_track = continuous_track > 0.5
    bed_path = os.path.join(DATA_DIR, f"{chrom}.bed")
    track_to_bed(binary_track, bed_path)
