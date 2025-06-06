# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "argparse",
#     "matplotlib",
#     "numpy",
#     "scikit-learn",
#     "umap-learn",
# ]
# ///
import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.manifold import TSNE
# from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import argparse
import pickle as pckl


def visualize_with_labels(vectors, labels, model, method="umap"):

    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30)
    else:  # default to UMAP
        reducer = UMAP()

    embedding = reducer.fit_transform(vectors)
    # Unique string labels
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # Choose a color map (adjust if >10 classes)
    cmap = get_cmap('tab10' if n_classes <= 10 else 'tab20')

    # Map each string label to an index (color)
    label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}

    # Plot with legend
    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        idx = labels == label
        plt.scatter(embedding[idx, 0], embedding[idx, 1], c=[label_to_color[label]], label=label, s=5, alpha=0.5)

    plt.legend(title="Hidden vectors from", loc="best")
    plt.xlabel(f"{method.upper()}-1")
    plt.ylabel(f"{method.upper()}-2")
    plt.title(f"{method.upper()} projection colored by labels for {model}")
    plt.tight_layout()
    plt.savefig(
        f"/home/hippolytepilchen/code/hp_v2/results/analysis/{model}_{method}_projection.png",
        dpi=300,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize embeddings with UMAP, PCA, or t-SNE."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model to visualize.",
        choices=["llama3B", "llama8B", "mistral7B"],
    )
    parser.add_argument(
        "--layer_list",
        type=str,
        default="0,1,16,28",
        help="Comma-separated list of layers to visualize.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    vector_dict = {}
    for i in args.layer_list.split(","):
        filename = f"/home/hippolytepilchen/code/hp_v2/results/analysis/{args.model}_embeds_layer_{i}.pkl"
        with open(filename, "rb") as f:
            data = pckl.load(f)
        # Load your data
        vectors = np.concatenate(data, axis=0)
        vector_dict[i] = vectors

    vectors = np.concatenate([v for k, v in vector_dict.items()], axis=0)
    labels = np.concatenate(
        [["Layer " + str(k)] * v.shape[0] for k, v in vector_dict.items()]
    )  # shape: (N,)

    visualize_with_labels(vectors, labels, args.model, method="umap")
    visualize_with_labels(vectors, labels, args.model, method="pca")
    visualize_with_labels(vectors, labels, args.model, method="tsne")
