import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import os

def visualize_gallery(gallery_path, output_path="gallery_visualization.png", top_n=20):
    if not os.path.exists(gallery_path):
        print(f"Error: Gallery file {gallery_path} not found.")
        return

    print(f"Loading gallery from {gallery_path}...")
    gallery = torch.load(gallery_path)
    
    all_embeddings = []
    labels = []
    
    # To keep the plot readable, we might want to only show top N athletes 
    # or a subset if there are hundreds.
    athlete_names = sorted(list(gallery.keys()))
    if len(athlete_names) > top_n:
        print(f"Gallery has {len(athlete_names)} athletes. Showing only the first {top_n} for clarity.")
        athlete_names = athlete_names[:top_n]
    
    for name in athlete_names:
        embeds = gallery[name]
        for e in embeds:
            # embeds might be torch tensors or numpy arrays
            if torch.is_tensor(e):
                e = e.cpu().numpy()
            all_embeddings.append(e)
            labels.append(name)
            
    if not all_embeddings:
        print("No embeddings found in gallery.")
        return
        
    X = np.array(all_embeddings)
    print(f"Running t-SNE on {X.shape[0]} embeddings...")
    
    # t-SNE reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    X_2d = tsne.fit_transform(X)
    
    # Plotting
    plt.figure(figsize=(12, 10))
    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.get_cmap('tab20', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=label, s=50, alpha=0.7)
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
    plt.title(f"t-SNE Visualization of Athlete Gallery Embeddings\n(Top {len(unique_labels)} athletes)")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery", type=str, default="athlete_gallery.pt")
    parser.add_argument("--output", type=str, default="gallery_visualization.png")
    parser.add_argument("--top-n", type=int, default=30, help="Max number of athletes to show")
    args = parser.parse_args()
    
    visualize_gallery(args.gallery, args.output, args.top_n)
