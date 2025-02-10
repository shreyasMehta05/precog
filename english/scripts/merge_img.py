from PIL import Image
import os

# Define input and output directories
input_dir = "english/evaluation_results/plots/"
output_dir = "english/evaluation_results/merged_plots/"
os.makedirs(output_dir, exist_ok=True)

# Group plots into three categories
groups = {
    "Correlation_vs_Dimensions.png": [
        "SimLex_Correlation_dims_ppmi.png",
        "SimLex_Correlation_dims_rownormalize.png",
        "SimLex_Correlation_dims_tfidf.png"
    ],
    "Clustering_Silhouette_Trends.png": [
        "Clustering_Silhouette_dims_ppmi.png",
        "Clustering_Silhouette_dims_rownormalize.png",
        "Clustering_Silhouette_dims_tfidf.png",
        "Clustering_Silhouette_window_dim100.png"
    ],
    "WordSim_Correlation_Trends.png": [
        "WordSim_Correlation_dims_ppmi.png",
        "WordSim_Correlation_dims_rownormalize.png",
        "WordSim_Correlation_dims_tfidf.png",
        "WordSim_Correlation_window_dim100.png"
    ]
}

def merge_images(image_list, output_name):
    """Merge images in a 2x2 or 1x3 grid."""
    images = [Image.open(os.path.join(input_dir, img)) for img in image_list]
    
    # Resize all images to the smallest found size
    min_width = min(img.width for img in images)
    min_height = min(img.height for img in images)
    images = [img.resize((min_width, min_height)) for img in images]
    
    # Determine grid size
    if len(images) == 3:
        merged = Image.new("RGB", (min_width * 3, min_height))
        for i, img in enumerate(images):
            merged.paste(img, (i * min_width, 0))
    else:
        merged = Image.new("RGB", (min_width * 2, min_height * 2))
        for i, img in enumerate(images):
            merged.paste(img, ((i % 2) * min_width, (i // 2) * min_height))
    
    merged.save(os.path.join(output_dir, output_name))
    print(f"Saved merged image: {output_name}")

# Merge images for each group
for output_name, image_list in groups.items():
    merge_images(image_list, output_name)

print("All merged plots saved in", output_dir)
