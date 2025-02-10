import json
from pathlib import Path

# Directory containing JSON files
json_dir = Path("processed_data/hindi_embeddings")

# List of JSON files to combine
json_files = [
    "hindi_reduction_results_10.json",
    "hindi_reduction_results_8.json",
    "hindi_reduction_results_6.json",
    "hindi_reduction_results_4.json",
    "hindi_reduction_results_2.json"
]

# Combined results dictionary
combined_results = {}

# Load and merge all JSON files
for json_file in json_files:
    file_path = json_dir / json_file
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        combined_results.update(data)

# Output file
output_file = json_dir / "hindi_reduction_results.json"

# Save merged JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(combined_results, f, indent=2, ensure_ascii=False)

print(f"Combined JSON saved to {output_file}")
