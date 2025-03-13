# Execute via Notebook

import os
import shutil

def split_json_to_files(input_folder, output_base="../../data/chunks", chunk_size=100):
    os.makedirs(output_base, exist_ok=True)
    file_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.json')]
    for i in range(0, len(file_paths), chunk_size):
        chunk_folder = os.path.join(output_base, f"chunk_{i // chunk_size + 1}")
        os.makedirs(chunk_folder, exist_ok=True)
        for file_path in file_paths[i:i + chunk_size]:
            shutil.move(file_path, os.path.join(chunk_folder, os.path.basename(file_path)))

split_json_to_files('../../data/aggregate-json')
