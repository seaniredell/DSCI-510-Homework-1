.﻿# Run Cosine Similarity
python3 /path_to/cosine-value-similarity.py \
 --inputDir ../../data/chunks  --outCSV ../../data/cosine_similarity.csv
# Run cosine-cosine-circle-packing
python3 /path_to/cosine-cosine-circle-packing.py \
--inputCSV ../../data/cosine_similarity.csv --cluster 0
# Run cosine-cosine-cluster
python3 /path_to/cosine-cosine-cluster.py \
--inputCSV ../../data/cosine_similarity.csv --cluster 2
# Run generateLevelCluster
python3 /path_to/generateLevelCluster.py
# Move visuals to folder
cp -R /path_to/etllib/html/* ~/Desktop/data
# Open local host
cd ~/Desktop/data
python3 -m http.server 8082
# Look at visuals
http://localhost:8082
# Create a Directory and Save Results with CSV
mkdir -p ~/Desktop/cosine_similarity_output
mv ~/Desktop/data/*.json ~/Desktop/cosine_similarity.csv ~/Desktop/cosine_similarity_output/
