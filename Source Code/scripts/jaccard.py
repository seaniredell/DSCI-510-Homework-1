﻿# Jaccard Similarity
# Run Jaccard Similarity 
python3 /path_to/jaccard_similarity.py --inputDir ../../data/chunks --outCSV ../../data/jaccard.csv
# Run edit-cosine-circle-packing
python3 /path_to/edit-cosine-circle-packing.py \
 --inputCSV ../../data/jaccard.csv --cluster 2
# Run edit-cosine-cluster
python3 /path_to/edit-cosine-cluster.py \
 --inputCSV ../../data/jaccard.csv --cluster 2
# Run generateLevelCluster
python3 /path_to/generateLevelCluster.py
# Run ETLibb
cp -R /path_to/etllib/html/* ~/Desktop/data
# Run Port to see visuals
cd ~/Desktop/data
python3 -m http.server 8082
# Look at visuals
http://localhost:8082
# Create a Directory and Save Results with CSV
mkdir -p ~/Desktop/jaccard_output
mv ~/Desktop/data/*.json ../../data/jaccard.csv ~/Desktop/jaccard_output/
