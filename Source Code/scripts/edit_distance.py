# Edit Distance
# Run Edit Distance
python3 /path_to/edit-value-similarity.py \
 --inputDir ../../data/chunks  --outCSV ../../data/jaccard.csv
# Run edit-cosine-circle-packing
python3 /path_to/edit-cosine-circle-packing.py \
--inputCSV ../../data/edit_distance.csv --cluster 0
# Run edit-cosine-cluster
python3 /path_to/edit-cosine-cluster.py \
--inputCSV ../../data/edit_distance.csv --cluster 2
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
mkdir -p ~/Desktop/edit_distance_output
mv ~/Desktop/data/*.json ~/Desktop/edit_distance.csv ~/Desktop/edit_distance_output/