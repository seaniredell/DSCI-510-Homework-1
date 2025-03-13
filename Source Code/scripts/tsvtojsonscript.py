# Ensure you are in the right environment (if your pynev environment was not configured correctly you will not be able to switch between versions)
# pyenv global 2.7.18


# Run CSV to TSV
csvformat -T ../../data/final_output > ../../data/final_output.tsv


# Create encoding.conf file
touch ../../data/conf/encoding.conf


# Create colheaders.conf
head -n 1 ../../data/final_output.tsv | tr '\t' '\n' > ../../data//conf/colheaders.conf
# Create a folder "aggregate-json"
mkdir -p "../../data/aggregate json"
# Run TSV to JSON
tsvtojson -t ../../data/final_output.tsv -j ../../data/aggregate-json/aggregate.json -c ../../data//conf/colheaders.conf -o pixstoryposts -e ../../data/conf/encoding.conf -s 0.8 -v
# Repackage your aggregate JSON file (when executing this step make sure Date Time format does not contain NaN values,
# if it is formatted incorrectly it will still parse with a warning)
cd ./../data/aggregate-json
repackage -j aggregate.json -o pixstoryposts -v