# Data Manipulation Utils

## Folder Structure

### gold_analyses_splitter.py
A script that takes a CSV file **full.csv** with ground-truth compound analyses and splits it
into training, validation and test datasets.

### download.sh
Downloads and unpacks the data needed for frequency- or embedding-based 
baselines:
- unigrams from `ruscorpora.ru`,
- bigrams from `ruscorpora.ru`,
- word embeddings from `rusvectores.org`.