cd ../../data/
echo 'Downloading unigram frequencies'
wget https://ruscorpora.ru/new/ngrams/1grams-3.zip
unzip -d ngram_freqs 1grams-3.zip
echo 'Downloading bigram frequencies'
wget https://ruscorpora.ru/new/ngrams/2grams-3.zip
unzip -d ngram_freqs 2grams-3.zip
echo 'Downloading word embeddings'
wget http://vectors.nlpl.eu/repository/20/182.zip
unzip -d embeddings 182.zip
