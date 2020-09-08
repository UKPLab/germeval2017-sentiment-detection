#!/bin/bash    
# Download the data
wget https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2483/final_set.zip
unzip final_set.zip -d data/experiments/germ_eval/
rm final_set.zip
# Download the embeddings
wget https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2483/twitter.wiki.germeval.all.100dim.mincount10.postag.vec.gz
wget https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2483/twitter.wiki.germeval.all.100dim.mincount10.vec.gz
mv twitter.wiki.germeval.all.100dim.mincount10.postag.vec.gz data/embeddings/postag_word2vec/
mv twitter.wiki.germeval.all.100dim.mincount10.vec.gz data/embeddings/word2vec/     
