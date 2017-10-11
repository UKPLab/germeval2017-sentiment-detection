# UKP TU-DA at GermEval 2017: Deep Learning for Aspect Based Sentiment Detection
## GermEval-2017 : Shared Task on Aspect-based Sentiment in Social Media Customer Feedback

This repository contains information about how to use the sentence embeddings we used for our experiments for the [GermEval2017 shared task](https://sites.google.com/view/germeval2017-absa/home) reported in Lee et al., *UKP TU-DA at GermEval 2017: Deep Learning for Aspect Based Sentiment Detection*. 


Please use the following citation:

```
@inproceedings{TUD-CS-2017-0241,
	title = {UKP TU-DA at GermEval 2017: Deep Learning for Aspect Based Sentiment Detection},
	author = {Lee, Ji-Ung and Eger, Steffen and Daxenberger, Johannes and Gurevych, Iryna},
	organization = {German Society for Computational Linguistics},
	booktitle = {Proceedings of the  GSCL GermEval Shared Task on Aspect-based Sentiment in Social Media Customer Feedback},
	pages = {(to appear)},
	month = sep,
	year = {2017},
	location = {Berlin, Germany},
}
```


> **Abstract:** This paper describes our submissions to the GermEval 2017 Shared Task, which focused on the analysis of customer feedback about the Deutsche Bahn AG.
> We used sentence embeddings and an ensemble of classifiers for two sub-tasks as well as state-of-the-art sequence taggers for two other sub-tasks.


Contact persons:

  * Ji-Ung Lee, lee@ukp.informatik.tu-darmstadt.de 
  * Steffen Eger, eger@ukp.informatik.tu-darmstadt.de
  * Johannes Daxenberger, daxenberger@ukp.informatik.tu-darmstadt.de


https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

## Project structure

Due to a big file size the embeddings are not stored in this repository. You can find them here:

* https://public.ukp.informatik.tu-darmstadt.de/GermEval2017_Embeddings/ -- Sent2Vec embeddings (.bin) 

The embeddings were trained on the shared task data, [Wikipedia data](https://sites.google.com/site/rmyeid/projects/polyglot), and Tweets from the [German Sentiment Corpus](https://spinningbytes.com/resources/germansentiment/).

Embedding dimensions are 500, 700, and 1000, as specified in their names.

## Requirements

* [Sent2Vec](https://github.com/epfml/sent2vec)


## Using the embeddings

On Linux you can unpack the embeddings with:

```
$tar --lzma -xvf ../path-to-model/model.bin.tar.lzma
```

For obtaining sentence embeddings from the sent2vec models do:

```
$./fasttext print-sentence-vectors ../path-to-model/model.bin < input-sentences.txt > embedding-vectors.txt
```

## References

### A Twitter Corpus and Benchmark Resources for German Sentiment Analysis. 
Mark Cieliebak, Jan Deriu, Fatih Uzdilli, and Dominic Egger. In “Proceedings of the 4th International Workshop on Natural Language Processing for Social Media (SocialNLP 2017)”, Valencia, Spain, 2017

### Polyglot: Distributed Word Representations for Multilingual NLP
Rami Al-Rfou, Bryan Perozzi, and Steven Skiena. In “Proceedings Seventeenth Conference on Computational Natural Language Learning (CoNLL 2013)”, Sofia, Bulgaria, 2013

