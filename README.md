# UKP TU-DA at GermEval 2017: Deep Learning for Aspect Based Sentiment Detection
## GermEval-2017 : Shared Task on Aspect-based Sentiment in Social Media Customer Feedback

This repository contains the sentence and word embeddings we used for our experiments for the shared task GermEval2017 reported in Lee et al., *UKP TU-DA at GermEval 2017: Deep Learning for Aspect Based Sentiment Detection*. 


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

* `embeddings/sent2vec` -- Sent2Vec embeddings (.bin) 
* `embeddings/word2vec` -- Word2Vec embeddings (.txt)


## Requirements

* Sent2Vec [https://github.com/epfml/sent2vec]


## Using the embeddings

For obtaining sentence embeddings from the sent2vec models do:

```
$./fasttext print-sentence-vectors ../path-to-model/model.bin < input-sentences.txt > embedding-vectors.txt
```



