# Limbic: Author-Based Sentiment Aspect Modeling Regularized with Word Embeddings and Discourse Relations

This repo houses a Java implementation of Limbic. Limbic is an unsupervised probabilistic model that addresses the
problem of discovering aspects and sentiments and associating them with authors
of opinionated texts. Limbic combines three ideas, incorporating authors,
discourse relations, and word embeddings. For discourse relations, Limbic
adopts a generative process regularized by a Markov Random Field. To promote
words with high semantic similarity into the same topic, Limbic captures
semantic regularities from word embeddings via a generalized Pólya Urn
process.

## Data

```
data/cross_validation_hotel.zip.001
data/cross_validation_hotel.zip.001
data/cross_validation_restaurant.zip
```

## Usage

To run the code, specify the following arguments:

Usage: Limbic [options]
```
 -a          Number of aspects
 -a1         Alpha of general
 -a2         Number of lexicon
 -a3         Number of non lexicon
 -b1         Beta of positive
 -b2         Beta of negative
 -g          Gamma
 -i          Number of iterations
 -l          Discourse promotion
 -n          Number of burn-in iterations
 -o          Output model files
 -q          Number of query iterations
 -r          Word promotion
 -t          Thin interval
```

## Citation

Zhe Zhang and Munindar P. Singh. 2018. Limbic: Author-Based Sentiment Aspect Modeling Regularized with Word Embeddings and Discourse Relations.  In <i> Proceedings of the 23<sup>rd</sup> Conference on Empirical Methods in Natural Language Processing (EMNLP)</i>, pages 3412-3422, Brussels. [[pdf]](http://aclweb.org/anthology/D18-1378) [[bib]](https://research.csc.ncsu.edu/mas/code/limbic/Limbic.bib)

## License

[Apache License 2.0](LICENSE)
