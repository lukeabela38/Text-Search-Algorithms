# SEARCH ENGINE ALGORITHMS
#### Video Demo:  https://youtu.be/8QhMXxcTQY4
#### Description:

Hello there, my name is Luke Abela and I am a 24 year old Electronics and Artificial Intelligence Engineer from the Attard, Island of Malta in Southern Europe. For my project I have implemented the TFIDF and BM25+ information retrieval search algorithms in a python class.

## [TFIDF](https://www.learndatasci.com/glossary/tf-idf-term-frequency-inverse-document-frequency/)

TFIDF - term frequency, inverse document frequency is a numerical statistic which is esigned to reflect the importance of a word to a collection of documents or corpus. It can be used as a weighting factor in information retrieval and text mining.

The TFIDF value increases proportionally to the number of a times a word appears in a document offset by the number of documents in the corpus that contains the world - this helps to adjust for the fact that certain words typically appear very often - words such as "is", "I", "me", "the", and so on.

Term Frequency: Weight of a term that occurs in a document that is simply proportional to the term frequency

Inverse Document Frequency: The specificity of a term can be quantified as an inverse function of the number of documents in which it occurs.

$ tf-idf(t, d) = tf(t, d) * idf(t)

## OKAPI BM25 Plus

A ranking function based on tfidf which is used to estimate the releavance of a document to a given search query. BM25 is a bag of words retrieval function which ranks a set of documents based on the terms from the query appearing in each document, regardless of their proximity within each document.

The BM25 plus specifically addresses a deficiency of the standard BM25 in which the component of the term frequency normalisation by document length is not properly lower bound - as a result  long docs may be unfairly scored to having excess repeated terms.

How to run:

$ python3 project.py corpus.txt "<query>"