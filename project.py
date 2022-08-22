import sys
import math

# Following constants are taken from Apache Lucene implementation
# https://lucene.apache.org/core/8_0_0/core/org/apache/lucene/search/similarities/BM25Similarity.html

BASE = 10
K1 = 1.2
B = 0.75
FREE = 1.0


class SEARCH():
    def __init__(self, corpus: list):
        # initial values for setters/getters
        self.corpus = corpus
        self.length = 0
        self.avgdl = 0
        self.term_frequency = []

    @property
    def corpus(self) -> list:
        return self._corpus

    @corpus.setter
    def corpus(self, corpus: list) -> None:
        self._corpus = self.process(corpus)

    @property
    def length(self) -> int:
        return self._length

    @length.setter
    def length(self, length: int) -> None:
        self._length = len(self._corpus)

    def preprocess(self, sentence: str) -> str:
        return sentence.lower()

    def process(self, corpus: list) -> list:

        for i in range(len(corpus)):
            corpus[i] = self.preprocess(corpus[i])
        return corpus

    @property
    def term_frequency(self) -> list:
        return self._term_frequency

    @term_frequency.setter
    def term_frequency(self, term_frequency: list = []) -> None:
        # create a list of term-frequency dictionaries

        term_frequency = list(map(term_frequencies, self.corpus))
        self._term_frequency = term_frequency

    def search_terms(self, term: str, index: int) -> int:
        try:
            return self._term_frequency[index][term]
        except:
            return 0

    # INVERSE DOCUMENT FREQUENCY

    def term_occurrences(self, term: str) -> int:
        # check in how many documents does the term appear

        df = 0

        for sentence in self.corpus:
            if term in sentence:
                df += 1

        return df

    def idf(self, term: str) -> float:
        length = (1+self.length)
        occurrences = (1+self.term_occurrences(term))
        return math.log((length/occurrences), BASE) + 1

    # BM25

    @property
    def avgdl(self) -> float:
        return self._avgdl

    @avgdl.setter
    def avgdl(self, avgdl: int = 0) -> None:

        avgdl = 0
        for index in range(self.length):
            avgdl += len(self.corpus[index])

        self._avgdl = avgdl/self.length

    # computation

    def bm25(self, term: str, index: int) -> float:

        tf = self.search_terms(term, index)
        nominator = tf * (K1 + 1)

        document_length = len(self.corpus[index])
        denominator = tf + K1*(1 - B + B * document_length/self.avgdl)

        return self.idf(term) * (nominator/denominator + FREE)

    def tfidf(self, term: str, index: int) -> float:
        return self.idf(term) * self.search_terms(term, index)

    # usable functions

    def search_tfidf(self, sentence: str) -> float:

        query: list = self.preprocess(sentence).split()

        scores = []
        for index in range(self.length):

            score: float = 0
            for term in query:
                score += self.tfidf(term, index)
            scores.append(score)

        return scores.index(max(scores))

    def search_bm25(self, sentence: str) -> float:

        query: list = self.preprocess(sentence).split()

        scores: list = []
        for index in range(self.length):
            score: float = 0
            for term in query:
                score += self.bm25(term, index)
            scores.append(score)

        return scores.index(max(scores))


def term_frequencies(sentence: str) -> dict:
    # check how many times in a document any term appear
    counts: dict = dict()
    sent: list = sentence.split()

    unit = 1/len(sent)

    for term in sent:
        if term in counts:
            counts[term] += unit
        else:
            counts[term] = unit

    return counts


def index_to_result(corpus: list, index: int) -> None:
    return corpus[index].strip()


def tfidf(corpus: list, query: str) -> float:
    return SEARCH(corpus).search_tfidf(query)


def bm25(corpus: list, query: str) -> float:
    return SEARCH(corpus).search_bm25(query)


def main():

    if len(sys.argv) != 3:
        print("Required format: python3 bm25.py <text_file_path> \"string\"")
        print(f"Arguments count: {len(sys.argv)}")
        return 1

    with open(sys.argv[1]) as f:
        corpus = f.readlines()

    if len(corpus) < 2:
        print("Corpus too small, please use a corpus of at least 2 lines.")
        return 1

    query = sys.argv[2]

    if len(query) < 1:
        print("Query must be at least one word long")
        return 1

    index = tfidf(corpus, query)
    print("TFIDF Result: ", index_to_result(corpus, index))

    index = bm25(corpus, query)
    print("BM25+ Result: ", index_to_result(corpus, index))

    return 0


if __name__ == "__main__":
    main()
