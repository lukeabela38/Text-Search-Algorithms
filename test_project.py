import project

corpus = ['My name is Luke and I like dancing.',
'CS50 is awesome.',
'Luke previously studied Electrical and electronics engineering.',
'Luke did a master of science in artificial intelligence.',
'Luke enjoys algorithm development.',
'I did engineering and artificial intelligence.']


def test_index_to_result() -> None:
    for i in range(len(corpus)):
        assert project.index_to_result(corpus, i) == corpus[i]

def test_term_frequencies() -> None:
    for i in range(len(corpus)):
        assert len(project.term_frequencies(corpus[i])) == len(corpus[i].split(" "))

def test_tfidf() -> None:
    query = "dancing"
    assert project.tfidf(corpus, query) == 0.0
    query = "science"
    assert project.tfidf(corpus, query) == 3.0

def test_bm25() -> None:
    query = "dancing"
    assert project.bm25(corpus, query) == 0.0
    query = "science"
    assert project.bm25(corpus, query) == 3.0