import math
from collections import defaultdict, Counter
from typing import List


def compute_tf(tokenized_sentences, n):
    tf_dict = defaultdict(Counter)
    for idx, tokenized_sentence in enumerate(tokenized_sentences):
        ngrams = zip(*[tokenized_sentence[i:] for i in range(n)])
        ngrams = [' '.join(ngram) for ngram in ngrams]
        ngram_counts = Counter(ngrams)
        total_ngrams = sum(ngram_counts.values())
        for ngram in ngram_counts:
            ngram_counts[ngram] /= total_ngrams
        tf_dict[idx] = ngram_counts
    return tf_dict


def compute_idf(tokenized_corpus: List[List[str]], n: int):
    idf_dict = defaultdict(float)
    total_docs = len(tokenized_corpus)
    ngram_doc_count = Counter()

    for tokenized_sentence in tokenized_corpus:
        ngrams = set(zip(*[tokenized_sentence[i:] for i in range(n)]))
        ngrams = [' '.join(ngram) for ngram in ngrams]
        for ngram in ngrams:
            ngram_doc_count[ngram] += 1

    for ngram, count in ngram_doc_count.items():
        idf_dict[ngram] = math.log(total_docs / count)

    return idf_dict


def compute_tf_idf(
    tokenized_sentences: List[List[str]],
    tokenized_corpus: List[List[str]],
    n: int
):
    tf = compute_tf(tokenized_sentences, n)
    idf = compute_idf(tokenized_corpus, n)
    tf_idf_dict = defaultdict(lambda: defaultdict(float))
    for idx, counter in tf.items():
        for ngram, freq in counter.items():
            tf_idf_dict[idx][ngram] = freq * idf.get(ngram, 0)
    return tf_idf_dict


def cosine_similarity(candidate, reference):
    # Extracting terms from both vectors
    candidate_terms = set(candidate.keys())
    reference_terms = set(reference.keys())
    # Combined terms
    all_terms = candidate_terms.union(reference_terms)
    
    # Create full vectors
    candidate_vector = {term: candidate.get(term, 0.0) for term in all_terms}
    reference_vector = {term: reference.get(term, 0.0) for term in all_terms}
    
    # Compute dot product
    dot_product = sum(candidate_vector[term] * reference_vector[term] for term in all_terms)
    
    # Compute magnitudes
    candidate_magnitude = math.sqrt(sum(value ** 2 for value in candidate_vector.values()))
    reference_magnitude = math.sqrt(sum(value ** 2 for value in reference_vector.values()))
    
    # Avoid division by zero
    if candidate_magnitude == 0 or reference_magnitude == 0:
        return 0.0
    
    # Compute cosine similarity
    return dot_product / (candidate_magnitude * reference_magnitude)


def compute_cider_n(
    tokenized_candidate_caption: List[str],
    tokenized_reference_captions: List[List[str]],
    tokenized_corpus: List[List[str]],
    n: int
):
    candidate_tf_idf_vector = compute_tf_idf([tokenized_candidate_caption], tokenized_corpus, n)
    reference_tf_idf_vectors = compute_tf_idf(tokenized_reference_captions, tokenized_corpus, n)
    cosine_similarity_scores = [cosine_similarity(candidate_tf_idf_vector[0], reference_tf_idf_vector)
                                    for reference_tf_idf_vector in reference_tf_idf_vectors.values()]
    cider_score = sum(cosine_similarity_scores) / len(tokenized_reference_captions)
    return cider_score


def compute_cider(
    tokenized_candidate_caption: List[str],
    tokenized_reference_captions: List[List[str]],
    tokenized_corpus: List[List[str]],
    n: int
):
    cider = 0
    
    for i in range(1, n + 1):
        cider += compute_cider_n(
            tokenized_candidate_caption,
            tokenized_reference_captions,
            tokenized_corpus,
            i
        )

    return cider
