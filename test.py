from cider import compute_cider

tokenized_reference_sentences = [['I', 'like', 'nlp', 'I', 'like', 'nlp'], ['I', 'like', 'cars']]

cider_score = compute_cider(["I", "like", "nlp"], tokenized_reference_sentences, tokenized_reference_sentences, 3)
print('score', cider_score)
