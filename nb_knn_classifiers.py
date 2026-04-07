import numpy as np
import pandas as pd
import spacy
import pickle
import os
import time
from collections import Counter
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz

# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def load_and_clean(path):
    df = pd.read_csv(path)
    df = df[["content", "score"]]  # Grabs only the two columns we're interested in

    df = df.dropna()  # removes all rows with NaN values
    df = df[df["score"] != 3]  # Removes ambiguous 3-star reviews

    # Finds rows where "content" is an empty string and removes them
    df['content'] = df['content'].str.strip()
    df = df[df["content"] != '']

    # Use np.where for vectorized conditional assignment
    df['score'] = np.where(df['score'] >= 4, 'positive', 'negative')

    reviews = df["content"].tolist()
    labels = df["score"].tolist()

    return reviews, labels


def tokenize(text):
    doc = nlp(text)
    lemmas = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    return lemmas


def count_tokens(tokens):
    token_counts = Counter()

    for doc in tokens:
        token_counts.update(doc)

    return token_counts


def build_vocab_and_vectors(docs, min_freq=5):
    total_word_counts = count_tokens(docs)
    vocab = sorted([word for word in total_word_counts if total_word_counts[word] >= min_freq])

    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    vectors = lil_matrix((len(docs), len(vocab)), dtype=np.int32)

    for idx, doc in enumerate(docs):
        word_counts = Counter(doc)
        for word, count in word_counts.items():
            if word in word_to_idx:
                vectors[idx, word_to_idx[word]] = count
    
    sparse_matrix = csr_matrix(vectors)

    return vocab, sparse_matrix


def train_naive_bayes(training_vecs, training_labels, vocabulary):
    training_labels = np.array(training_labels)
    
    pos_mask = training_labels == "positive"
    neg_mask = training_labels == "negative"

    v = len(vocabulary)

    total_training_docs = training_vecs.shape[0]

    p_pos = np.sum(pos_mask) / total_training_docs
    p_neg = np.sum(neg_mask) / total_training_docs

    pos_counts = np.asarray( training_vecs[pos_mask].sum(axis=0) ).flatten()  
    neg_counts = np.asarray( training_vecs[neg_mask].sum(axis=0) ).flatten()

    # Laplace (Add-1) Smoothing
    pos_probs = (pos_counts + 1) / (
            pos_counts.sum() + v)  # pos_counts.sum() is the total number of words in the pos documents
    neg_probs = (neg_counts + 1) / (
            neg_counts.sum() + v)  # neg_counts.sum() is the total number of words in the neg documents

    return p_pos, p_neg, pos_probs, neg_probs


def classify_naive_bayes(test_vectors, p_pos, p_neg, pos_probs, neg_probs):
    pos_log_scores = np.log(p_pos) + test_vectors.dot( np.log(pos_probs) )
    neg_log_scores = np.log(p_neg) + test_vectors.dot( np.log(neg_probs) )

    max_scores = np.maximum(pos_log_scores, neg_log_scores)  # needs to be np.maximum NOT np.max
    pos_log_scores -= max_scores
    neg_log_scores -= max_scores

    pos_exps = np.exp(pos_log_scores)
    neg_exps = np.exp(neg_log_scores)

    # Softmax Calculation
    p_pos_given_doc = pos_exps / (pos_exps + neg_exps)
    p_neg_given_doc = neg_exps / (pos_exps + neg_exps)

    preds = np.where(p_pos_given_doc >= p_neg_given_doc, 'positive', 'negative')

    return p_pos_given_doc, p_neg_given_doc, preds


def evaluate(pred_labels, actual_labels):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for pred, actual in zip(pred_labels, actual_labels):
        if pred == actual and pred == "positive":
            true_pos += 1
        elif pred == actual and pred == "negative":
            true_neg += 1
        elif pred != actual and pred == "positive":
            false_pos += 1
        elif pred != actual and pred == "negative":
            false_neg += 1

    sensitivity = true_pos / (true_pos + false_neg) if (true_pos + false_neg) != 0 else 0  # TP / (TP + FN)
    specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) != 0 else 0  # TN / (TN + FP)
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) != 0 else 0  # TP / (TP + FP)
    neg_pred_value = true_neg / (true_neg + false_neg) if (true_neg + false_neg) != 0 else 0  # TN / (TN + FN)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)  # (TP + TN) / (TP + TN + FP + FN)
    f_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0

    return true_pos, true_neg, false_pos, false_neg, sensitivity, specificity, precision, neg_pred_value, accuracy, f_score


def normalize(csr):
    magnitudes = np.sqrt( csr.multiply(csr).sum(axis=1) )
    magnitudes = np.where(magnitudes == 0, 1, magnitudes)
    
    normalized_csr = csr.multiply( 1 / magnitudes ).tocsr()  # csr.multiply( 1 / magnitutdes ) returns a COO matrix, so it must be converted back to CSR for slicing operations in KNN classification

    return normalized_csr


def classify_knn(test_csr_matrix, normalized_train_csr_matrix, training_labels, k=3):
    normalized_test_csr_matrix = normalize(test_csr_matrix)

    binary_labels = np.where(np.array(training_labels) == "positive", 1, 0)

    num_samples = normalized_test_csr_matrix.shape[0]

    knn_preds = []

    for i in range(num_samples):
        test_sample = normalized_test_csr_matrix[i]
        cosine_similarities = normalized_train_csr_matrix.dot(test_sample.T)
        sorted_indices = np.argsort( cosine_similarities.toarray().flatten() )
        top_k_indices = sorted_indices[-k:]
        pred = "positive" if np.sum(binary_labels[top_k_indices]) > k // 2 else "negative"
        knn_preds.append(pred)

    return knn_preds


path = "bumble_google_play_reviews.csv"

min_freq = 3

tokens_cache_path = "tokens_cache.pkl"
bow_cache_path = f"bow_min_freq_{min_freq}.npz"
vocab_cache_path = f"vocab_min_freq_{min_freq}.pkl"

# === Layer 1: Tokens ===
if os.path.exists(tokens_cache_path):
    with open(tokens_cache_path, "rb") as f:
        data = pickle.load(f)
        reviews = data["reviews"]
        labels = data["labels"]
        tokens = data["tokens"]
    print("Loaded tokens from cache.")
else:
    start = time.time()
    reviews, labels = load_and_clean(path)
    print(f"Cleaning: {time.time() - start:.2f}s")

    start = time.time()
    tokens = [tokenize(review) for review in reviews]
    print(f"Tokenization: {time.time() - start:.2f}s")

    with open(tokens_cache_path, "wb") as f:
        pickle.dump({"reviews": reviews, "labels": labels, "tokens": tokens}, f)
    print("Saved tokens to cache.")

# === Layer 2: BoW CSR Matrix & Vocabulary ===
if os.path.exists(bow_cache_path) and os.path.exists(vocab_cache_path):
    BoW_csr_matrix = load_npz(bow_cache_path)
    with open(vocab_cache_path, "rb") as f:
        vocab = pickle.load(f)
    print(f"Loaded BoW CSR Matrix and Vocabulary (min_freq={min_freq}) from cache.")
else:
    start = time.time()
    vocab, BoW_csr_matrix = build_vocab_and_vectors(tokens, min_freq=min_freq)
    print(f"Building BoW CSR Matrix and Vocabulary (min_freq={min_freq}): {time.time() - start:.2f}s")

    save_npz(bow_cache_path, BoW_csr_matrix)
    with open(vocab_cache_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Saved BoW CSR Matrix and Vocabulary (min_freq={min_freq}) to cache.")

print(f"Number of documents: {len(tokens)}")
print(f"Vocabulary size (min_freq={min_freq}): {len(vocab)}")

# Splitting the training and testing data
num_samples = BoW_csr_matrix.shape[0]
train_size = 0.8
test_size = 0.2

num_train = np.floor(num_samples * train_size).astype(int)
num_test = np.floor(num_samples * test_size).astype(int)

# Samples
train_csr = BoW_csr_matrix[:num_train]
test_csr = BoW_csr_matrix[-num_test:]

# Labels
train_labels = labels[:num_train]
test_labels = labels[-num_test:]

# Normalized training data will be using in KNN classification
norm_train_csr = normalize(train_csr)

pos_train_count = train_labels.count("positive")
neg_train_count = train_labels.count("negative")

print(f"Positive Training Samples: {pos_train_count}")
print(f"Negative Training Samples: {neg_train_count}\n")
print(f"Vocab length: {len(vocab)}")
