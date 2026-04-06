import numpy as np
import pandas as pd
import spacy
import pickle
import os
import time
from collections import Counter
from scipy.sparse import lil_matrix, csr_matrix

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


# def build_vocab_and_vectors(docs):
#     vocab = set()
#     vectors = []
#
#     # Iterating over each document in docs to create a vocabulary of unique words
#     for doc in docs:
#         vocab.update(doc)
#     vocab = sorted(vocab)
#
#     for doc in docs:
#         counter = Counter(doc)  # Creates a dictionary containing the total count of each token in vocab
#         vector = [counter[token] for token in
#                   vocab]  # Creates a non-binary BoW vector for the document using the counter dictionary
#         vectors.append(vector)
#
#     vectors = np.array(vectors)
#
#     return vocab, vectors


# def build_vocab_and_vectors(docs, min_freq=5):
#     total_word_counts = count_tokens(docs)
#     vocab = sorted([word for word in total_word_counts if total_word_counts[word] >= min_freq])

#     vectors = []
#     for doc in docs:
#         counter = Counter(doc)
#         vector = [counter[token] for token in vocab]
#         vectors.append(vector)

#     vectors = np.array(vectors)

#     return vocab, vectors


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
    v = len(vocabulary)
    p_pos = 0
    p_neg = 0

    pos_counts = np.zeros(v)
    neg_counts = np.zeros(v)

    for vec, label in zip(training_vecs, training_labels):
        if label == "positive":
            p_pos += 1
            pos_counts += vec
        else:
            p_neg += 1
            neg_counts += vec

    p_pos /= len(training_vecs)  # (num pos docs) / (num training docs)
    p_neg /= len(training_vecs)  # (num neg docs) / (num training docs)

    # Laplace (Add-1) Smoothing
    pos_probs = (pos_counts + 1) / (
            pos_counts.sum() + v)  # pos_counts.sum() is the total number of words in the pos documents
    neg_probs = (neg_counts + 1) / (
            neg_counts.sum() + v)  # neg_counts.sum() is the total number of words in the neg documents

    return p_pos, p_neg, pos_probs, neg_probs


def classify_naive_bayes(bow_vec, p_pos, p_neg, pos_probs, neg_probs):
    pos_log_score = np.log(p_pos) + np.dot(bow_vec, np.log(pos_probs))
    neg_log_score = np.log(p_neg) + np.dot(bow_vec, np.log(neg_probs))

    max_score = np.maximum(pos_log_score, neg_log_score)  # needs to be np.maximum NOT np.max
    pos_log_score -= max_score
    neg_log_score -= max_score

    exp_pos = np.exp(pos_log_score)
    exp_neg = np.exp(neg_log_score)

    # p_pos_given_doc = np.exp(pos_log_score) / (np.exp(pos_log_score) + np.exp(neg_log_score))
    # p_neg_given_doc = np.exp(neg_log_score) / (np.exp(pos_log_score) + np.exp(neg_log_score))

    p_pos_given_doc = exp_pos / (exp_pos + exp_neg)
    p_neg_given_doc = exp_neg / (exp_pos + exp_neg)

    pred = "positive" if p_pos_given_doc > p_neg_given_doc else "negative"

    return p_pos_given_doc, p_neg_given_doc, pred


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


def normalize(arr):
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    magnitudes = np.linalg.norm(arr, axis=1, keepdims=True)
    magnitudes = np.where(magnitudes == 0, 1, magnitudes)
    return arr / magnitudes


def classify_knn(test_arr, normalized_train, training_labels, k):
    normalized_test = normalize(test_arr)

    all_cosine_similarities = np.dot(normalized_train, normalized_test.T)

    sorted_indices = np.argsort(all_cosine_similarities.T, axis=1)
    top_k = sorted_indices[:, -k:]

    preds = []
    top_k_labels = []

    for neighbors in top_k:
        neighbor_labels = [training_labels[i] for i in neighbors]
        top_k_labels.append(neighbor_labels)

        pred = "positive" if neighbor_labels.count("positive") > neighbor_labels.count("negative") else "negative"
        preds.append(pred)

    return preds, top_k_labels


path = "bumble_google_play_reviews.csv"

if os.path.exists("bow_vectors.npy") and os.path.exists("precomputed.pkl"):
    BoW_vectors = np.load("bow_vectors.npy")
    with open("precomputed.pkl", "rb") as f:
        data = pickle.load(f)
        vocab = data["vocab"]
        reviews = data["reviews"]
        labels = data["labels"]
        tokens = data["tokens"]
    print("Loaded precomputed data from disk.")
else:
    start = time.time()
    reviews, labels = load_and_clean(path)
    print(f"Cleaning: {time.time() - start:.2f}s")

    start = time.time()
    tokens = [tokenize(review) for review in reviews]
    print(f"Tokenization: {time.time() - start:.2f}s")
    print(f"Number of documents: {len(tokens)}")
    all_words = set()
    for doc in tokens:
        all_words.update(doc)
    print(f"Vocabulary size: {len(all_words)}")

    start = time.time()
    vocab, BoW_vectors = build_vocab_and_vectors(tokens)
    print(f"Building vocab and non-binary BoW Vectors: {time.time() - start:.2f}s")

    np.save("bow_vectors.npy", BoW_vectors)
    with open("precomputed.pkl", "wb") as f:
        pickle.dump({
            "vocab": vocab,
            "reviews": reviews,
            "labels": labels,
            "tokens": tokens
        }, f)
    print("Computed and saved data to disk.")

# Splitting the training and testing data
num_samples = len(BoW_vectors)
train_size = 0.6
test_size = 0.2

num_train = np.floor(num_samples * train_size).astype(int)
num_test = np.floor(num_samples * test_size).astype(int)

train_vecs = BoW_vectors[:num_train]
train_labels = labels[:num_train]
test_vecs = BoW_vectors[-num_test:]
test_labels = labels[-num_test:]

# Normalized training data will be using in KNN classification
norm_train = normalize(train_vecs)

pos_train = train_labels.count("positive")
neg_train = train_labels.count("negative")

print(f"Positive Training Samples: {pos_train}")
print(f"Negative Training Samples: {neg_train}\n")
print(f"Vocab length: {len(vocab)}")
