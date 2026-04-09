import numpy as np
import pandas as pd
import spacy
import pickle
import os
import time
import csv
from collections import Counter
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
from datetime import datetime

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


# def build_vocab_and_vectors(docs, min_freq=5):
#     total_word_counts = count_tokens(docs)
#     vocab = sorted([word for word in total_word_counts if total_word_counts[word] >= min_freq])

#     word_to_idx = {word: idx for idx, word in enumerate(vocab)}

#     vectors = lil_matrix((len(docs), len(vocab)), dtype=np.int32)

#     for idx, doc in enumerate(docs):
#         word_counts = Counter(doc)
#         for word, count in word_counts.items():
#             if word in word_to_idx:
#                 vectors[idx, word_to_idx[word]] = count
    
#     sparse_matrix = csr_matrix(vectors)

#     return vocab, sparse_matrix


def build_vocab(training_docs, training_labels, min_freq=5, max_ratio=1.2):
    # Separte training samples by label
    pos_reviews = [review for review, label in zip(training_docs, training_labels) if label == "positive"]
    neg_reviews = [review for review, label in zip(training_docs, training_labels) if label == "negative"]

    num_pos_docs = len(pos_reviews)
    num_neg_docs = len(neg_reviews)

    # Count frequencies
    pos_token_counts = count_tokens(pos_reviews)        # dictionary containing total counts of all pos tokens
    neg_token_counts = count_tokens(neg_reviews)        # dictionary containing total counts of all neg tokens
    total_token_counts = count_tokens(training_docs)    # dictionary containing total counts of every token across all training docs

    valid_tokens = []
    sum_pos_tokens = sum(pos_token_counts.values())
    sum_neg_tokens = sum(neg_token_counts.values())

    # We only want to keep tokens that are discriminative 
    for token, total_freq in total_token_counts.items():
        if total_freq >= min_freq:  # Filters out low frequency words (mostly typos and rare words)

            # Frequency of token in pos/neg documents
            # Safely returns 0 if "token" doesn't appear in pos_review or neg_reviews because of Counter()
            pos_freq = pos_token_counts[token]
            neg_freq = neg_token_counts[token]

            # Probability of token in given class
            p_token_given_pos = pos_freq / sum_pos_tokens 
            p_token_given_neg = neg_freq / sum_neg_tokens

            # Calculate ratio: a ratio from 1 to max_ratio is non-discriminative and will NOT be added to the vocabulary
            max_prob = max(p_token_given_pos, p_token_given_neg)
            min_prob = min(p_token_given_pos, p_token_given_neg) + 1e-9  # prevents division by 0
            ratio = max_prob / min_prob

            if ratio >= max_ratio:
                valid_tokens.append(token)

    vocabulary = sorted(valid_tokens)

    return vocabulary


def vectorize_docs(docs, vocabulary):
    word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}

    # Initialize an empty lil_matrix we can modify using word_to_idx
    vectors = lil_matrix((len(docs), len(vocabulary)), dtype=np.int32)

    for idx, doc in enumerate(docs):
        word_counts = Counter(doc)
        for word, count in word_counts.items():
            if word in word_to_idx:
                vectors[idx, word_to_idx[word]] = count

    sparse_matrix = csr_matrix(vectors)

    return sparse_matrix


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


# def classify_knn(test_csr_matrix, normalized_train_csr_matrix, training_labels, k=3):
#     normalized_test_csr_matrix = normalize(test_csr_matrix)

#     binary_labels = np.where(np.array(training_labels) == "positive", 1, 0)

#     num_samples = normalized_test_csr_matrix.shape[0]

#     knn_preds = []

#     for i in range(num_samples):
#         test_sample = normalized_test_csr_matrix[i]
#         cosine_similarities = normalized_train_csr_matrix.dot(test_sample.T)
#         sorted_indices = np.argsort( cosine_similarities.toarray().flatten() )
#         top_k_indices = sorted_indices[-k:]
#         pred = "positive" if np.sum(binary_labels[top_k_indices]) > k // 2 else "negative"
#         knn_preds.append(pred)

#     return knn_preds


def classify_knn(test_csr_matrix, normalized_train_csr_matrix, training_labels, k=3, batch_size=1000):
    normalized_test_csr_matrix = normalize(test_csr_matrix)
    binary_labels = np.where(np.array(training_labels) == "positive", 1, 0)

    knn_preds = []

    for batch in range(0, normalized_test_csr_matrix.shape[0], batch_size):
        test_batch = normalized_test_csr_matrix[batch:batch+batch_size]

        cosine_similarities = test_batch.dot(normalized_train_csr_matrix.T)  # We want each ROW to represent a test sample
        sorted_indices = np.argsort(cosine_similarities.toarray(), axis=1)
        top_k_indices = sorted_indices[:, -k:]


        for neighbors in top_k_indices:
            pred = "positive" if np.sum(binary_labels[neighbors]) > k // 2 else "negative"
            knn_preds.append(pred)

    return knn_preds


def log_results(algorithm, min_freq, vocab_size, num_train, num_test,
                true_pos, true_neg, false_pos, false_neg, 
                sensitivity, specificity, precision, npv, accuracy, f_score, 
                k=None, runtime=None):
    file_exists = os.path.exists("experiment_log.csv")
    with open("experiment_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["algorithm", "min_freq", "vocab_size", "k",
                             "num_train", "num_test", 
                             "tp", "tn", "fp", "fn",
                             "sensitivity", "specificity", "precision", "nvp", 
                             "acc", "f_score", "runtime"])
        writer.writerow([algorithm, min_freq, vocab_size, k,
                         num_train, num_test, 
                         true_pos, true_neg, false_pos, false_neg, 
                         sensitivity, specificity, precision, npv, 
                         accuracy, f_score, runtime])


path = "bumble_google_play_reviews.csv"

user_text = input("Enter a review: ")

tokens_cache_path = "tokens_cache.pkl"
# bow_cache_path = f"bow_min_freq_{min_freq}.npz"
# vocab_cache_path = f"vocab_min_freq_{min_freq}.pkl"

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

# # === Layer 2: BoW CSR Matrix & Vocabulary ===
# if os.path.exists(bow_cache_path) and os.path.exists(vocab_cache_path):
#     BoW_csr_matrix = load_npz(bow_cache_path)
#     with open(vocab_cache_path, "rb") as f:
#         vocab = pickle.load(f)
#     print(f"Loaded BoW CSR Matrix and Vocabulary (min_freq={min_freq}) from cache.")
# else:
#     start = time.time()
#     vocab, BoW_csr_matrix = build_vocab_and_vectors(tokens, min_freq=min_freq)
#     print(f"Building BoW CSR Matrix and Vocabulary (min_freq={min_freq}): {time.time() - start:.2f}s")

#     save_npz(bow_cache_path, BoW_csr_matrix)
#     with open(vocab_cache_path, "wb") as f:
#         pickle.dump(vocab, f)
#     print(f"Saved BoW CSR Matrix and Vocabulary (min_freq={min_freq}) to cache.")

print(f"Number of documents: {len(tokens)}")

# Splitting the training and testing data
num_samples = len(tokens)
train_size = 0.8
test_size = 0.2

num_train = np.floor(num_samples * train_size).astype(int)
num_test = np.floor(num_samples * test_size).astype(int)

# Training / Testing split
train_tokens = tokens[:num_train]
test_tokens = tokens[-num_test:]

train_labels = labels[:num_train]
test_labels = labels[-num_test:]

# pos_train_count = train_labels.count("positive")
# neg_train_count = train_labels.count("negative")

# print(f"Positive Training Samples: {pos_train_count}")
# print(f"Negative Training Samples: {neg_train_count}\n")

user_tokens = tokenize(user_text)

for i, min_freq in enumerate([2, 10], 1):  # Only using 2 values to test if pipeline works
    print(f"=====================")
    print(f"=== EXPERIMENT {i} ===")
    print(f"=====================\n")

    start = time.time()
    vocab = build_vocab(train_tokens, train_labels, min_freq=min_freq)
    train_csr = vectorize_docs(train_tokens, vocab)
    test_csr = vectorize_docs(test_tokens, vocab)

    # vectorizing user-input for classifying
    user_csr = vectorize_docs([user_tokens], vocab)

    # normalized training data to be used in KNN classification
    norm_train_csr = normalize(train_csr)
    
    print(f"Vocabulary & Vectorization runtime: {time.time() - start:.2f}s")
    print(f"Vocabulary size (min_freq={min_freq}): {len(vocab)}\n")

    # ========================================
    # === Part 1: Naive Bayes Calculations ===
    # ========================================

    start = time.time()
    p_pos, p_neg, pos_probs, neg_probs = train_naive_bayes(train_csr, train_labels, vocab)
    p_pos_given_doc, p_neg_given_doc, NB_preds = classify_naive_bayes(test_csr, p_pos, p_neg, pos_probs, neg_probs)
    nb_runtime = time.time() - start

    tp, tn, fp, fn, sens, spec, prec, npv, acc, f1 = evaluate(NB_preds, test_labels)
    print(f"Naive Bayes Results (min_freq={min_freq}):")
    print(f"Classification time: {nb_runtime:.2f}s")
    print("-------------------------")
    print(f" - True Positive: {tp}")
    print(f" - True Negative: {tn}")
    print(f" - False Positive: {fp}")
    print(f" - False Negative: {fn}")
    print("-------------------------")
    print(f" - Sensitivity: {sens}")
    print(f" - Specificity: {spec}")
    print(f" - Precision: {prec}")
    print(f" - Negative Predictive Value: {npv}")
    print("-------------------------")
    print(f"Accuracy: {acc}")
    print(f"F-Score: {f1}\n")

    log_results("naive_bayes", min_freq=min_freq, vocab_size=len(vocab), 
                num_train=num_train, num_test=num_test, 
                true_pos=tp, true_neg=tn, false_pos=fp, false_neg=fn,
                sensitivity=sens, specificity=spec, precision=prec, npv=npv,
                accuracy=acc, f_score=f1, runtime=nb_runtime)
    
    p_pos_given_doc, p_neg_given_doc, user_NB_pred = classify_naive_bayes(user_csr, p_pos, p_neg, pos_probs, neg_probs)
    print(f"User Naive Bayes (min_freq={min_freq}):")
    print(f" - Probability of Positive Class: {p_pos_given_doc[0]}")
    print(f" - Probability of Negative Class: {p_neg_given_doc[0]}")
    print(f" - User Predicted Class: {user_NB_pred[0]}\n")

    for k in [3, 5]:  # Only using 2 values to test if pipeline works
        
        # ================================
        # === Part 2: KNN Calculations ===
        # ================================

        start = time.time()
        knn_preds = classify_knn(test_csr, norm_train_csr, train_labels, k, batch_size=500)
        knn_runtime = time.time() - start

        tp, tn, fp, fn, sens, spec, prec, npv, acc, f1 = evaluate(knn_preds, test_labels)
        print(f"KNN Results (k={k}):")
        print(f"Classification time: {knn_runtime:.2f}s")
        print("-------------------------")
        print(f" - True Positive: {tp}")
        print(f" - True Negative: {tn}")
        print(f" - False Positive: {fp}")
        print(f" - False Negative: {fn}")
        print("-------------------------")
        print(f" - Sensitivity: {sens}")
        print(f" - Specificity: {spec}")
        print(f" - Precision: {prec}")
        print(f" - Negative Predictive Value: {npv}")
        print("-------------------------")
        print(f"Accuracy: {acc}")
        print(f"F-Score: {f1}\n")

        log_results("knn", min_freq=min_freq, vocab_size=len(vocab), k=k, 
                num_train=num_train, num_test=num_test, 
                true_pos=tp, true_neg=tn, false_pos=fp, false_neg=fn,
                sensitivity=sens, specificity=spec, precision=prec, npv=npv,
                accuracy=acc, f_score=f1, runtime=knn_runtime)
        
        user_knn_pred = classify_knn(user_csr, norm_train_csr, train_labels, k, batch_size=500)
        print(f"User KNN (k={k}):")
        print(f" - User Predicted Class: {user_knn_pred[0]}\n")

# # ========================================
# # === Part 1: Naive Bayes Calculations ===
# # ========================================

# start = time.time()
# p_pos, p_neg, pos_probs, neg_probs = train_naive_bayes(train_csr, train_labels, vocab)
# p_pos_given_doc, p_neg_given_doc, NB_preds = classify_naive_bayes(test_csr, p_pos, p_neg, pos_probs, neg_probs)
# print(f"Naive Bayes training and classification time: {time.time() - start:.2f}s")

# tp, tn, fp, fn, sens, spec, prec, npv, acc, f1 = evaluate(NB_preds, test_labels)
# print("Naive Bayes Results:")
# print("-------------------------")
# print(f" - True Positive: {tp}")
# print(f" - True Negative: {tn}")
# print(f" - False Positive: {fp}")
# print(f" - False Negative: {fn}")
# print("-------------------------")
# print(f" - Sensitivity: {sens}")
# print(f" - Specificity: {spec}")
# print(f" - Precision: {prec}")
# print(f" - Negative Predictive Value: {npv}")
# print("-------------------------")
# print(f"Accuracy: {acc}")
# print(f"F-Score: {f1}\n")


# # ================================
# # === Part 2: KNN Calculations ===
# # ================================

# # k = 3

# for k in [3, 21, 101]:
#     start = time.time()
#     knn_preds = classify_knn(test_csr, norm_train_csr, train_labels, k)
#     print(f"KNN classification time: {time.time() - start:.2f}s")

#     tp, tn, fp, fn, sens, spec, prec, npv, acc, f1 = evaluate(knn_preds, test_labels)
#     print(f"KNN Results k={k}:")
#     print("-------------------------")
#     print(f" - True Positive: {tp}")
#     print(f" - True Negative: {tn}")
#     print(f" - False Positive: {fp}")
#     print(f" - False Negative: {fn}")
#     print("-------------------------")
#     print(f" - Sensitivity: {sens}")
#     print(f" - Specificity: {spec}")
#     print(f" - Precision: {prec}")
#     print(f" - Negative Predictive Value: {npv}")
#     print("-------------------------")
#     print(f"Accuracy: {acc}")
#     print(f"F-Score: {f1}\n")


# # ======================================
# # === Part 3: Classifying User Input ===
# # ======================================

# word_to_idx = {word: idx for idx, word in enumerate(vocab)}

# user_text = input("Enter a review: ")
# user_tokens = tokenize(user_text)

# lil_mat = lil_matrix((1, len(vocab)), dtype=np.int32)

# user_word_counts = Counter(user_tokens)
# for word, count in user_word_counts.items():
#     if word in word_to_idx:
#         lil_mat[0, word_to_idx[word]] = count

# # user_BoW_csr = csr_matrix( np.array([user_tokens.count(token) for token in vocab]) )
# user_BoW_csr = csr_matrix(lil_mat)  # Or lil_mat.tocsr()

# # === Naive Bayes: ===
# p_pos_given_user, p_neg_given_user, user_NB_pred = classify_naive_bayes(user_BoW_csr, p_pos, p_neg, pos_probs, neg_probs)

# print("User Naive Bayes:")
# print("-----------------")
# print(f" - Probability of Positive Class: {p_pos_given_user[0]}")
# print(f" - Probability of Negative Class: {p_neg_given_user[0]}")
# print(f"NB Predicted Class: {user_NB_pred[0]}\n")

# # === KNN: ===

# k = 3
# user_knn_pred = classify_knn(user_BoW_csr, norm_train_csr, train_labels, k)
# knn_pred_val = user_knn_pred[0]

# print("User KNN:")
# print("-----------------")
# print(f"KNN Predicted Class: {knn_pred_val}\n")
