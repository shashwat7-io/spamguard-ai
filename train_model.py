

import os
import re
import sys
import json
import time
import email
import tarfile
import urllib.request
from html import unescape
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
CSV_PATH = os.path.join(DATA_DIR, "spam_dataset.csv")
PIPELINE_PATH = os.path.join(MODEL_DIR, "spam_classifier_pipeline.joblib")
BASELINE_PIPELINE_PATH = os.path.join(MODEL_DIR, "baseline_nb_pipeline.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


DATASET_URLS = {
    "ham_easy": {
        "url": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
        "archive_name": "20030228_easy_ham.tar.bz2",
        "folder_name": "easy_ham",
        "label": 0  
    },
    "spam_1": {
        "url": "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
        "archive_name": "20030228_spam.tar.bz2",
        "folder_name": "spam",
        "label": 1 
    },
    "spam_2": {
        "url": "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",
        "archive_name": "20050311_spam_2.tar.bz2",
        "folder_name": "spam_2",
        "label": 1  
    }
}


def download_progress_hook(count, block_size, total_size):
    """Callback to print download progress percentage."""
    if total_size > 0:
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r  Downloading... {percent}% ({count * block_size / 1024 / 1024:.2f} MB)")
        sys.stdout.flush()


def download_and_extract_corpus() -> None:
    """
    Downloads SpamAssassin public email corpora archives and extracts raw email files.
    Skips downloading if the archives or target folders already exist.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    for key, info in DATASET_URLS.items():
        archive_path = os.path.join(DATA_DIR, info["archive_name"])
        target_dir = os.path.join(DATA_DIR, info["folder_name"])

        
        if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 10:
            print(f"[+] Dataset '{key}' already extracted in {target_dir}")
            continue

        
        if not os.path.exists(archive_path):
            print(f"[*] Fetching genuine dataset '{key}' from {info['url']}...")
            try:
                headers = {'User-Agent': 'Spam-Email-Detection-Downloader/1.0'}
                req = urllib.request.Request(info["url"], headers=headers)
                with urllib.request.urlopen(req) as response, open(archive_path, 'wb') as out_file:
                    total_size = int(response.info().get('Content-Length', -1))
                    count = 0
                    block_size = 16384
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        count += 1
                        download_progress_hook(count, block_size, total_size)
                print(f"\n[+] Successfully downloaded {info['archive_name']}")
            except Exception as e:
                print(f"\n[-] Error downloading {key} from {info['url']}: {e}")
                raise

        
        print(f"[*] Extracting {info['archive_name']} into {DATA_DIR}...")
        try:
            with tarfile.open(archive_path, "r:bz2") as tar:
                tar.extractall(path=DATA_DIR)
            print(f"[+] Extraction complete for {key}.")
        except Exception as e:
            print(f"[-] Failed extracting {archive_path}: {e}")
            raise


def extract_text_from_email_file(file_path: str) -> str:
    """
    Parses a single raw RFC 822 email file:
    Extracts Subject header, cleans HTML/Plain bodies, handles multipart messages,
    and returns a consolidated raw text string.
    """
    try:
        with open(file_path, "rb") as f:
            raw_bytes = f.read()
    except Exception:
        return ""

    try:
        msg = email.message_from_bytes(raw_bytes)
    except Exception:
        try:
            msg = email.message_from_string(raw_bytes.decode('utf-8', errors='ignore'))
        except Exception:
            return raw_bytes.decode('latin-1', errors='ignore')

    # Extract subject
    subject = msg.get("Subject", "") or ""

    # Extract body content
    body_parts = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdisp = str(part.get("Content-Disposition", ""))
            # Ignore binary attachments
            if "attachment" in cdisp:
                continue
            if ctype in ["text/plain", "text/html"]:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        text = payload.decode(charset, errors="ignore")
                    except Exception:
                        text = payload.decode("latin-1", errors="ignore")
                    body_parts.append(text)
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            try:
                body_parts.append(payload.decode(charset, errors="ignore"))
            except Exception:
                body_parts.append(payload.decode("latin-1", errors="ignore"))
        else:
            raw_text = msg.get_payload()
            if isinstance(raw_text, str):
                body_parts.append(raw_text)

    body = "\n".join(body_parts)
    full_email = f"Subject: {subject}\n\n{body}".strip()

    
    if not full_email or len(full_email) < 10:
        full_email = raw_bytes.decode("latin-1", errors="ignore")

    return full_email


def clean_email_text(text: str) -> str:
    """
    NLP Text Cleaning & Preprocessing Routine:
    1. Unescapes HTML entities (&amp;, &lt;, etc.)
    2. Strips HTML markup tags (<p>, <a href=...>)
    3. Normalizes URLs to 'httpaddr' token
    4. Normalizes email addresses to 'emailaddr' token
    5. Normalizes currency and dollar amounts to 'dollar_amount' token
    6. Normalizes generic digits to 'num_token'
    7. Lowercases all characters
    8. Removes excessive punctuation while preserving clean token semantics
    9. Strips multiple consecutive whitespaces
    """
    if not isinstance(text, str):
        return ""

    # Unescape HTML
    text = unescape(text)

    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Normalize web URLs
    text = re.sub(r'http[s]?://[^\s]+', ' httpaddr ', text)
    text = re.sub(r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', ' httpaddr ', text)

    # Normalize email addresses
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', ' emailaddr ', text)

    # Normalize currency/dollars
    text = re.sub(r'[$£€¥]\s*\d+([.,]\d+)?', ' dollar_amount ', text)
    text = re.sub(r'[$£€¥]', ' dollar_symbol ', text)

    # Normalize numbers
    text = re.sub(r'\b\d+\b', ' num_token ', text)

    # Lowercase
    text = text.lower()

    # Remove special characters except token underscores
    text = re.sub(r'[^a-z0-9_\s]', ' ', text)

    # Collapse consecutive spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def build_or_load_dataset() -> pd.DataFrame:
    """
    Loads preprocessed dataset from data/spam_dataset.csv if present.
    Otherwise, downloads raw SpamAssassin corpora, parses all messages,
    cleans the text, and saves to CSV.
    """
    if os.path.exists(CSV_PATH):
        print(f"[+] Found cached processed dataset at {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        df['text'] = df['text'].fillna('')
        return df

    print("[*] Dataset CSV not found. Building dataset from official SpamAssassin corpus...")
    download_and_extract_corpus()

    records = []
    for key, info in DATASET_URLS.items():
        folder_path = os.path.join(DATA_DIR, info["folder_name"])
        label = info["label"]
        if not os.path.isdir(folder_path):
            continue

        files = [f for f in os.listdir(folder_path) if not f.startswith(".")]
        print(f"[*] Processing {len(files)} email files from {info['folder_name']} (Label: {'Spam' if label==1 else 'Ham'})...")

        for fname in files:
            file_path = os.path.join(folder_path, fname)
            raw_text = extract_text_from_email_file(file_path)
            if not raw_text or len(raw_text.strip()) < 15:
                continue

            cleaned = clean_email_text(raw_text)
            if len(cleaned.split()) >= 3:  # Keep meaningful emails
                records.append({
                    "raw_text": raw_text[:2000],  # store reasonable snippet for review
                    "text": cleaned,
                    "label": label,
                    "label_name": "spam" if label == 1 else "ham"
                })

    df = pd.DataFrame(records)
    # Deduplicate
    df = df.drop_duplicates(subset=["text"])
    print(f"[+] Total processed emails: {len(df)}")
    print(f"    - Legitimate (Ham): {sum(df['label'] == 0)}")
    print(f"    - Spam:             {sum(df['label'] == 1)}")

    # Save to disk
    df.to_csv(CSV_PATH, index=False)
    print(f"[+] Saved clean dataset to {CSV_PATH}")
    return df


def train_and_evaluate_model() -> Dict[str, Any]:
    """
    Multi-Model Training & Benchmarking Pipeline:
    1. Loads genuine Apache SpamAssassin dataset.
    2. Performs an 80/20 Stratified train/test split.
    3. Fits TF-IDF vectorizer on training data only to guarantee ZERO data leakage.
    4. Trains 4 candidate classifiers:
       - Multinomial Naive Bayes (Baseline)
       - Logistic Regression (L2 regularized)
       - Random Forest (100 estimators, max_depth=40)
       - Linear SVM (LinearSVC with dual='auto')
    5. Measures Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
    6. Dynamically selects the best-performing model based on test F1-score.
    7. Retains Multinomial Naive Bayes as the benchmark baseline.
    8. Serializes best pipeline and baseline pipeline to disk.
    9. Saves comprehensive evaluation and explainability metrics to JSON.
    """
    start_time = time.time()
    print("=" * 75)
    print("SPAM EMAIL DETECTION - MULTI-MODEL ML BENCHMARK & SELECTION PIPELINE")
    print("=" * 75)

    # 1. Load Dataset
    df = build_or_load_dataset()
    X = df["text"]
    y = df["label"]

    total_samples = len(df)
    ham_count = int(sum(y == 0))
    spam_count = int(sum(y == 1))

    # 2. Stratified Train / Test Split (80% Train, 20% Test)
    # Using stratify=y to ensure identical class proportions and prevent leakage
    print("\n[*] Splitting dataset: 80% Training, 20% Testing (Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"    - Total Corpus: {total_samples} emails (Ham: {ham_count}, Spam: {spam_count})")
    print(f"    - Training Set: {len(X_train)} samples")
    print(f"    - Testing Set:  {len(X_test)} samples")

    # 3. TF-IDF Feature Extraction (Fit ONLY on Training Data)
    print("\n[*] Initializing TF-IDF Vectorizer (15,000 max features, unigrams + bigrams)...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        max_features=15000,
        stop_words='english'
    )

    train_vec_start = time.time()
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    vec_time = round(time.time() - train_vec_start, 3)
    vocab_size = len(tfidf.vocabulary_)
    print(f"[+] TF-IDF Vectorization complete in {vec_time}s. Vocabulary size: {vocab_size} n-grams.")

    # 4. Define Candidate Model Suite
    candidate_models = {
        "Multinomial Naive Bayes": {
            "instance": MultinomialNB(alpha=0.1),
            "description": "Probabilistic Bayesian classifier with Laplace smoothing (alpha=0.1)",
            "family": "Naive Bayes",
            "is_baseline": True
        },
        "Logistic Regression": {
            "instance": LogisticRegression(C=1.0, solver="liblinear", random_state=42, max_iter=1000),
            "description": "L2-regularized linear classifier optimizing log-loss with liblinear solver",
            "family": "Generalized Linear Model",
            "is_baseline": False
        },
        "Random Forest": {
            "instance": RandomForestClassifier(n_estimators=100, max_depth=40, random_state=42, n_jobs=-1),
            "description": "Ensemble of 100 decorrelated decision trees with max depth of 40",
            "family": "Ensemble Trees",
            "is_baseline": False
        },
        "Linear SVM": {
            "instance": LinearSVC(C=1.0, random_state=42, dual='auto', max_iter=2000),
            "description": "Maximum-margin hyperplane classifier using Linear Support Vector Classification",
            "family": "Support Vector Machine",
            "is_baseline": False
        }
    }

    # 5. Train and Evaluate Each Candidate
    print("\n" + "=" * 75)
    print("TRAINING & EVALUATING CANDIDATE MODELS (HELD-OUT TEST PARTITION)")
    print("=" * 75)

    comparison_results = []
    trained_estimators = {}
    reports = {}

    for name, config in candidate_models.items():
        estimator = config["instance"]
        print(f"\n[*] Training {name}...")
        t0 = time.time()
        estimator.fit(X_train_vec, y_train)
        fit_duration = round(time.time() - t0, 4)
        trained_estimators[name] = estimator

        # Predict on held-out test partition
        y_pred = estimator.predict(X_test_vec)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred).tolist()
        tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
        report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"], output_dict=True)
        reports[name] = report

        model_stat = {
            "model_name": name,
            "family": config["family"],
            "description": config["description"],
            "is_baseline": config["is_baseline"],
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "accuracy_percent": round(acc * 100, 2),
            "precision_percent": round(prec * 100, 2),
            "recall_percent": round(rec * 100, 2),
            "f1_percent": round(f1 * 100, 2),
            "fit_time_seconds": fit_duration,
            "confusion_matrix": {
                "matrix": cm,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
                "specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4),
                "sensitivity": round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4)
            }
        }
        comparison_results.append(model_stat)

        print(f"    - Accuracy:  {acc * 100:.2f}%")
        print(f"    - Precision: {prec * 100:.2f}%")
        print(f"    - Recall:    {rec * 100:.2f}%")
        print(f"    - F1-Score:  {f1 * 100:.2f}%")
        print(f"    - Latency:   {fit_duration}s | TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # 6. Dynamic Model Selection based on Measured Test Results
    # Selection criterion: Highest test F1-score (harmonic balance of precision and recall)
    # Secondary criterion: Accuracy
    comparison_results.sort(key=lambda m: (m["f1_score"], m["accuracy"]), reverse=True)
    best_model_name = comparison_results[0]["model_name"]
    baseline_model_name = "Multinomial Naive Bayes"

    for m in comparison_results:
        m["is_best"] = (m["model_name"] == best_model_name)

    print("\n" + "=" * 75)
    print(f"MODEL SELECTION VERDICT: Best Performer is '{best_model_name}'")
    print(f"Baseline Retained:      '{baseline_model_name}'")
    print("=" * 75)

    best_estimator = trained_estimators[best_model_name]
    baseline_estimator = trained_estimators[baseline_model_name]

    # 7. Construct Full Inference Pipelines
    # Scikit-Learn Pipeline combining TF-IDF and Classifier
    best_pipeline = Pipeline([
        ('tfidf', tfidf),
        ('classifier', best_estimator)
    ])

    baseline_pipeline = Pipeline([
        ('tfidf', tfidf),
        ('nb', baseline_estimator)
    ])

    # Serialize Best Pipeline to Disk
    print(f"\n[*] Serializing best model pipeline to {PIPELINE_PATH}...")
    joblib.dump(best_pipeline, PIPELINE_PATH)
    print(f"[+] Serialized best pipeline ({best_model_name}) successfully.")

    # Serialize Baseline Pipeline to Disk
    print(f"[*] Serializing baseline pipeline to {BASELINE_PIPELINE_PATH}...")
    joblib.dump(baseline_pipeline, BASELINE_PIPELINE_PATH)
    print(f"[+] Serialized baseline pipeline ({baseline_model_name}) successfully.")

    # 8. Extract Explainable AI Feature Weights
    feature_names = np.array(tfidf.get_feature_names_out())
    top_spam_keywords = []
    top_ham_keywords = []

    if hasattr(best_estimator, "coef_"):
        # Linear models (Linear SVM, Logistic Regression)
        weights = best_estimator.coef_[0]
        top_spam_indices = np.argsort(weights)[-30:][::-1]
        top_ham_indices = np.argsort(weights)[:30]
        top_spam_keywords = [
            {"word": str(feature_names[i]), "weight": round(float(weights[i]), 3)}
            for i in top_spam_indices
        ]
        top_ham_keywords = [
            {"word": str(feature_names[i]), "weight": round(float(weights[i]), 3)}
            for i in top_ham_indices
        ]
    elif hasattr(best_estimator, "feature_log_prob_"):
        # Naive Bayes log-odds ratio
        log_odds = best_estimator.feature_log_prob_[1] - best_estimator.feature_log_prob_[0]
        top_spam_indices = np.argsort(log_odds)[-30:][::-1]
        top_ham_indices = np.argsort(log_odds)[:30]
        top_spam_keywords = [
            {"word": str(feature_names[i]), "log_ratio": round(float(log_odds[i]), 3)}
            for i in top_spam_indices
        ]
        top_ham_keywords = [
            {"word": str(feature_names[i]), "log_ratio": round(float(log_odds[i]), 3)}
            for i in top_ham_indices
        ]

    # Baseline Naive Bayes log-odds for reference
    nb_log_odds = baseline_estimator.feature_log_prob_[1] - baseline_estimator.feature_log_prob_[0]
    nb_top_spam_indices = np.argsort(nb_log_odds)[-30:][::-1]
    nb_top_ham_indices = np.argsort(nb_log_odds)[:30]
    baseline_spam_keywords = [
        {"word": str(feature_names[i]), "log_ratio": round(float(nb_log_odds[i]), 3)}
        for i in nb_top_spam_indices
    ]
    baseline_ham_keywords = [
        {"word": str(feature_names[i]), "log_ratio": round(float(nb_log_odds[i]), 3)}
        for i in nb_top_ham_indices
    ]

    # Find the stats record for the best model and baseline
    best_stats = next(item for item in comparison_results if item["model_name"] == best_model_name)
    baseline_stats = next(item for item in comparison_results if item["model_name"] == baseline_model_name)

    # 9. Structure Final Evaluation Metrics JSON
    metrics_data = {
        "model_name": f"Spam Email Detection {best_model_name} + TF-IDF",
        "best_model_name": best_model_name,
        "baseline_model_name": baseline_model_name,
        "selection_criterion": "Dynamically selected by highest test F1-score and generalization accuracy",
        "dataset_name": "Apache SpamAssassin Public Email Corpus",
        "total_dataset_samples": total_samples,
        "ham_samples": ham_count,
        "spam_samples": spam_count,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "split_ratio": "80% Training, 20% Testing (Stratified)",
        "vocabulary_size": vocab_size,
        "metrics": {
            "accuracy": best_stats["accuracy"],
            "precision": best_stats["precision"],
            "recall": best_stats["recall"],
            "f1_score": best_stats["f1_score"],
            "accuracy_percent": best_stats["accuracy_percent"],
            "precision_percent": best_stats["precision_percent"],
            "recall_percent": best_stats["recall_percent"],
            "f1_percent": best_stats["f1_percent"]
        },
        "confusion_matrix": best_stats["confusion_matrix"],
        "classification_report": reports[best_model_name],
        "model_comparison": comparison_results,
        "baseline_metrics": {
            "model_name": baseline_model_name,
            "accuracy_percent": baseline_stats["accuracy_percent"],
            "precision_percent": baseline_stats["precision_percent"],
            "recall_percent": baseline_stats["recall_percent"],
            "f1_percent": baseline_stats["f1_percent"],
            "confusion_matrix": baseline_stats["confusion_matrix"]
        },
        "top_spam_keywords": top_spam_keywords if top_spam_keywords else baseline_spam_keywords,
        "top_ham_keywords": top_ham_keywords if top_ham_keywords else baseline_ham_keywords,
        "baseline_top_spam_keywords": baseline_spam_keywords,
        "baseline_top_ham_keywords": baseline_ham_keywords,
        "training_time_seconds": best_stats["fit_time_seconds"],
        "total_pipeline_time_seconds": round(time.time() - start_time, 2),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[+] All empirical metrics & model comparisons saved to {METRICS_PATH}")

    total_duration = round(time.time() - start_time, 2)
    print(f"\n[+] Entire multi-model pipeline completed in {total_duration}s.")
    return metrics_data


if __name__ == "__main__":
    train_and_evaluate_model()
