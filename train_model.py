"""
=============================================================================
SPAMGUARD AI - MODEL TRAINING PIPELINE
=============================================================================
Author: SPAMGUARD AI ML Engineering Team
Objective: Download genuine public email corpus (Apache SpamAssassin),
           clean and preprocess raw RFC 822 email texts, extract features using
           TF-IDF Vectorization, train a Multinomial Naive Bayes classifier,
           evaluate with full statistical metrics (Accuracy, Precision, Recall,
           F1-Score, Confusion Matrix), and serialize the trained pipeline with Joblib.
=============================================================================
"""

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

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
CSV_PATH = os.path.join(DATA_DIR, "spam_dataset.csv")
PIPELINE_PATH = os.path.join(MODEL_DIR, "spam_classifier_pipeline.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# Genuine public datasets: Official Apache SpamAssassin public corpus archives
DATASET_URLS = {
    "ham_easy": {
        "url": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
        "archive_name": "20030228_easy_ham.tar.bz2",
        "folder_name": "easy_ham",
        "label": 0  # Ham (Legitimate)
    },
    "spam_1": {
        "url": "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
        "archive_name": "20030228_spam.tar.bz2",
        "folder_name": "spam",
        "label": 1  # Spam
    },
    "spam_2": {
        "url": "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",
        "archive_name": "20050311_spam_2.tar.bz2",
        "folder_name": "spam_2",
        "label": 1  # Spam
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

        # Check if already extracted
        if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 10:
            print(f"[+] Dataset '{key}' already extracted in {target_dir}")
            continue

        # Download if archive doesn't exist
        if not os.path.exists(archive_path):
            print(f"[*] Fetching genuine dataset '{key}' from {info['url']}...")
            try:
                headers = {'User-Agent': 'SpamGuard-AI-Downloader/1.0'}
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

        # Extract archive
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

    # Fallback to decode if parsing produced nothing
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
    Trains the TF-IDF + Multinomial Naive Bayes pipeline without data leakage.
    Evaluates accuracy, precision, recall, F1-score, and confusion matrix.
    Extracts top predictive spam/ham indicators.
    Saves the pipeline using Joblib and metrics to JSON.
    """
    start_time = time.time()
    print("=" * 70)
    print("SPAMGUARD AI - TRAINING PIPELINE INITIALIZATION")
    print("=" * 70)

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

    print(f"    - Training Set: {len(X_train)} samples")
    print(f"    - Testing Set:  {len(X_test)} samples")

    # 3. Construct Scikit-Learn Pipeline
    # TfidfVectorizer:
    # - sublinear_tf=True applies logarithmic term frequency scaling (1 + log(tf))
    # - ngram_range=(1, 2) extracts both unigrams and critical bigrams ("claim prize", "wire transfer")
    # - stop_words='english' filters out common uninformative words
    # - min_df=2 ignores rare single-occurrence typos
    # MultinomialNB:
    # - alpha=0.1 Laplace smoothing preserves sensitivity for distinct spam signals
    print("\n[*] Initializing ML Pipeline (TF-IDF Vectorizer + Multinomial Naive Bayes)...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
            max_features=15000,
            stop_words='english'
        )),
        ('nb', MultinomialNB(alpha=0.1))
    ])

    # 4. Train Model
    print("[*] Training model on training data...")
    train_start = time.time()
    pipeline.fit(X_train, y_train)
    train_duration = round(time.time() - train_start, 3)
    print(f"[+] Model training completed in {train_duration}s.")

    # 5. Evaluate on Held-out Test Set
    print("\n[*] Evaluating on independent test set...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])

    report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"], output_dict=True)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS (HELD-OUT TEST SET)")
    print("=" * 50)
    print(f" Accuracy:  {acc * 100:.2f}%")
    print(f" Precision: {prec * 100:.2f}%")
    print(f" Recall:    {rec * 100:.2f}%")
    print(f" F1-Score:  {f1 * 100:.2f}%")
    print("-" * 50)
    print("Confusion Matrix:")
    print(f"  True Negatives (Legitimate Correct): {tn}")
    print(f"  False Positives (False Alarms):       {fp}")
    print(f"  False Negatives (Missed Spam):        {fn}")
    print(f"  True Positives (Spam Blocked):        {tp}")
    print("=" * 50)

    # 6. Extract Top Informative Features (Explainable AI)
    vectorizer = pipeline.named_steps['tfidf']
    nb = pipeline.named_steps['nb']
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Log-odds ratio between Spam (class 1) and Ham (class 0)
    # log(P(w|Spam) / P(w|Ham)) = log_prob[1] - log_prob[0]
    log_odds = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
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

    # 7. Save Pipeline to disk
    print(f"\n[*] Serializing pipeline to {PIPELINE_PATH}...")
    joblib.dump(pipeline, PIPELINE_PATH)
    print("[+] Pipeline serialized successfully.")

    # 8. Save Metrics & Metadata to JSON
    metrics_data = {
        "model_name": "SpamGuard Multinomial Naive Bayes + TF-IDF",
        "dataset_name": "Apache SpamAssassin Public Email Corpus",
        "total_dataset_samples": total_samples,
        "ham_samples": ham_count,
        "spam_samples": spam_count,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "accuracy_percent": round(acc * 100, 2),
            "precision_percent": round(prec * 100, 2),
            "recall_percent": round(rec * 100, 2),
            "f1_percent": round(f1 * 100, 2)
        },
        "confusion_matrix": {
            "matrix": cm,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        },
        "classification_report": report,
        "top_spam_keywords": top_spam_keywords,
        "top_ham_keywords": top_ham_keywords,
        "training_time_seconds": train_duration,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[+] Metrics & explainability metadata saved to {METRICS_PATH}")

    total_duration = round(time.time() - start_time, 2)
    print(f"\n[+] Entire pipeline completed in {total_duration}s.")
    return metrics_data


if __name__ == "__main__":
    train_and_evaluate_model()
