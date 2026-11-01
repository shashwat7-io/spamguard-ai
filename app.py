"""
=============================================================================
SPAMGUARD AI - FLASK WEB APPLICATION & PREDICTION API
=============================================================================
Author: SPAMGUARD AI Engineering Team
Description: Production-ready Flask server providing a cybersecurity-themed
             web interface and RESTful APIs for real-time email spam and phishing
             detection powered by Scikit-Learn TF-IDF + Multinomial Naive Bayes.
=============================================================================
"""

import os
import re
import json
import time
from html import unescape
from typing import Dict, Any, List

from flask import Flask, request, jsonify, render_template
import joblib

# Application Initialization
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
PIPELINE_PATH = os.path.join(MODEL_DIR, "spam_classifier_pipeline.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# Global variables for cached model pipeline and metrics
pipeline = None
metrics_data = {}

# Known phishing and spam trigger indicators for Explainable AI heuristics
SECURITY_PATTERNS = [
    (r"\b(verify|validate|confirm)\s+(your\s+)?(account|identity|details|password|pin|ssn|wallet)\b", "Account Verification Request"),
    (r"\b(urgent|immediate|action\s+required|critical\s+alert|suspended|terminated|deactivated)\b", "Urgency / Threat Signal"),
    (r"\b(wire\s+transfer|western\s+union|bank\s+transfer|money\s+gram|cryptocurrency|bitcoin)\b", "Financial / Wire Transfer Request"),
    (r"\b(congratulations|you\s+won|lottery|prize|jackpot|selected|winner|sweepstakes)\b", "Lottery / Prize Bait"),
    (r"\b(million\s+dollars|inherit(ance)?|deceased|prince|confidential\s+proposal)\b", "Advance-Fee / Inheritance Scam"),
    (r"\b(click\s+(here|below|this\s+link)|follow\s+this\s+link|download\s+attachment)\b", "Suspicious Call-To-Action (CTA)"),
    (r"\b(claim\s+your|claim\s+now|exclusive\s+deal|free\s+access|100%\s+free|risk\s+free)\b", "Promotional Bait / Unsolicited Offer"),
    (r"\b(refinance|mortgage|lowest\s+rates|pre-approved|bad\s+credit|no\s+obligation)\b", "Financial Solicitation"),
    (r"\b(viagra|cialis|pharmacy|weight\s+loss|pills|medications)\b", "Pharmaceutical Spam"),
    (r"\b(opt\s*out|remove\s+here|unsubscribe|future\s+mailings)\b", "Bulk Mailing / Unsubscribe Notice"),
]


def load_model_and_metrics() -> None:
    """
    Loads serialized ML pipeline and metrics JSON into memory.
    If not yet trained, prints a clear instruction.
    """
    global pipeline, metrics_data

    # 1. Load pipeline
    if os.path.exists(PIPELINE_PATH):
        try:
            pipeline = joblib.load(PIPELINE_PATH)
            print(f"[+] Loaded ML pipeline from {PIPELINE_PATH}")
        except Exception as e:
            print(f"[-] Error loading pipeline: {e}")
            pipeline = None
    else:
        print(f"[!] Pipeline not found at {PIPELINE_PATH}. Please run: python train_model.py")
        pipeline = None

    # 2. Load metrics
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            print(f"[+] Loaded model evaluation metrics from {METRICS_PATH}")
        except Exception as e:
            print(f"[-] Error loading metrics: {e}")
            metrics_data = {}
    else:
        print(f"[!] Metrics not found at {METRICS_PATH}")
        metrics_data = {}


def clean_email_text(text: str) -> str:
    """
    NLP Text Preprocessing Routine (must match training pipeline exactly):
    - Unescape HTML characters
    - Strip HTML markup
    - Normalize URLs to 'httpaddr'
    - Normalize email addresses to 'emailaddr'
    - Normalize currency amounts to 'dollar_amount' / 'dollar_symbol'
    - Normalize digits to 'num_token'
    - Lowercase and collapse whitespace
    """
    if not isinstance(text, str):
        return ""

    text = unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http[s]?://[^\s]+', ' httpaddr ', text)
    text = re.sub(r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', ' httpaddr ', text)
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', ' emailaddr ', text)
    text = re.sub(r'[$£€¥]\s*\d+([.,]\d+)?', ' dollar_amount ', text)
    text = re.sub(r'[$£€¥]', ' dollar_symbol ', text)
    text = re.sub(r'\b\d+\b', ' num_token ', text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9_\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_security_signals(raw_text: str, cleaned_text: str) -> List[Dict[str, str]]:
    """
    Scans the email for known phishing/spam semantic triggers,
    structural anomalies, and keyword patterns.
    """
    signals = []
    text_lower = raw_text.lower()

    # Regex security patterns
    for pattern, description in SECURITY_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            signals.append({
                "type": "pattern",
                "label": description,
                "matched": match.group(0).strip()
            })

    # Check structural attributes
    url_count = len(re.findall(r'http[s]?://|www\.', raw_text, re.IGNORECASE))
    if url_count > 0:
        signals.append({
            "type": "structural",
            "label": f"Embedded Hyperlinks ({url_count} found)",
            "matched": f"{url_count} URL{'s' if url_count > 1 else ''}"
        })

    currency_matches = re.findall(r'[$£€¥]\s*\d+', raw_text)
    if currency_matches:
        signals.append({
            "type": "financial",
            "label": "Monetary / Currency Claims",
            "matched": ", ".join(currency_matches[:3])
        })

    # Top spam vocabulary lookup if available from metrics
    top_vocab = [item["word"] for item in metrics_data.get("top_spam_keywords", [])[:20]]
    cleaned_tokens = set(cleaned_text.split())
    found_top_terms = [word for word in top_vocab if word in cleaned_tokens]
    if found_top_terms:
        signals.append({
            "type": "keyword",
            "label": "High-Confidence Spam Vocabulary",
            "matched": ", ".join(found_top_terms[:5])
        })

    return signals


# Benchmark sample emails for user experimentation
BENCHMARK_SAMPLES = {
    "phishing": {
        "title": "🚨 Urgent Security / Account Phishing",
        "category": "High Risk Spam",
        "content": (
            "Subject: URGENT: Your Online Banking Access Has Been Suspended!\n\n"
            "Dear Valued Customer,\n\n"
            "We detected unauthorized login attempts to your bank account from an unrecognized IP address. "
            "For your immediate protection, your debit card and online access have been restricted.\n\n"
            "To restore access, you must verify your identity within 24 hours by clicking the link below:\n"
            "http://secure-banking-verification-update99.com/auth/login.php\n\n"
            "Failure to confirm your credentials will result in permanent account termination and a $150 penalty fee.\n\n"
            "Security Department,\n"
            "Global Bank Trust Operations"
        )
    },
    "lottery": {
        "title": "🎁 International Lottery / Crypto Prize Scam",
        "category": "High Risk Spam",
        "content": (
            "Subject: CONGRATULATIONS! You Have Won $2,500,000.00 USD In The International Promo Award!\n\n"
            "Attention Lucky Winner,\n\n"
            "Your email address was officially selected as the winner of Two Million Five Hundred Thousand Dollars ($2,500,000) "
            "in our automated year-end sweepstakes draw.\n\n"
            "To claim your prize, reply immediately with:\n"
            "1. Full Name\n"
            "2. Direct Phone Number\n"
            "3. Home Address\n"
            "4. Bank Account Details for Wire Transfer\n\n"
            "Contact our claim agent Dr. Raymond Adams at: claim-agent@swiss-award-payouts.net.\n\n"
            "Act now! All unclaimed prizes will be forfeited after 48 hours."
        )
    },
    "business_meeting": {
        "title": "💼 Executive Meeting / Q3 Product Review",
        "category": "Legitimate Ham",
        "content": (
            "Subject: Agenda: Q3 Product Roadmap & Architecture Sync - Thursday 2:00 PM\n\n"
            "Hi Team,\n\n"
            "Please review the agenda for our upcoming product review this Thursday at 2:00 PM EST:\n"
            "1. Sprint 24 retrospective and deliverables\n"
            "2. Database migration timeline and benchmark results\n"
            "3. API latency optimizations for the customer portal\n"
            "4. Open Q&A and action items\n\n"
            "The updated slide deck is available in our shared Google Drive folder. "
            "Let me know if there are additional topics you would like to include.\n\n"
            "Best regards,\n"
            "Sarah Jenkins\n"
            "Senior Engineering Manager"
        )
    },
    "release_notes": {
        "title": "📄 Documentation & Milestone Release Update",
        "category": "Legitimate Ham",
        "content": (
            "Subject: Documentation and User Guide Updates for Version 2.4 Milestone\n\n"
            "Hi Team,\n\n"
            "Please review the updated technical documentation and user guide for our upcoming 2.4 milestone release. "
            "All API parameter definitions and error code tables have been finalized. "
            "Let me know if any sections need clarification before we publish to the knowledge base tomorrow morning.\n\n"
            "Best regards,\n"
            "Alex Chen\n"
            "Technical Writer"
        )
    }
}


# =============================================================================
# WEB ROUTES
# =============================================================================

@app.route("/")
def index():
    """Renders the main SPAMGUARD AI landing page & analysis studio."""
    return render_template("index.html")


# =============================================================================
# RESTful API ENDPOINTS
# =============================================================================

@app.route("/api/health", methods=["GET"])
def health():
    """
    Health check endpoint returning system status and model readiness.
    """
    is_ready = pipeline is not None
    return jsonify({
        "status": "healthy" if is_ready else "degraded",
        "service": "SPAMGUARD AI - Spam Detection Engine",
        "version": "1.0.0",
        "model_loaded": is_ready,
        "metrics_available": bool(metrics_data),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }), 200


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """
    Returns empirical model evaluation metrics, confusion matrix,
    dataset split information, and top predictive features.
    """
    if not metrics_data:
        # Fallback check if file exists
        if os.path.exists(METRICS_PATH):
            load_model_and_metrics()

    if not metrics_data:
        return jsonify({
            "error": "Model metrics not available. Please run train_model.py first."
        }), 404

    return jsonify(metrics_data), 200


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """
    Provides curated sample emails for testing spam and legitimate patterns.
    """
    return jsonify(BENCHMARK_SAMPLES), 200


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Analyzes email text content and returns genuine ML prediction:
    - SPAM vs NOT SPAM
    - Calibrated probability percentages (Spam % vs Ham %)
    - Cybersecurity Risk Level
    - Detected threat signals and indicators
    - Explainable metadata
    """
    start_time = time.time()

    # Verify model is ready
    if pipeline is None:
        load_model_and_metrics()
        if pipeline is None:
            return jsonify({
                "error": "Machine learning model pipeline is not loaded. Please execute: python train_model.py"
            }), 503

    # Parse and validate incoming JSON payload
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "error": "Invalid JSON request payload. Expected format: {'email_text': '...'}"
        }), 400

    raw_text = data.get("email_text", "")
    if not isinstance(raw_text, str):
        return jsonify({
            "error": "Field 'email_text' must be a valid string."
        }), 400

    # Clean and check length
    trimmed = raw_text.strip()
    if not trimmed:
        return jsonify({
            "error": "Email content cannot be empty. Please paste email body or subject."
        }), 400

    if len(trimmed) < 5:
        return jsonify({
            "error": "Email content is too short for accurate classification (minimum 5 characters required)."
        }), 400

    # NLP Preprocessing
    cleaned_text = clean_email_text(trimmed)
    if not cleaned_text or len(cleaned_text.split()) < 1:
        cleaned_text = trimmed.lower()

    # Execute Model Inference
    try:
        # Predict class label: 0 = Ham, 1 = Spam
        prediction_int = int(pipeline.predict([cleaned_text])[0])
        # Predict class probabilities [P(Ham), P(Spam)]
        probabilities = pipeline.predict_proba([cleaned_text])[0]
        ham_prob = float(probabilities[0]) * 100
        spam_prob = float(probabilities[1]) * 100
    except Exception as e:
        return jsonify({
            "error": f"Inference execution failed: {str(e)}"
        }), 500

    is_spam = bool(prediction_int == 1)
    prediction_label = "SPAM" if is_spam else "NOT SPAM"

    # Determine confidence score based on the predicted class
    confidence = spam_prob if is_spam else ham_prob

    # Determine Cybersecurity Risk Level
    if spam_prob >= 85.0:
        risk_level = "CRITICAL"
        risk_color = "#FF0055"
        verdict_summary = "High-threat malicious spam or credential phishing campaign."
        security_action = "Quarantine immediately. Do not click links or open attachments."
    elif spam_prob >= 50.0:
        risk_level = "HIGH"
        risk_color = "#FF6B6B"
        verdict_summary = "Unsolicited promotional or suspicious email communication."
        security_action = "Mark as spam and treat unverified requests with caution."
    elif spam_prob >= 25.0:
        risk_level = "MEDIUM"
        risk_color = "#FFAA00"
        verdict_summary = "Borderline content containing minor marketing or external links."
        security_action = "Inspect sender headers and verify legitimacy through a known channel."
    else:
        risk_level = "SAFE"
        risk_color = "#00F2FE"
        verdict_summary = "Standard legitimate correspondence matching normal email patterns."
        security_action = "Safe to read. No anomalous threat signals detected."

    # Extract Explainable AI security signals
    detected_signals = extract_security_signals(raw_text, cleaned_text)

    inference_ms = round((time.time() - start_time) * 1000, 2)

    response_payload = {
        "success": True,
        "prediction": prediction_label,
        "is_spam": is_spam,
        "confidence": round(confidence, 2),
        "spam_probability": round(spam_prob, 2),
        "ham_probability": round(ham_prob, 2),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "verdict_summary": verdict_summary,
        "security_action": security_action,
        "detected_signals": detected_signals,
        "metrics": {
            "character_count": len(raw_text),
            "word_count": len(raw_text.split()),
            "preprocessed_tokens": len(cleaned_text.split()),
            "inference_time_ms": inference_ms
        },
        "preprocessed_snippet": cleaned_text[:300] + ("..." if len(cleaned_text) > 300 else "")
    }

    return jsonify(response_payload), 200


# Initial loading when module is imported
load_model_and_metrics()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"\n==================================================================")
    print(f" SPAMGUARD AI SERVER ONLINE")
    print(f" Local URL: http://{host}:{port}")
    print(f"==================================================================\n")
    app.run(host=host, port=port, debug=False)
