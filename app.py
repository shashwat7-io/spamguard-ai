

import os
import re
import json
import time
from html import unescape
from typing import Dict, Any, List, Tuple, Optional

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

# Application Initialization
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
PIPELINE_PATH = os.path.join(MODEL_DIR, "spam_classifier_pipeline.joblib")
BASELINE_PIPELINE_PATH = os.path.join(MODEL_DIR, "baseline_nb_pipeline.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# Maximum input limits to prevent Resource Exhaustion / Denial of Service
MAX_BODY_LENGTH = 100000
MAX_SUBJECT_LENGTH = 1000
MAX_SENDER_LENGTH = 320

# Global variables for cached model pipelines and metrics
pipeline = None               # Primary (Best-performing: Linear SVM)
baseline_pipeline = None      # Baseline (Multinomial Naive Bayes)
metrics_data: Dict[str, Any] = {}

# High-risk TLDs commonly abused in disposable spam and phishing campaigns
SUSPICIOUS_TLDS = {
    "xyz", "top", "click", "loan", "work", "gq", "cf", "tk", "ml",
    "fit", "surf", "monster", "club", "buzz", "rest", "icu", "download",
    "accountant", "racing", "kim", "party", "review", "country", "stream"
}

# Known URL shorteners used to obfuscate destinations
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "ow.ly", "buff.ly",
    "cutt.ly", "rb.gy", "goo.gl", "shorturl.at"
}

# Targeted financial, tech, and enterprise brand lookalikes
TARGETED_BRANDS = [
    "paypal", "apple", "google", "bankofamerica", "chase", "wellsfargo",
    "citibank", "microsoft", "netflix", "amazon", "facebook", "instagram",
    "binance", "coinbase", "metamask", "blockchain", "irs", "gov", "dhl", "fedex"
]

# Structured Phishing & Threat Keyword Patterns
THREAT_PATTERNS = [
    {
        "category": "Credential Theft",
        "pattern": r"\b(verify|validate|confirm|re-activate|reactivate|authenticate)\s+(your\s+)?(account|identity|details|password|pin|ssn|wallet|credentials|passcode|login)\b",
        "label": "Account Verification / Credential Request",
        "severity": "CRITICAL",
        "weight": 25
    },
    {
        "category": "Urgency & Threats",
        "pattern": r"\b(urgent|immediate|action\s+required|critical\s+alert|suspended|terminated|deactivated|restricted|disabled\s+within|24\s+hours|final\s+notice|penalty\s+fee|legal\s+action)\b",
        "label": "Urgency / Account Suspension Threat",
        "severity": "CRITICAL",
        "weight": 25
    },
    {
        "category": "Financial Solicitation",
        "pattern": r"\b(wire\s+transfer|western\s+union|bank\s+transfer|money\s*gram|cryptocurrency|bitcoin|btc|ethereum|gift\s+card|steam\s+card)\b",
        "label": "Irreversible Wire / Cryptocurrency Transfer",
        "severity": "HIGH",
        "weight": 20
    },
    {
        "category": "Lottery & Sweepstakes Bait",
        "pattern": r"\b(congratulations|you\s+won|lottery|prize|jackpot|selected|winner|sweepstakes|award|bonus\s+payout)\b",
        "label": "Lottery / Advance-Fee Prize Bait",
        "severity": "HIGH",
        "weight": 20
    },
    {
        "category": "Advance-Fee & Inheritance Scam",
        "pattern": r"\b(million\s+dollars|inherit(ance)?|deceased|prince|confidential\s+proposal|next\s+of\s+kin|beneficiary|unclaimed\s+funds)\b",
        "label": "Advance-Fee / Inheritance Scam Pattern",
        "severity": "HIGH",
        "weight": 20
    },
    {
        "category": "Deceptive Call-To-Action",
        "pattern": r"\b(click\s+(here|below|this\s+link)|follow\s+this\s+link|verify\s+now|claim\s+now|download\s+attachment|open\s+the\s+attached|view\s+invoice)\b",
        "label": "Suspicious Call-To-Action (CTA)",
        "severity": "MEDIUM",
        "weight": 15
    },
    {
        "category": "Promotional Solicitation",
        "pattern": r"\b(claim\s+your|exclusive\s+deal|free\s+access|100%\s+free|risk\s+free|lowest\s+rates|pre-approved|bad\s+credit)\b",
        "label": "Unsolicited Commercial / Loan Solicitation",
        "severity": "LOW",
        "weight": 10
    },
    {
        "category": "Bulk Mailing Notice",
        "pattern": r"\b(opt\s*out|remove\s+here|unsubscribe|future\s+mailings|email\s+preferences)\b",
        "label": "Bulk Mailing / Unsubscribe Notice",
        "severity": "LOW",
        "weight": 5
    }
]


def load_model_and_metrics() -> None:
    """
    Loads serialized ML pipelines and metrics JSON into memory.
    Supports both best model (Linear SVM) and baseline (MultinomialNB).
    """
    global pipeline, baseline_pipeline, metrics_data

    
    if os.path.exists(PIPELINE_PATH):
        try:
            pipeline = joblib.load(PIPELINE_PATH)
            print(f"[+] Loaded Primary ML pipeline from {PIPELINE_PATH}")
        except Exception as e:
            print(f"[-] Error loading primary pipeline: {e}")
            pipeline = None
    else:
        print(f"[!] Pipeline not found at {PIPELINE_PATH}. Please run: python train_model.py")
        pipeline = None

    
    if os.path.exists(BASELINE_PIPELINE_PATH):
        try:
            baseline_pipeline = joblib.load(BASELINE_PIPELINE_PATH)
            print(f"[+] Loaded Baseline ML pipeline from {BASELINE_PIPELINE_PATH}")
        except Exception as e:
            print(f"[-] Error loading baseline pipeline: {e}")
            baseline_pipeline = None

    
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


def inspect_urls_for_threats(text: str) -> List[Dict[str, Any]]:
    """
    Scans email text for hyperlinks and evaluates phishing/evasion characteristics:
    - IP address hostnames (e.g. http://192.168.1.1/...)
    - Known suspicious or abused Top-Level Domains (.xyz, .top, .click, etc.)
    - Brand lookalike spoofing (e.g. login-paypal-verification.com)
    - URL shorteners (bit.ly, tinyurl.com)
    - Embedded '@' authentication evasion
    - Non-standard HTTP ports
    """
    results = []
    
    url_matches = re.findall(r'https?://[^\s<>"\'\)]+|www\.[^\s<>"\'\)]+', text, re.IGNORECASE)

    for raw_url in url_matches:
        url_clean = raw_url.strip(".,;:?!")
        reasons = []
        severity = "LOW"

        
        ip_match = re.search(r'https?://(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', url_clean)
        if ip_match:
            reasons.append(f"Direct IP-based URL ({ip_match.group(1)}) bypassing DNS verification")
            severity = "CRITICAL"

        
        host_match = re.search(r'https?://([^/:\s]+)', url_clean, re.IGNORECASE)
        host = host_match.group(1).lower() if host_match else ""

        if host:
            # Check TLD
            parts = host.split(".")
            if len(parts) >= 2:
                tld = parts[-1].lower()
                if tld in SUSPICIOUS_TLDS:
                    reasons.append(f"Abused / High-Risk TLD (.{tld}) commonly associated with phishing campaigns")
                    severity = "HIGH" if severity != "CRITICAL" else severity

            # Check URL shorteners
            if any(host == shortener or host.endswith("." + shortener) for shortener in URL_SHORTENERS):
                reasons.append(f"Obfuscated URL shortener ({host}) hiding real destination")
                severity = "HIGH" if severity != "CRITICAL" else severity

            
            for brand in TARGETED_BRANDS:
                if brand in host and not host.endswith(f".{brand}.com") and not host == f"{brand}.com":
                    reasons.append(f"Deceptive brand impersonation keyword detected: '{brand}' in '{host}'")
                    severity = "CRITICAL"
                    break

            
            if host.count("-") >= 2:
                reasons.append("Multiple hyphenated subdomains commonly utilized in disposable credential harvesters")
                if severity == "LOW":
                    severity = "MEDIUM"

        # Non-standard ports
        port_match = re.search(r'https?://[^/]+:(\d+)', url_clean)
        if port_match:
            port = port_match.group(1)
            if port not in ["80", "443"]:
                reasons.append(f"Non-standard HTTP port (:{port})")
                if severity == "LOW":
                    severity = "MEDIUM"

        # '@' evasion inside URL
        if "@" in url_clean.split("?")[0].replace("http://", "").replace("https://", ""):
            reasons.append("Credential obfuscation attempt utilizing '@' authority in URL")
            severity = "CRITICAL"

        results.append({
            "url": url_clean,
            "host": host or url_clean[:30],
            "is_suspicious": len(reasons) > 0,
            "severity": severity,
            "reasons": reasons if reasons else ["Standard domain structure"]
        })

    return results


def inspect_sender_security(sender: str) -> Dict[str, Any]:
    """
    Evaluates the sender email address for spoofing, free-mail mismatches,
    and syntactic anomalies.
    """
    if not sender or not isinstance(sender, str):
        return {
            "provided": False,
            "is_suspicious": False,
            "flags": []
        }

    sender = sender.strip()
    flags = []
    is_suspicious = False

    # Extract email address if format is "Name <user@domain.com>"
    email_match = re.search(r'<([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})>', sender)
    addr = email_match.group(1) if email_match else sender

    # Basic regex validation
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', addr):
        flags.append("Malformed or non-standard RFC email syntax")
        is_suspicious = True

    # Domain checks
    if "@" in addr:
        local_part, domain = addr.split("@", 1)
        domain = domain.lower()
        tld = domain.split(".")[-1] if "." in domain else ""

        if tld in SUSPICIOUS_TLDS:
            flags.append(f"Sender domain uses high-abuse TLD (.{tld})")
            is_suspicious = True

        # Brand impersonation in local-part while using free webmail
        free_providers = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"]
        for brand in ["security", "support", "billing", "account", "verification", "service"]:
            if brand in local_part.lower() and domain in free_providers:
                flags.append(f"Free webmail provider ({domain}) impersonating corporate '{brand}' handle")
                is_suspicious = True
                break

        # High entropy random alphanumeric localpart
        if len(local_part) >= 12 and sum(c.isdigit() for c in local_part) >= 5:
            flags.append("Randomized high-entropy address typical of automated botnet mailers")
            is_suspicious = True

    return {
        "provided": True,
        "address": addr,
        "is_suspicious": is_suspicious,
        "flags": flags
    }


def analyze_email_threats(
    subject: str,
    sender: str,
    body: str,
    raw_text: str,
    cleaned_text: str
) -> Dict[str, Any]:
    """
    Comprehensive rule-based security scanner:
    - Extracts phishing keyword triggers
    - Analyzes URL risks
    - Evaluates sender address
    - Computes an independent heuristic Phishing Score (0-100)
    """
    combined_raw = f"{subject} {sender} {body} {raw_text}".lower()

    # 1. Threat Keyword Patterns
    matched_signals = []
    phishing_score = 0

    for item in THREAT_PATTERNS:
        match = re.search(item["pattern"], combined_raw, re.IGNORECASE)
        if match:
            matched_text = match.group(0).strip()
            matched_signals.append({
                "category": item["category"],
                "label": item["label"],
                "severity": item["severity"],
                "matched": matched_text
            })
            phishing_score += item["weight"]

    # 2. Suspicious URL Analysis
    url_results = inspect_urls_for_threats(f"{subject} {body} {raw_text}")
    suspicious_urls = [u for u in url_results if u["is_suspicious"]]
    critical_urls = [u for u in url_results if u["severity"] == "CRITICAL"]

    if critical_urls:
        phishing_score += 40
    elif suspicious_urls:
        phishing_score += 20

    # 3. Sender Security
    sender_eval = inspect_sender_security(sender)
    if sender_eval.get("is_suspicious"):
        phishing_score += 20

    # 4. Currency / Financial Claims
    currency_matches = re.findall(r'[$£€¥]\s*\d+', raw_text)
    if currency_matches:
        matched_signals.append({
            "category": "Monetary Claims",
            "label": "Direct Currency / Payout Sums",
            "severity": "MEDIUM",
            "matched": ", ".join(currency_matches[:3])
        })
        phishing_score += 10

    # Cap score at 100
    phishing_score = min(100, phishing_score)

    # Determine Phishing Risk Level
    if phishing_score >= 60 or critical_urls:
        phishing_risk = "CRITICAL"
        phishing_verdict = "Active Phishing Threat Detected"
    elif phishing_score >= 35 or suspicious_urls:
        phishing_risk = "HIGH"
        phishing_verdict = "Suspicious Phishing Indicators Found"
    elif phishing_score >= 15:
        phishing_risk = "MEDIUM"
        phishing_verdict = "Moderate Risk (Unsolicited Marketing / CTA)"
    else:
        phishing_risk = "LOW"
        phishing_verdict = "No Obvious Phishing Heuristics Detected"

    return {
        "phishing_detected": phishing_score >= 35,
        "phishing_score": phishing_score,
        "phishing_risk_level": phishing_risk,
        "phishing_verdict": phishing_verdict,
        "threat_signals": matched_signals,
        "url_analysis": {
            "total_urls_found": len(url_results),
            "suspicious_urls_count": len(suspicious_urls),
            "all_urls": url_results,
            "suspicious_urls": suspicious_urls
        },
        "sender_analysis": sender_eval
    }


def compute_model_probabilities(active_pipe: Any, cleaned_text: str) -> Tuple[int, float, float]:
    """
    Extracts prediction and class probabilities from any Scikit-Learn pipeline.
    Handles classifiers with native `predict_proba` (MNB, Logistic Regression, Random Forest)
    as well as Linear Support Vector Machines (LinearSVC) via Platt/Sigmoid calibration.
    Returns: (predicted_class, ham_probability_percent, spam_probability_percent)
    """
    pred_int = int(active_pipe.predict([cleaned_text])[0])

    # Check if classifier or pipeline exposes predict_proba
    classifier = None
    for step_name in ['classifier', 'nb', 'svm', 'model', 'clf']:
        if step_name in active_pipe.named_steps:
            classifier = active_pipe.named_steps[step_name]
            break

    if classifier is not None and hasattr(classifier, 'predict_proba'):
        probs = active_pipe.predict_proba([cleaned_text])[0]
        ham_pct = float(probs[0]) * 100.0
        spam_pct = float(probs[1]) * 100.0
    elif classifier is not None and hasattr(classifier, 'decision_function'):
        # Platt/Sigmoid scaling for Linear SVM:
        # P(Spam) = 1 / (1 + exp(-6.0 * decision))
        df = float(active_pipe.decision_function([cleaned_text])[0])
        scaled_df = max(-25.0, min(25.0, df * 6.0))
        spam_prob = 1.0 / (1.0 + np.exp(-scaled_df))
        spam_pct = float(spam_prob) * 100.0
        ham_pct = 100.0 - spam_pct
    else:
        # Fallback to binary confidence
        spam_pct = 95.0 if pred_int == 1 else 5.0
        ham_pct = 100.0 - spam_pct

    return pred_int, ham_pct, spam_pct


# Benchmark sample emails for testing & experimentation
BENCHMARK_SAMPLES = {
    "phishing": {
        "title": "🚨 Urgent Security / Account Phishing",
        "category": "High Risk Spam & Phishing",
        "subject": "URGENT: Your Online Banking Access Has Been Suspended!",
        "sender": "security-alert@verify-banking-account99.xyz",
        "body": (
            "Dear Valued Customer,\n\n"
            "We detected unauthorized login attempts to your bank account from an unrecognized IP address. "
            "For your immediate protection, your debit card and online access have been restricted.\n\n"
            "To restore access, you must verify your identity within 24 hours by clicking the link below:\n"
            "http://secure-banking-verification-update99.com/auth/login.php\n\n"
            "Failure to confirm your credentials will result in permanent account termination and a $150 penalty fee.\n\n"
            "Security Department,\n"
            "Global Bank Trust Operations"
        ),
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
        "subject": "CONGRATULATIONS! You Have Won $2,500,000.00 USD In The International Promo Award!",
        "sender": "claim-agent@swiss-award-payouts.net",
        "body": (
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
        ),
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
        "subject": "Agenda: Q3 Product Roadmap & Architecture Sync - Thursday 2:00 PM",
        "sender": "sarah.jenkins@enterprise-techcorp.internal",
        "body": (
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
        ),
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
        "subject": "Documentation and User Guide Updates for Version 2.4 Milestone",
        "sender": "alex.chen@devtools-org.org",
        "body": (
            "Hi Team,\n\n"
            "Please review the updated technical documentation and user guide for our upcoming 2.4 milestone release. "
            "All API parameter definitions and error code tables have been finalized. "
            "Let me know if any sections need clarification before we publish to the knowledge base tomorrow morning.\n\n"
            "Best regards,\n"
            "Alex Chen\n"
            "Technical Writer"
        ),
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
    """Renders the main Spam Email Detection landing page, analyzer console & dashboard."""
    return render_template("index.html")


# =============================================================================
# RESTful API ENDPOINTS
# =============================================================================

@app.route("/api/health", methods=["GET"])
def health():
    """
    Health check endpoint returning system status, model readiness,
    and multi-model availability.
    """
    is_ready = pipeline is not None
    active_model_name = metrics_data.get("best_model_name", "Linear SVM") if is_ready else "None"

    return jsonify({
        "status": "healthy" if is_ready else "degraded",
        "service": "Spam Email Detection - Threat Intelligence & Spam Engine",
        "version": "2.0.0",
        "model_loaded": is_ready,
        "active_model": active_model_name,
        "baseline_model": metrics_data.get("baseline_model_name", "Multinomial Naive Bayes"),
        "metrics_available": bool(metrics_data),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }), 200


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """
    Returns empirical evaluation metrics across all 4 benchmarked models:
    - Multinomial Naive Bayes (Baseline)
    - Logistic Regression
    - Random Forest
    - Linear SVM (Best Model)
    Includes confusion matrices, train/test counts, dataset statistics,
    and explainability feature weights.
    """
    if not metrics_data:
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
    Provides curated sample emails with separate subject, sender, and body
    for interactive testing.
    """
    return jsonify(BENCHMARK_SAMPLES), 200


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Comprehensive Threat Detection API:
    - Accepts multi-field input: {subject, sender, body} OR single {email_text}
    - Classifies combined subject + body through Scikit-Learn NLP pipeline
    - Executes rule-based suspicious URL & phishing keyword analysis
    - Clearly distinguishes statistical ML spam verdict from heuristic phishing indicators
    - Defense-in-depth: Never guarantees an email is safe.
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
            "error": "Invalid JSON request payload. Expected format: {'subject': '...', 'sender': '...', 'body': '...'} or {'email_text': '...'}"
        }), 400

    # Extract fields with backwards-compatibility support
    subject = data.get("subject", "")
    sender = data.get("sender", data.get("sender_email", ""))
    body = data.get("body", "")
    raw_email_text = data.get("email_text", "")
    model_choice = str(data.get("model", "best")).lower()

    # Input type validation
    if not isinstance(subject, str) or not isinstance(sender, str) or not isinstance(body, str):
        return jsonify({"error": "Fields 'subject', 'sender', and 'body' must be strings."}), 400

    if raw_email_text is not None and not isinstance(raw_email_text, str):
        return jsonify({"error": "Field 'email_text' must be a string."}), 400

    # Length limits to prevent Denial of Service / Memory Exhaustion
    if len(body) > MAX_BODY_LENGTH:
        return jsonify({
            "error": f"Email body exceeds maximum allowed size of {MAX_BODY_LENGTH:,} characters."
        }), 400

    if len(subject) > MAX_SUBJECT_LENGTH:
        return jsonify({
            "error": f"Email subject exceeds maximum allowed size of {MAX_SUBJECT_LENGTH:,} characters."
        }), 400

    if len(sender) > MAX_SENDER_LENGTH:
        return jsonify({
            "error": f"Sender address exceeds maximum allowed size of {MAX_SENDER_LENGTH:,} characters."
        }), 400

    if len(raw_email_text) > MAX_BODY_LENGTH:
        return jsonify({
            "error": f"Email content exceeds maximum allowed size of {MAX_BODY_LENGTH:,} characters."
        }), 400

    # Combine subject and body for classification
    if raw_email_text and not body:
        # Check if raw_email_text has a Subject header
        subj_match = re.search(r'^Subject:\s*(.+)$', raw_email_text, re.MULTILINE | re.IGNORECASE)
        if subj_match and not subject:
            subject = subj_match.group(1).strip()
        full_text = raw_email_text.strip()
    else:
        # Combine subject and body
        parts = []
        if subject.strip():
            parts.append(f"Subject: {subject.strip()}")
        if body.strip():
            parts.append(body.strip())
        full_text = "\n\n".join(parts).strip()

    # Emptiness and minimum length checks
    if not full_text:
        return jsonify({
            "error": "Email content cannot be empty. Please provide email body, subject, or text."
        }), 400

    if len(full_text) < 5:
        return jsonify({
            "error": "Email content is too short for accurate classification (minimum 5 characters required)."
        }), 400

    # Model selection (Best vs Baseline)
    active_pipe = pipeline
    active_model_name = metrics_data.get("best_model_name", "Linear SVM")

    if model_choice in ["baseline", "naive_bayes", "multinomialnb", "nb"] and baseline_pipeline is not None:
        active_pipe = baseline_pipeline
        active_model_name = "Multinomial Naive Bayes"

    # NLP Preprocessing
    cleaned_text = clean_email_text(full_text)
    if not cleaned_text or len(cleaned_text.split()) < 1:
        cleaned_text = full_text.lower()

    # Execute Machine Learning Inference
    try:
        pred_int, ham_prob, spam_prob = compute_model_probabilities(active_pipe, cleaned_text)
    except Exception as e:
        return jsonify({
            "error": f"Machine learning inference failed: {str(e)}"
        }), 500

    is_spam = bool(pred_int == 1)
    ml_prediction_label = "SPAM" if is_spam else "NOT SPAM"
    ml_confidence = spam_prob if is_spam else ham_prob

    # Execute Rule-Based Suspicious URL & Phishing Threat Intelligence
    threat_assessment = analyze_email_threats(
        subject=subject,
        sender=sender,
        body=body,
        raw_text=full_text,
        cleaned_text=cleaned_text
    )

    # Correlate Statistical ML Prediction with Heuristic Phishing Indicators
    # Determine Unified Severity Level
    phishing_risk = threat_assessment["phishing_risk_level"]
    phishing_score = threat_assessment["phishing_score"]

    if spam_prob >= 85.0 or phishing_risk == "CRITICAL":
        risk_level = "CRITICAL"
        risk_color = "#FF0055"
        verdict_summary = "High-threat malicious spam or credential harvesting phishing campaign."
        security_action = "Quarantine immediately. Do not click links, open attachments, or input credentials."
    elif spam_prob >= 50.0 or phishing_risk == "HIGH":
        risk_level = "HIGH"
        risk_color = "#FF6B6B"
        verdict_summary = "Unsolicited spam or suspicious communication with potential threat indicators."
        security_action = "Mark as spam and treat unverified requests with extreme caution."
    elif spam_prob >= 25.0 or phishing_risk == "MEDIUM":
        risk_level = "MEDIUM"
        risk_color = "#FFAA00"
        verdict_summary = "Borderline communication with marketing patterns or external hyperlinks."
        security_action = "Inspect sender headers and verify legitimacy through a secondary communication channel."
    else:
        # Statistical ham pattern - defense in depth applies
        risk_level = "SAFE"
        risk_color = "#10B981"
        verdict_summary = "Low-risk communication matching legitimate correspondence patterns."
        security_action = "Standard reading safe, but zero detected indicators does NOT guarantee safety. Exercise vigilance."

    # Backwards-compatible detected_signals list
    detected_signals = threat_assessment["threat_signals"]

    # Add URL signals to legacy detected_signals
    for s_url in threat_assessment["url_analysis"]["suspicious_urls"]:
        for reason in s_url["reasons"]:
            detected_signals.append({
                "type": "url",
                "label": f"Suspicious Link: {s_url['host']}",
                "matched": reason
            })

    inference_ms = round((time.time() - start_time) * 1000, 2)

    # Mandatory Defense-In-Depth Safety Disclaimer
    safety_disclaimer = (
        "Defense-in-Depth Notice: Machine learning models and heuristic rules assess statistical "
        "and lexical indicators. A classification of 'NOT SPAM' or 'LOW RISK' is NOT a guarantee "
        "that an email is safe. Sophisticated spear-phishing and zero-day attacks may evade detection. "
        "Always verify unexpected requests through trusted out-of-band channels."
    )

    response_payload = {
        "success": True,
        # Legacy fields for complete client and test backwards-compatibility
        "prediction": ml_prediction_label,
        "is_spam": is_spam,
        "confidence": round(ml_confidence, 2),
        "spam_probability": round(spam_prob, 2),
        "ham_probability": round(ham_prob, 2),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "verdict_summary": verdict_summary,
        "security_action": security_action,
        "detected_signals": detected_signals,
        "safety_guarantee": False,
        "safety_disclaimer": safety_disclaimer,

        # Granular Model Classification Details
        "model_classification": {
            "model_name": active_model_name,
            "prediction": ml_prediction_label,
            "is_spam": is_spam,
            "confidence": round(ml_confidence, 2),
            "spam_probability": round(spam_prob, 2),
            "ham_probability": round(ham_prob, 2)
        },

        # Distinct Heuristic Phishing Assessment
        "phishing_assessment": threat_assessment,

        # Input Telemetry & Metadata
        "input_summary": {
            "has_subject": bool(subject.strip()),
            "has_sender": bool(sender.strip()),
            "subject": subject[:100],
            "sender": sender[:100],
            "character_count": len(full_text),
            "word_count": len(full_text.split()),
            "preprocessed_tokens": len(cleaned_text.split()),
            "total_urls_detected": threat_assessment["url_analysis"]["total_urls_found"]
        },
        "metrics": {
            "character_count": len(full_text),
            "word_count": len(full_text.split()),
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
    print(f" SPAM EMAIL DETECTION SERVER ONLINE (v2.0)")
    print(f" Local URL: http://{host}:{port}")
    print(f"==================================================================\n")
    app.run(host=host, port=port, debug=False)
