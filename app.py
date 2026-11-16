import os
import re
import json
import time
from html import unescape

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
PIPELINE_PATH = os.path.join(MODEL_DIR, "spam_classifier_pipeline.joblib")
BASELINE_PIPELINE_PATH = os.path.join(MODEL_DIR, "baseline_nb_pipeline.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

MAX_BODY_LENGTH = 100000
MAX_SUBJECT_LENGTH = 1000
MAX_SENDER_LENGTH = 320

pipeline = None
baseline_pipeline = None
metrics_data = {}

SUSPICIOUS_TLDS = {
    "xyz", "top", "click", "loan", "work", "gq", "cf", "tk", "ml",
    "fit", "surf", "monster", "club", "buzz", "rest", "icu", "download",
    "accountant", "racing", "kim", "party", "review", "country", "stream"
}

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "ow.ly", "buff.ly",
    "cutt.ly", "rb.gy", "goo.gl", "shorturl.at"
}

TARGETED_BRANDS = [
    "paypal", "apple", "google", "bankofamerica", "chase", "wellsfargo",
    "citibank", "microsoft", "netflix", "amazon", "facebook", "instagram",
    "binance", "coinbase", "metamask", "blockchain", "irs", "gov", "dhl", "fedex"
]

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

def load_model_and_metrics():
    global pipeline, baseline_pipeline, metrics_data

    if os.path.exists(PIPELINE_PATH):
        try:
            pipeline = joblib.load(PIPELINE_PATH)
            print("Loaded primary model from", PIPELINE_PATH)
        except Exception as e:
            print("Error loading model:", e)
            pipeline = None
    else:
        print("Model file not found:", PIPELINE_PATH)
        pipeline = None

    if os.path.exists(BASELINE_PIPELINE_PATH):
        try:
            baseline_pipeline = joblib.load(BASELINE_PIPELINE_PATH)
            print("Loaded baseline model from", BASELINE_PIPELINE_PATH)
        except Exception as e:
            print("Error loading baseline model:", e)
            baseline_pipeline = None

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            print("Loaded metrics from", METRICS_PATH)
        except Exception as e:
            print("Error loading metrics:", e)
            metrics_data = {}
    else:
        print("Metrics file not found:", METRICS_PATH)
        metrics_data = {}

def clean_email_text(text):
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

def inspect_urls_for_threats(text):
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
            parts = host.split(".")
            if len(parts) >= 2:
                tld = parts[-1].lower()
                if tld in SUSPICIOUS_TLDS:
                    reasons.append(f"Abused / High-Risk TLD (.{tld}) commonly associated with phishing campaigns")
                    severity = "HIGH" if severity != "CRITICAL" else severity

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

        port_match = re.search(r'https?://[^/]+:(\d+)', url_clean)
        if port_match:
            port = port_match.group(1)
            if port not in ["80", "443"]:
                reasons.append(f"Non-standard HTTP port (:{port})")
                if severity == "LOW":
                    severity = "MEDIUM"

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

def inspect_sender_security(sender):
    if not sender or not isinstance(sender, str):
        return {
            "provided": False,
            "is_suspicious": False,
            "flags": []
        }

    sender = sender.strip()
    flags = []
    is_suspicious = False

    email_match = re.search(r'<([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})>', sender)
    addr = email_match.group(1) if email_match else sender

    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', addr):
        flags.append("Malformed or non-standard RFC email syntax")
        is_suspicious = True

    if "@" in addr:
        local_part, domain = addr.split("@", 1)
        domain = domain.lower()
        tld = domain.split(".")[-1] if "." in domain else ""

        if tld in SUSPICIOUS_TLDS:
            flags.append(f"Sender domain uses high-abuse TLD (.{tld})")
            is_suspicious = True

        free_providers = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"]
        for brand in ["security", "support", "billing", "account", "verification", "service"]:
            if brand in local_part.lower() and domain in free_providers:
                flags.append(f"Free webmail provider ({domain}) impersonating corporate '{brand}' handle")
                is_suspicious = True
                break

        if len(local_part) >= 12 and sum(c.isdigit() for c in local_part) >= 5:
            flags.append("Randomized high-entropy address typical of automated botnet mailers")
            is_suspicious = True

    return {
        "provided": True,
        "address": addr,
        "is_suspicious": is_suspicious,
        "flags": flags
    }

def analyze_email_threats(subject, sender, body, raw_text, cleaned_text):
    combined_raw = f"{subject} {sender} {body} {raw_text}".lower()

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

    url_results = inspect_urls_for_threats(f"{subject} {body} {raw_text}")
    suspicious_urls = [u for u in url_results if u["is_suspicious"]]
    critical_urls = [u for u in url_results if u["severity"] == "CRITICAL"]

    if critical_urls:
        phishing_score += 40
    elif suspicious_urls:
        phishing_score += 20

    sender_eval = inspect_sender_security(sender)
    if sender_eval.get("is_suspicious"):
        phishing_score += 20

    currency_matches = re.findall(r'[$£€¥]\s*\d+', raw_text)
    if currency_matches:
        matched_signals.append({
            "category": "Monetary Claims",
            "label": "Direct Currency / Payout Sums",
            "severity": "MEDIUM",
            "matched": ", ".join(currency_matches[:3])
        })
        phishing_score += 10

    phishing_score = min(100, phishing_score)

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

def compute_model_probabilities(active_pipe, cleaned_text):
    pred_int = int(active_pipe.predict([cleaned_text])[0])

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
        df = float(active_pipe.decision_function([cleaned_text])[0])
        scaled_df = max(-25.0, min(25.0, df * 6.0))
        spam_prob = 1.0 / (1.0 + np.exp(-scaled_df))
        spam_pct = float(spam_prob) * 100.0
        ham_pct = 100.0 - spam_pct
    else:
        spam_pct = 95.0 if pred_int == 1 else 5.0
        ham_pct = 100.0 - spam_pct

    return pred_int, ham_pct, spam_pct

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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/health", methods=["GET"])
def health():
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
    return jsonify(BENCHMARK_SAMPLES), 200

@app.route("/api/predict", methods=["POST"])
def predict():
    start_time = time.time()

    if pipeline is None:
        load_model_and_metrics()
        if pipeline is None:
            return jsonify({
                "error": "Machine learning model pipeline is not loaded. Please execute: python train_model.py"
            }), 503

    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "error": "Invalid JSON request payload. Expected format: {'subject': '...', 'sender': '...', 'body': '...'} or {'email_text': '...'}"
        }), 400

    subject = data.get("subject", "")
    sender = data.get("sender", data.get("sender_email", ""))
    body = data.get("body", "")
    raw_email_text = data.get("email_text", "")
    model_choice = str(data.get("model", "best")).lower()

    if not isinstance(subject, str) or not isinstance(sender, str) or not isinstance(body, str):
        return jsonify({"error": "Fields 'subject', 'sender', and 'body' must be strings."}), 400

    if raw_email_text is not None and not isinstance(raw_email_text, str):
        return jsonify({"error": "Field 'email_text' must be a string."}), 400

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

    if raw_email_text and not body:
        subj_match = re.search(r'^Subject:\s*(.+)$', raw_email_text, re.MULTILINE | re.IGNORECASE)
        if subj_match and not subject:
            subject = subj_match.group(1).strip()
        full_text = raw_email_text.strip()
    else:
        parts = []
        if subject.strip():
            parts.append(f"Subject: {subject.strip()}")
        if body.strip():
            parts.append(body.strip())
        full_text = "\n\n".join(parts).strip()

    if not full_text:
        return jsonify({
            "error": "Email content cannot be empty. Please provide email body, subject, or text."
        }), 400

    if len(full_text) < 5:
        return jsonify({
            "error": "Email content is too short for accurate classification (minimum 5 characters required)."
        }), 400

    active_pipe = pipeline
    active_model_name = metrics_data.get("best_model_name", "Linear SVM")

    if model_choice in ["baseline", "naive_bayes", "multinomialnb", "nb"] and baseline_pipeline is not None:
        active_pipe = baseline_pipeline
        active_model_name = "Multinomial Naive Bayes"

    cleaned_text = clean_email_text(full_text)
    if not cleaned_text or len(cleaned_text.split()) < 1:
        cleaned_text = full_text.lower()

    try:
        pred_int, ham_prob, spam_prob = compute_model_probabilities(active_pipe, cleaned_text)
    except Exception as e:
        return jsonify({
            "error": f"Machine learning inference failed: {str(e)}"
        }), 500

    is_spam = bool(pred_int == 1)
    ml_prediction_label = "SPAM" if is_spam else "NOT SPAM"
    ml_confidence = spam_prob if is_spam else ham_prob

    threat_assessment = analyze_email_threats(
        subject=subject,
        sender=sender,
        body=body,
        raw_text=full_text,
        cleaned_text=cleaned_text
    )

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
        risk_level = "SAFE"
        risk_color = "#10B981"
        verdict_summary = "Low-risk communication matching legitimate correspondence patterns."
        security_action = "Standard reading safe, but zero detected indicators does NOT guarantee safety. Exercise vigilance."

    detected_signals = threat_assessment["threat_signals"]

    for s_url in threat_assessment["url_analysis"]["suspicious_urls"]:
        for reason in s_url["reasons"]:
            detected_signals.append({
                "type": "url",
                "label": f"Suspicious Link: {s_url['host']}",
                "matched": reason
            })

    inference_ms = round((time.time() - start_time) * 1000, 2)

    safety_disclaimer = (
        "Defense-in-Depth Notice: Machine learning models and heuristic rules assess statistical "
        "and lexical indicators. A classification of 'NOT SPAM' or 'LOW RISK' is NOT a guarantee "
        "that an email is safe. Sophisticated spear-phishing and zero-day attacks may evade detection. "
        "Always verify unexpected requests through trusted out-of-band channels."
    )

    response_payload = {
        "success": True,
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
        "model_classification": {
            "model_name": active_model_name,
            "prediction": ml_prediction_label,
            "is_spam": is_spam,
            "confidence": round(ml_confidence, 2),
            "spam_probability": round(spam_prob, 2),
            "ham_probability": round(ham_prob, 2)
        },
        "phishing_assessment": threat_assessment,
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

load_model_and_metrics()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting server at http://{host}:{port}")
    app.run(host=host, port=port, debug=True)
