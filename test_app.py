"""
=============================================================================
SPAM EMAIL DETECTION - AUTOMATED TEST SUITE
=============================================================================
Author: Spam Email Detection QA & ML Test Engineering
Description: Unit and integration tests covering:
             - Backend health & status endpoints
             - Empirical model metrics validation
             - True positive (Spam & Phishing) email predictions
             - True negative (Legitimate & Ham) email predictions
             - Input validation (empty, whitespace, short strings, invalid payloads)
             - Edge cases (special characters, numbers, URLs)
=============================================================================
"""

import unittest
import json
from app import app


class SpamEmailDetectionTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Configure test client."""
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_01_health_endpoint(self):
        """Verify GET /api/health returns 200 and indicates model is loaded."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertTrue(data["metrics_available"])

    def test_02_metrics_endpoint(self):
        """Verify GET /api/metrics delivers accurate test metrics."""
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("metrics", data)
        self.assertIn("accuracy_percent", data["metrics"])
        self.assertGreater(data["metrics"]["accuracy_percent"], 95.0)
        self.assertIn("confusion_matrix", data)
        self.assertIn("top_spam_keywords", data)

    def test_03_samples_endpoint(self):
        """Verify GET /api/samples provides curated email samples."""
        response = self.client.get("/api/samples")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("phishing", data)
        self.assertIn("lottery", data)
        self.assertIn("business_meeting", data)
        self.assertIn("release_notes", data)

    def test_04_validation_empty_input(self):
        """Verify empty email text triggers HTTP 400 validation error."""
        response = self.client.post("/api/predict", json={"email_text": ""})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)
        self.assertIn("empty", data["error"].lower())

    def test_05_validation_whitespace_input(self):
        """Verify whitespace-only string triggers HTTP 400 validation error."""
        response = self.client.post("/api/predict", json={"email_text": "     \n\t   "})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_06_validation_short_input(self):
        """Verify string shorter than 5 chars triggers HTTP 400 error."""
        response = self.client.post("/api/predict", json={"email_text": "Hi!"})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("too short", data["error"].lower())

    def test_07_validation_missing_key(self):
        """Verify missing 'email_text' key triggers HTTP 400 error."""
        response = self.client.post("/api/predict", json={"wrong_key": "some text"})
        self.assertEqual(response.status_code, 400)

    def test_08_predict_urgent_phishing(self):
        """Verify urgent banking credential theft email is classified as SPAM."""
        email_content = (
            "Subject: Urgent: Verify Your Bank Credentials Immediately!\n\n"
            "Dear Customer, your bank account access has been restricted due to suspicious logins. "
            "Click here immediately: http://login-secure-verification99.info to verify your password, "
            "SSN and account PIN, or your card will be deactivated permanently."
        )
        response = self.client.post("/api/predict", json={"email_text": email_content})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["prediction"], "SPAM")
        self.assertTrue(data["is_spam"])
        self.assertGreater(data["spam_probability"], 50.0)
        self.assertIn(data["risk_level"], ["HIGH", "CRITICAL"])
        self.assertGreater(len(data["detected_signals"]), 0)

    def test_09_predict_lottery_scam(self):
        """Verify classic international lottery sweepstakes email is classified as SPAM."""
        email_content = (
            "Subject: CONGRATULATIONS! You Won $1,000,000 Cash Prize\n\n"
            "You are the lucky winner of our international sweepstakes draw! "
            "Send your bank wire transfer details and full address to claim your $1,000,000 cash payout today."
        )
        response = self.client.post("/api/predict", json={"email_text": email_content})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["prediction"], "SPAM")
        self.assertTrue(data["is_spam"])
        self.assertGreater(data["spam_probability"], 50.0)

    def test_10_predict_legitimate_work_meeting(self):
        """Verify regular workplace sync email is classified as NOT SPAM."""
        email_content = (
            "Subject: Agenda for Architecture Review - Tuesday at 10:00 AM\n\n"
            "Hi Alex,\n\n"
            "Attached is the technical design specification for the search service refactor. "
            "Let's review the Redis caching layer and database indexing strategies during tomorrow's team sync. "
            "Please add any discussion topics to the shared Google doc.\n\n"
            "Thanks,\nDavid"
        )
        response = self.client.post("/api/predict", json={"email_text": email_content})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["prediction"], "NOT SPAM")
        self.assertFalse(data["is_spam"])
        self.assertGreater(data["ham_probability"], 50.0)
        self.assertEqual(data["risk_level"], "SAFE")

    def test_11_predict_legitimate_release_notes(self):
        """Verify project release notes message is classified as NOT SPAM."""
        email_content = (
            "Subject: Documentation and User Guide Updates for Version 2.4 Milestone\n\n"
            "Hi Team,\n\n"
            "Please review the updated technical documentation and user guide for our upcoming 2.4 milestone release. "
            "All API parameter definitions and error code tables have been finalized. "
            "Let me know if any sections need clarification before we publish to the knowledge base tomorrow morning.\n\n"
            "Best regards,\n"
            "Alex Chen"
        )
        response = self.client.post("/api/predict", json={"email_text": email_content})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["prediction"], "NOT SPAM")
        self.assertFalse(data["is_spam"])
        self.assertGreater(data["ham_probability"], 50.0)

    def test_12_predict_suspicious_mixed_content(self):
        """Verify suspicious email with mixed signals is processed with valid risk level."""
        email_content = (
            "Subject: Special Opportunity: Exclusive free consultation for your team\n\n"
            "Hello,\n"
            "We are offering a free trial of our productivity software for qualified accounts. "
            "Click http://partner-offers-network.org/trial to review details. "
            "If you do not wish to receive future mailings, click unsubscribe below."
        )
        response = self.client.post("/api/predict", json={"email_text": email_content})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("prediction", data)
        self.assertIn("risk_level", data)
        self.assertIn(data["risk_level"], ["MEDIUM", "HIGH", "CRITICAL"])
        self.assertGreater(len(data["detected_signals"]), 0)

    def test_13_ui_route_available(self):
        """Verify GET / renders HTML template with 200 OK."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"SPAM EMAIL", response.data)

    def test_14_predict_multi_field_payload(self):
        """Verify prediction using separate subject, sender, and body fields."""
        payload = {
            "subject": "Action Required: Confirm your payroll deposit information",
            "sender": "payroll-verification@internal-hr-portal.xyz",
            "body": "Dear Employee, please verify your direct deposit routing number and password at http://192.168.1.50/payroll or your salary payout will be delayed."
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["prediction"], "SPAM")
        self.assertTrue(data["is_spam"])
        self.assertIn("phishing_assessment", data)
        self.assertTrue(data["phishing_assessment"]["phishing_detected"])
        self.assertIn("input_summary", data)
        self.assertTrue(data["input_summary"]["has_subject"])
        self.assertTrue(data["input_summary"]["has_sender"])

    def test_15_suspicious_url_detection(self):
        """Verify suspicious URL detection (IP hostname, abused TLDs, and shorteners)."""
        payload = {
            "subject": "Invoice Follow-Up",
            "sender": "billing@quick-invoices.top",
            "body": "Please view your overdue invoice here: http://203.0.113.15/download and backup mirror http://tinyurl.com/invoice-xyz"
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        phishing = data["phishing_assessment"]
        self.assertGreater(phishing["url_analysis"]["suspicious_urls_count"], 0)
        reasons_concat = " ".join([
            " ".join(u["reasons"]) for u in phishing["url_analysis"]["suspicious_urls"]
        ])
        self.assertTrue(
            "Direct IP-based URL" in reasons_concat or "URL shortener" in reasons_concat or "Abused" in reasons_concat
        )

    def test_16_non_guarantee_safety_disclaimer(self):
        """Verify safety_guarantee is False and a clear defense-in-depth disclaimer is included."""
        payload = {
            "subject": "Quarterly Planning Meeting Notes",
            "sender": "alice@company.com",
            "body": "Here are the team action items from this morning's planning session. Please review before Friday."
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertFalse(data["safety_guarantee"])
        self.assertIn("safety_disclaimer", data)
        self.assertIn("NOT a guarantee", data["safety_disclaimer"])

    def test_17_input_length_limits(self):
        """Verify oversized payloads trigger HTTP 400 validation errors."""
        # Body over 100,000 characters
        oversized_body = "A" * 100005
        response = self.client.post("/api/predict", json={"body": oversized_body})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("exceeds maximum allowed size", data["error"])

        # Subject over 1,000 characters
        oversized_subj = "S" * 1005
        response = self.client.post("/api/predict", json={"subject": oversized_subj, "body": "Normal body"})
        self.assertEqual(response.status_code, 400)

    def test_18_multi_model_metrics_structure(self):
        """Verify /api/metrics provides comparisons across all 4 models and dataset statistics."""
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("model_comparison", data)
        self.assertEqual(len(data["model_comparison"]), 4)

        model_names = [m["model_name"] for m in data["model_comparison"]]
        self.assertIn("Multinomial Naive Bayes", model_names)
        self.assertIn("Logistic Regression", model_names)
        self.assertIn("Random Forest", model_names)
        self.assertIn("Linear SVM", model_names)

        self.assertIn("best_model_name", data)
        self.assertIn("baseline_model_name", data)
        self.assertEqual(data["baseline_model_name"], "Multinomial Naive Bayes")
        self.assertEqual(data["total_dataset_samples"], 4235)
        self.assertEqual(data["train_samples"], 3388)
        self.assertEqual(data["test_samples"], 847)

    def test_19_model_selection_parameter(self):
        """Verify selecting baseline vs best model works in prediction API."""
        payload = {
            "email_text": "Subject: Important update\n\nPlease find the project timeline attached.",
            "model": "baseline"
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["model_classification"]["model_name"], "Multinomial Naive Bayes")

    def test_20_distinct_phishing_vs_spam_indicators(self):
        """Verify clear distinction between ML spam classification and heuristic phishing indicators."""
        payload = {
            "subject": "CRITICAL: Suspicious activity on your account",
            "sender": "security-support@paypal-security-alert.xyz",
            "body": "Your account was temporarily suspended. Click http://192.168.0.1/verify-login to reactivate immediately."
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        # Both ML classification and Phishing assessment are present and structured
        self.assertIn("model_classification", data)
        self.assertIn("phishing_assessment", data)
        self.assertIn("phishing_score", data["phishing_assessment"])
        self.assertIn("threat_signals", data["phishing_assessment"])
        self.assertGreater(len(data["phishing_assessment"]["threat_signals"]), 0)


if __name__ == "__main__":
    unittest.main()
