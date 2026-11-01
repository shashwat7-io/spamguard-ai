"""
=============================================================================
SPAMGUARD AI - AUTOMATED TEST SUITE
=============================================================================
Author: SPAMGUARD AI QA & ML Test Engineering
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


class SpamGuardTestCase(unittest.TestCase):

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
        self.assertIn(b"SPAMGUARD", response.data)


if __name__ == "__main__":
    unittest.main()
