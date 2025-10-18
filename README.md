# SpamShield API


Our model achieves >99% precision and recall on standard SMS spam detection
tasks but struggles with message types not present in the training distribution
(e.g., extortion or phishing attempts). This illustrates a key limitation of
linear models with TF-IDF features: they rely heavily on vocabulary overlap and
cannot capture context or intent. A natural extension would be to incorporate
modern phishing data or experiment with contextual embedding models
(e.g., BERT) to improve robustness.
