class SpamModel:
    def __init__(self, model, metadata):
        self._pipeline = model
        self._metadata = metadata
        self._hash = metadata["model_sha256"]
        self._threshold = metadata["threshold"]

    @property
    def version(self) -> str:
        return f"1.0.0+{self._metadata['model_sha256']}"

    def predict(self, texts: list[str]) -> list[tuple[str, float]]:
        proba = self._pipeline.predict_proba(texts)
        spam_idx = int(self._pipeline.classes_.tolist().index("spam"))
        out = []
        for p in proba:
            prob_spam = float(p[spam_idx])
            label = "spam" if prob_spam >= self._threshold else "ham"
            out.append((label, prob_spam))
        return out
