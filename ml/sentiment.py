import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ml.preprocess import clean_text

MODEL_NAME = "seara/rubert-tiny2-russian-sentiment"
MAX_LEN = 256
_FALLBACK = {0: "neutral", 1: "positive", 2: "negative"}  # порядок из карточки модели
LABEL_RU = {"positive": "позитивный", "neutral": "нейтральный", "negative": "негативный"}

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def _label(model, idx: int) -> str:
    name = model.config.id2label.get(idx, "")
    return name if name in LABEL_RU else _FALLBACK[idx]


def predict_batch(texts: list[str], batch_size: int = 32) -> list[tuple[str, float]]:
    """Для файлов с отзывами (CSV/TXT) - быстрее, чем по одному."""
    tok, model = _load()
    cleaned = [clean_text(t) for t in texts]
    results = []
    for i in range(0, len(cleaned), batch_size):
        chunk = cleaned[i:i + batch_size]
        enc = tok([c or "." for c in chunk], return_tensors="pt",
                  padding=True, truncation=True, max_length=MAX_LEN)
        with torch.inference_mode():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        for c, cf, ix in zip(chunk, conf.tolist(), idx.tolist()):
            results.append(("neutral", 0.0) if not c else (_label(model, ix), round(cf, 4)))
    return results


def predict(text: str) -> tuple[str, float]:
    """Возвращает (класс, уверенность), например ("negative", 0.94).
    Классы: "positive" / "neutral" / "negative"."""
    return predict_batch([text])[0]