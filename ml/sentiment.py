from transformers import pipeline
from typing import Tuple
import re

# Загружаем модель один раз
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="seara/rubert-tiny2-russian-sentiment",
    tokenizer="seara/rubert-tiny2-russian-sentiment",
    device=-1  # CPU
)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Убираем ссылки
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Убираем эмодзи
    text = re.sub(
        r'[\U0001F600-\U0001F64F'
        r'\U0001F300-\U0001F5FF'
        r'\U0001F680-\U0001F6FF'
        r'\U0001F1E0-\U0001F1FF'
        r'\U00002702-\U000027B0'
        r'\U000024C2-\U0001F251'
        r'\U0001F900-\U0001F9FF'
        r'\U0001FA00-\U0001FA6F'
        r'\U0001FA70-\U0001FAFF'
        r'\U00002600-\U000026FF]+',
        '',
        text
    )

    # Лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()

    # Нижний регистр
    text = text.lower()

    # Повторяющиеся знаки !!! → !
    text = re.sub(r'([!?.]){2,}', r'\1', text)

    return text

def predict(text: str) -> Tuple[str, float]:
    """
    Возвращает (класс, уверенность)
    классы: positive | negative | neutral
    """
    text = clean_text(text)
    if not text:
        return "neutral", 0.0

    result = sentiment_pipeline(text[:512])[0]
    return result["label"], float(result["score"])

if __name__ == "__main__":
    test_reviews = [
        "Отличный товар, очень доволен покупкой! Рекомендую всем",
        "Ужасное качество, сломалось через неделю. Деньги на ветер",
        "Нормально, ничего особенного. Как и ожидал.",
        "Доставка быстрая, упаковка целая, товар соответствует описанию.",
        "Не рекомендую! Обман и развод.",
        "Хороший сервис, вежливые менеджеры.",
        "Средненько. Есть плюсы и минусы.",
        "Лучший магазин в городе, всегда беру только здесь!",
        "Товар пришёл бракованный, возврат отказали.",
        "Всё супер, спасибо продавцу!",
    ]

    print("Результаты:")
    for i, review in enumerate(test_reviews, 1):
        label, conf = predict(review)
        print(f"{i:2d}. [{label:8s} {conf:.3f}] {review}")
