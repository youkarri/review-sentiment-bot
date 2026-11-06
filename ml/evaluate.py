"""Считает Accuracy, F1-macro, матрицу ошибок и скорость на размеченном CSV.
Запуск: python -m ml.evaluate data/labeled.csv"""
import sys
import time
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from ml.sentiment import predict

LABELS = ["negative", "neutral", "positive"]


def main(path: str) -> None:
    df = pd.read_csv(path).dropna(subset=["text", "label"])
    df["label"] = df["label"].str.strip().str.lower()

    start = time.perf_counter()
    preds = [predict(t) for t in df["text"]]
    elapsed = time.perf_counter() - start

    df["pred"] = [p[0] for p in preds]
    df["conf"] = [p[1] for p in preds]

    print(f"Примеров: {len(df)}")
    print(f"Accuracy : {accuracy_score(df['label'], df['pred']):.3f}")
    print(f"F1-macro : {f1_score(df['label'], df['pred'], average='macro'):.3f}")
    print(f"Скорость : {elapsed / len(df) * 1000:.0f} мс на отзыв\n")
    print(classification_report(df["label"], df["pred"], labels=LABELS, zero_division=0))
    print("Матрица ошибок (строки - истина, столбцы - прогноз):")
    print(pd.DataFrame(confusion_matrix(df["label"], df["pred"], labels=LABELS),
                       index=LABELS, columns=LABELS))

    # ошибки сохраняем, чтобы разобрать вручную
    df[df["label"] != df["pred"]].to_csv("data/errors.csv", index=False)
    print("\nОшибки модели сохранены в data/errors.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/labeled.csv")