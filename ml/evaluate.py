"""Считает Accuracy, F1-macro, матрицу ошибок и скорость на размеченном CSV.
Запуск: python -m ml.evaluate data/labeled.csv"""
import sys
import time
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from ml.sentiment import predict, predict_batch

LABELS = ["negative", "neutral", "positive"]


def report(y_true, y_pred, title: str) -> None:
    print(f"--- {title} ---")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.3f}")
    print(f"F1-macro : {f1_score(y_true, y_pred, average='macro'):.3f}")


def main(path: str) -> None:
    df = pd.read_csv(path).dropna(subset=["text", "label"])
    df["label"] = df["label"].str.strip().str.lower()

    preds = predict_batch(df["text"].tolist())           # быстро, пачками
    df["pred"] = [p[0] for p in preds]
    df["conf"] = [p[1] for p in preds]

    print(f"Примеров: {len(df)}\n")
    if "label_auto" in df.columns:
        report(df["label_auto"], df["pred"], "по звёздам (label_auto), до ручной проверки")
    report(df["label"], df["pred"], "по итоговым меткам (label)")
    print()
    print(classification_report(df["label"], df["pred"], labels=LABELS, zero_division=0))
    print("Матрица ошибок (строки - истина, столбцы - прогноз):")
    print(pd.DataFrame(confusion_matrix(df["label"], df["pred"], labels=LABELS),
                       index=LABELS, columns=LABELS))

    # грубые ошибки: негатив <-> позитив
    wrong = df[df["label"] != df["pred"]]
    gross = wrong[((wrong["label"] == "negative") & (wrong["pred"] == "positive")) |
                  ((wrong["label"] == "positive") & (wrong["pred"] == "negative"))]
    if len(wrong):
        print(f"\nГрубых ошибок (негатив<->позитив): {len(gross)} "
              f"из {len(wrong)} ошибок = {100 * len(gross) / len(wrong):.1f}% (цель < 3%)")

    # скорость одного отзыва (так, как её будет вызывать бот)
    sample = df["text"].sample(min(100, len(df)), random_state=1).tolist()
    start = time.perf_counter()
    for t in sample:
        predict(t)
    print(f"Скорость одного отзыва: {(time.perf_counter() - start) / len(sample) * 1000:.0f} мс (цель до 120 мс)")

    # 1) ошибки для ручного разбора: сначала те, где модель уверена
    cols = ["text", "rating", "label_auto", "label", "pred", "conf"]
    cols = [c for c in cols if c in df.columns]
    errors = wrong.sort_values("conf", ascending=False)[cols].copy()
    errors["label_fixed"] = ""    # заполняете вручную
    errors["comment"] = ""
    errors.to_csv("data/errors.csv", index=False)

    # 2) случайные 10% из верных предсказаний - выборочная проверка, что метки по звёздам не врут
    ok = df[df["label"] == df["pred"]]
    spot = ok.sample(min(len(ok), max(30, len(ok) // 10)), random_state=7)[cols].copy()
    spot["label_fixed"] = ""
    spot["comment"] = ""
    spot.to_csv("data/spot_check.csv", index=False)
    print(f"\nОшибок: {len(errors)} -> data/errors.csv; выборка верных для проверки: {len(spot)} -> data/spot_check.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/labeled.csv")
