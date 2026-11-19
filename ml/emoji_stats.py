"""Какие эмодзи встречаются в датасете, каких нет в словаре и как они связаны с классом.
Запуск: python -m ml.emoji_stats data/labeled.csv"""
import sys
from collections import Counter
import emoji
import pandas as pd
from ml.preprocess import EMOJI_WORDS, normalize_emoji


def main(path: str) -> None:
    df = pd.read_csv(path).dropna(subset=["text"])
    label_col = "label" if "label" in df.columns else "label_auto"

    counts = Counter()                      # сколько раз встретился
    by_class = {}                           # в каких классах
    for text, label in zip(df["text"], df[label_col]):
        for item in emoji.emoji_list(text):
            key = normalize_emoji(item["emoji"])
            counts[key] += 1
            by_class.setdefault(key, Counter())[label] += 1

    rows = []
    for key, n in counts.most_common():
        c = by_class[key]
        rows.append({
            "emoji": key, "встреч": n,
            "в_словаре": "да" if (key in EMOJI_WORDS or key[:1] in EMOJI_WORDS) else "НЕТ",
            "negative%": round(100 * c["negative"] / n),
            "neutral%": round(100 * c["neutral"] / n),
            "positive%": round(100 * c["positive"] / n),
        })
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    print(f"Всего уникальных эмодзи: {len(res)}\n")
    print("=== Частые эмодзи, которых НЕТ в словаре (добавляйте первыми) ===")
    print(res[res["в_словаре"] == "НЕТ"].head(25).to_string(index=False))
    print("\n=== 25 самых частых эмодзи и их связь с классом ===")
    print(res.head(25).to_string(index=False))
    res.to_csv("data/emoji_stats.csv", index=False)
    print("\nПолная таблица: data/emoji_stats.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/labeled.csv")
