"""Готовит data/labeled.csv из Geo Reviews: метки по рейтингу, часть отзывов с эмодзи."""
import pandas as pd
from datasets import load_dataset
from ml.preprocess import has_emoji

N_PER_CLASS = 60     # 3 класса -> 180 отзывов
EMOJI_SHARE = 0.4    # доля отзывов с эмодзи в каждом классе


def rating_to_label(r: float) -> str:
    if r <= 2:
        return "negative"
    if r == 3:
        return "neutral"
    return "positive"


ds = load_dataset("d0rj/geo-reviews-dataset-2023", split="train", streaming=True)
ds = ds.shuffle(seed=42, buffer_size=10_000)

rows = []
for i, row in enumerate(ds):
    if i >= 30_000:
        break
    rows.append(row)

df = pd.DataFrame(rows)
print("Колонки:", df.columns.tolist())   # если названия другие - поправьте "text"/"rating" ниже

df = df[df["rating"].between(1, 5)]
df = df[df["text"].str.len().between(20, 400)]
df["label"] = df["rating"].map(rating_to_label)
df["has_emoji"] = df["text"].map(has_emoji)

parts = []
for _, g in df.groupby("label"):
    with_e, without_e = g[g.has_emoji], g[~g.has_emoji]
    n_e = min(int(N_PER_CLASS * EMOJI_SHARE), len(with_e))
    parts.append(with_e.sample(n_e, random_state=42))
    parts.append(without_e.sample(N_PER_CLASS - n_e, random_state=42))

out = pd.concat(parts).sample(frac=1, random_state=42)[["text", "rating", "label"]]
out.to_csv("data/labeled.csv", index=False)
print(out["label"].value_counts())