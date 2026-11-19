"""Собирает data/labeled.csv из Geo Reviews (Яндекс Карты).
Метка label_auto ставится по звёздам, label = label_auto (потом правится вручную
через data/fixes*.csv и ml/apply_fixes.py)."""
import pandas as pd
from ml.preprocess import has_emoji

N_PER_CLASS = 1500     # сколько отзывов в каждом классе (3 класса -> 4500 всего)
EMOJI_SHARE = 0.2      # желаемая доля отзывов с эмодзи в классе
MIN_LEN, MAX_LEN = 20, 500
SEED = 42


def rating_to_label(r: float) -> str:
    if r <= 2:
        return "negative"
    if r == 3:
        return "neutral"
    return "positive"


def build_dataset(df: pd.DataFrame, n_per_class: int = N_PER_CLASS,
                  emoji_share: float = EMOJI_SHARE, seed: int = SEED) -> pd.DataFrame:
    df = df.dropna(subset=["text", "rating"]).copy()
    df = df[df["rating"].between(1, 5)]
    df = df[df["text"].str.len().between(MIN_LEN, MAX_LEN)]
    df = df.drop_duplicates(subset="text")
    df["label_auto"] = df["rating"].map(rating_to_label)
    df["has_emoji"] = df["text"].map(has_emoji)

    parts = []
    for label, g in df.groupby("label_auto"):
        with_e, without_e = g[g["has_emoji"]], g[~g["has_emoji"]]
        n_e = min(round(n_per_class * emoji_share), len(with_e))
        n_ne = min(n_per_class - n_e, len(without_e))
        if n_e + n_ne < n_per_class:
            print(f"ВНИМАНИЕ: в классе {label} нашлось только {n_e + n_ne} из {n_per_class}")
        parts += [with_e.sample(n_e, random_state=seed),
                  without_e.sample(n_ne, random_state=seed)]

    out = pd.concat(parts).sample(frac=1, random_state=seed)
    out["label"] = out["label_auto"]
    return out[["text", "rating", "label_auto", "label", "has_emoji"]]


if __name__ == "__main__":
    from datasets import load_dataset
    ds = load_dataset("d0rj/geo-reviews-dataset-2023", split="train")  # один раз скачается и кэшируется
    df = ds.to_pandas()
    print("Колонки:", df.columns.tolist())   # если названия другие - поправьте "text"/"rating"
    out = build_dataset(df)
    out.to_csv("data/labeled.csv", index=False)
    print(out["label_auto"].value_counts())
    print("С эмодзи:", out["has_emoji"].sum(), "из", len(out))
