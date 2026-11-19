"""Применяет ручные исправления меток.
Берёт все файлы data/fixes*.csv (колонки: text, label_fixed, comment) и обновляет
колонку label в data/labeled.csv. label_auto (метка по звёздам) не трогается.
label_fixed: positive / neutral / negative / drop (drop - выкинуть отзыв из датасета).
Пустой label_fixed = метка по рейтингу верна, ничего не меняем.
Запуск: python -m ml.apply_fixes"""
import glob
import pandas as pd

VALID = {"positive", "neutral", "negative", "drop"}


def main(labeled="data/labeled.csv") -> None:
    df = pd.read_csv(labeled)
    df["label"] = df["label_auto"]            # всегда пересчитываем с нуля - безопасно запускать повторно

    files = sorted(glob.glob("data/fixes*.csv"))
    if not files:
        print("Файлов data/fixes*.csv нет - ничего не исправлено")
        df.to_csv(labeled, index=False)
        return
    fixes = pd.concat([pd.read_csv(f) for f in files])
    fixes = fixes.dropna(subset=["text", "label_fixed"])
    fixes["label_fixed"] = fixes["label_fixed"].str.strip().str.lower()

    wrong = fixes[~fixes["label_fixed"].isin(VALID)]
    if len(wrong):
        print("Недопустимые значения label_fixed (пропущены):")
        print(wrong[["text", "label_fixed"]].head())
    fixes = fixes[fixes["label_fixed"].isin(VALID)].drop_duplicates("text", keep="last")
    mapping = dict(zip(fixes["text"], fixes["label_fixed"]))

    fixed = df["text"].map(mapping)
    changed = fixed.notna() & (fixed != "drop") & (fixed != df["label_auto"])
    dropped = fixed == "drop"
    df.loc[fixed.notna() & ~dropped, "label"] = fixed[fixed.notna() & ~dropped]
    df = df[~dropped]

    df.to_csv(labeled, index=False)
    print(f"Файлов с исправлениями: {len(files)}; правок меток: {int(changed.sum())}; "
          f"удалено отзывов: {int(dropped.sum())}; в датасете осталось: {len(df)}")


if __name__ == "__main__":
    main()
