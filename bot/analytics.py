import statistics
from collections import Counter

SCORE = {"negative": -1, "neutral": 0, "positive": 1}
ORDER = ("positive", "neutral", "negative")


def build_report(results, ratings, label_ru):
    """results: [(текст, класс, уверенность)], ratings: [число или None]."""
    total = len(results)
    counts = Counter(label for _, label, _ in results)
    lines = [f"Всего отзывов: {total}", "", "Соотношение:"]
    for label in ORDER:
        n = counts.get(label, 0)
        lines.append(f"  {label_ru.get(label, label)}: {n} ({n / total:.0%})")

    confs = [c for _, _, c in results]
    lengths = [len(t) for t, _, _ in results]
    low = sum(c < 0.6 for c in confs)
    lines += [
        "",
        "Статистика:",
        f"  средняя уверенность модели: {statistics.mean(confs):.0%}",
        f"  средняя длина отзыва: {statistics.mean(lengths):.0f} симв.",
        f"  отзывов с низкой уверенностью (<60%): {low}",
    ]

    scores = [SCORE.get(label, 0) for _, label, _ in results]
    pairs = [(r, s) for r, s in zip(ratings, scores) if r is not None]
    if len(pairs) >= 3:
        xs, ys = zip(*pairs)
        name = "оценка и тональность"
    else:
        xs, ys = lengths, scores
        name = "длина отзыва и тональность"
    try:
        corr = f"{statistics.correlation(xs, ys):+.2f}"
    except statistics.StatisticsError:
        corr = "не определена (нет разброса данных)"
    lines += ["", f"Корреляция ({name}): {corr}"]
    return "\n".join(lines)