"""Аналитика по набору отзывов: чтение CSV/TXT, статистика, частые темы, текст ответа, диаграмма."""
import csv
import io
from collections import Counter
from functools import lru_cache
from pathlib import Path
import re

from ml.preprocess import clean_text, has_emoji

try:                                   # лемматизация нужна, чтобы "доставка/доставку/доставки" считались одним словом
    import pymorphy3
    _morph = pymorphy3.MorphAnalyzer()
except Exception:                      # без pymorphy3 всё работает, просто темы чуть грубее
    _morph = None

ORDER = ["positive", "neutral", "negative"]
TITLES = {"positive": "позитивных", "neutral": "нейтральных", "negative": "негативных"}
MAX_REVIEWS = 5000
TEXT_COLUMNS = {"text", "review", "reviews", "comment", "comments",
                "отзыв", "отзывы", "текст", "комментарий", "комментарии"}

STOPWORDS = set("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему
теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут
где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой
совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть после
над больше тот всё через эти нас про всего них всё какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя
такой им более всегда конечно всю между это очень просто вообще пока наш ваш свой такие этих тех мои мою
""".split())
NOISE_NOUNS = {"раз", "день", "год", "человек", "вещь", "отзыв", "штука", "время", "мой", "весь"}
_TOKEN = re.compile(r"[а-яё]{3,}")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


# ---------------------------------------------------------------- чтение файлов
def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Не удалось определить кодировку файла (нужен UTF-8 или Windows-1251)")


def load_reviews_csv(path) -> list[str]:
    text = _decode(Path(path).read_bytes())
    if not text.strip():
        raise ValueError("Файл пустой")
    try:
        dialect = csv.Sniffer().sniff(text[:5000], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(io.StringIO(text), dialect) if r]
    if not rows:
        raise ValueError("Файл пустой")

    header = [c.strip().lower() for c in rows[0]]
    idx = next((i for i, c in enumerate(header) if c in TEXT_COLUMNS), None)
    if idx is not None:                                   # нашли колонку по названию
        data = rows[1:]
    elif len(rows[0]) == 1:                               # одна колонка без заголовка
        idx, data = 0, rows
    else:                                                 # берём колонку с самым длинным текстом
        data = rows[1:]
        width = max(len(r) for r in data) if data else 0
        avg = [sum(len(r[i]) for r in data[:50] if len(r) > i) for i in range(width)]
        idx = avg.index(max(avg)) if avg else 0

    texts = [r[idx].strip() for r in data if len(r) > idx and r[idx].strip()]
    if not texts:
        raise ValueError("В файле не нашлось текстов отзывов")
    return texts[:MAX_REVIEWS]


def load_reviews_txt(path) -> list[str]:
    lines = [ln.strip() for ln in _decode(Path(path).read_bytes()).splitlines()]
    texts = [ln for ln in lines if ln]
    if not texts:
        raise ValueError("Файл пустой")
    return texts[:MAX_REVIEWS]


def load_reviews(path) -> list[str]:
    """Одна точка входа для бота: CSV или TXT (один отзыв на строку)."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return load_reviews_csv(path)
    if suffix == ".txt":
        return load_reviews_txt(path)
    raise ValueError("Поддерживаются только файлы .csv и .txt")


# ---------------------------------------------------------------- темы и частые фразы
@lru_cache(maxsize=100_000)
def _lemma(word: str) -> tuple[str, str | None]:
    if _morph is None:
        return word, None
    p = _morph.parse(word)[0]
    return p.normal_form, p.tag.POS


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(clean_text(text, emoji_mode="drop"))


def _clauses(text: str) -> list[list[str]]:
    """Текст -> список фрагментов между знаками препинания (в каждом свои слова)."""
    parts = re.split(r"[.,;:!?()\n\-—]+", clean_text(text, emoji_mode="drop"))
    return [c for c in (_TOKEN.findall(part) for part in parts) if c]


def top_topics(texts: list[str], top_n: int = 5) -> list[tuple[str, int]]:
    """Самые частые существительные (в скольких отзывах встретились)."""
    counter = Counter()
    for t in texts:
        seen = set()
        for w in _tokens(t):
            lemma, pos = _lemma(w)
            if w in STOPWORDS or lemma in NOISE_NOUNS:
                continue
            if _morph is not None and pos != "NOUN":
                continue
            seen.add(lemma)
        counter.update(seen)
    return counter.most_common(top_n)


def top_phrases(texts: list[str], top_n: int = 5, min_count: int = 2) -> list[tuple[str, int]]:
    """Частые двухсловные сочетания ("долгая доставка")."""
    counter, surface = Counter(), {}
    for t in texts:
        seen = set()
        for toks in _clauses(t):
            for a, b in zip(toks, toks[1:]):
                if a in STOPWORDS or b in STOPWORDS:
                    continue
                key = (_lemma(a)[0], _lemma(b)[0])
                seen.add(key)
                surface.setdefault(key, Counter())[f"{a} {b}"] += 1
        counter.update(seen)
    return [(surface[k].most_common(1)[0][0], n)
            for k, n in counter.most_common(top_n) if n >= min_count]


def frequent_comments(texts: list[str], top_n: int = 3, min_count: int = 2) -> list[tuple[str, int]]:
    """Одинаковые отзывы, которые повторяются ("Спасибо", "Всё отлично")."""
    counter = Counter()
    for t in texts:
        norm = _PUNCT.sub("", clean_text(t, emoji_mode="drop")).strip()
        if 0 < len(norm) <= 100:
            counter[norm] += 1
    return [(k, n) for k, n in counter.most_common(top_n) if n >= min_count]


# ---------------------------------------------------------------- главный расчёт
def analyze(texts: list[str], preds: list[tuple[str, float]] | None = None) -> dict:
    if not texts:
        raise ValueError("Нет отзывов для анализа")
    if preds is None:
        from ml.sentiment import predict_batch
        preds = predict_batch(texts)

    labels = [p[0] for p in preds]
    confs = [p[1] for p in preds]
    total = len(texts)
    cnt = Counter(labels)
    by_label = {l: [t for t, lb in zip(texts, labels) if lb == l] for l in ORDER}
    neg_sorted = sorted(((c, t) for t, lb, c in zip(texts, labels, confs) if lb == "negative"),
                        reverse=True)
    neg_examples, seen_ex = [], set()
    for c, t in neg_sorted:                       # без повторов
        if t not in seen_ex:
            seen_ex.add(t)
            neg_examples.append((c, t))
        if len(neg_examples) == 3:
            break

    return {
        "total": total,
        "counts": {l: cnt.get(l, 0) for l in ORDER},
        "percent": {l: round(100 * cnt.get(l, 0) / total, 1) for l in ORDER},
        "pos_neg_ratio": round(cnt.get("positive", 0) / cnt["negative"], 2) if cnt.get("negative") else None,
        "avg_confidence": round(sum(confs) / total, 2),
        "low_confidence_share": round(100 * sum(c < 0.6 for c in confs) / total, 1),
        "with_emoji_share": round(100 * sum(has_emoji(t) for t in texts) / total, 1),
        "topics": {l: top_topics(by_label[l]) for l in ("positive", "negative")},
        "phrases": {l: top_phrases(by_label[l]) for l in ("positive", "negative")},
        "frequent_comments": frequent_comments(texts),
        "negative_examples": [t[:150] for _, t in neg_examples],
    }


def format_report(r: dict) -> str:
    """Текст ответа для Telegram (до 4096 символов)."""
    lines = [f"📊 Проанализировано отзывов: {r['total']}", ""]
    for l in ORDER:
        lines.append(f"{'🟢🟡🔴'[ORDER.index(l)]} {TITLES[l].capitalize()}: {r['counts'][l]} ({r['percent'][l]}%)")
    ratio = r["pos_neg_ratio"]
    lines.append(f"\n⚖️ Позитивных к негативным: {f'{ratio} : 1' if ratio is not None else 'негативных нет'}")

    def block(title, items, suffix):
        if items:
            lines.append(f"\n{title}")
            lines.extend(f"• {name} — {n} {suffix}" for name, n in items)

    block("👍 Что хвалят (темы):", r["topics"]["positive"], "отз.")
    block("👎 На что жалуются (темы):", r["topics"]["negative"], "отз.")
    block("🔁 Частые фразы в негативе:", r["phrases"]["negative"], "раз")
    block("🔁 Частые фразы в позитиве:", r["phrases"]["positive"], "раз")
    block("💬 Повторяющиеся комментарии:", r["frequent_comments"], "раз")
    if r["negative_examples"]:
        lines.append("\n⚠️ Самые уверенные негативные отзывы:")
        lines.extend(f"• {t}" for t in r["negative_examples"])
    lines.append(f"\nСредняя уверенность модели: {r['avg_confidence']}; "
                 f"с эмодзи: {r['with_emoji_share']}% отзывов")
    return "\n".join(lines)[:4000]


def make_pie_chart(r: dict, path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")                      # без окна, работает на сервере
    import matplotlib.pyplot as plt
    colors = {"positive": "#4caf50", "neutral": "#ffc107", "negative": "#f44336"}
    items = [(l, r["counts"][l]) for l in ORDER if r["counts"][l] > 0]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie([n for _, n in items], labels=[TITLES[l] for l, _ in items],
           colors=[colors[l] for l, _ in items], autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Тональность отзывов (всего {r['total']})")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
