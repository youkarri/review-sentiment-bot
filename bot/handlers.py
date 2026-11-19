import asyncio
import csv
import io
from collections import Counter

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.types import Message

from bot import db
from bot.analytics import build_report
from ml.sentiment import predict, LABEL_RU

router = Router()

MAX_REVIEWS = 300
MAX_FILE_SIZE = 1_000_000
TEXT_NAMES = ("text", "review", "отзыв", "текст")
RATING_NAMES = ("rating", "оценка", "рейтинг", "stars")


def decode_bytes(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1251", errors="ignore")


def parse_reviews(text: str, filename: str):
    """Возвращает список (текст, оценка или None)."""
    if not filename.endswith(".csv"):
        return [(line.strip(), None) for line in text.splitlines() if line.strip()]
    first = text.splitlines()[0] if text.strip() else ""
    delimiter = ";" if first.count(";") > first.count(",") else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    if not rows:
        return []
    header = [c.strip().lower() for c in rows[0]]
    t_col = next((i for i, n in enumerate(header) if n in TEXT_NAMES), None)
    r_col = next((i for i, n in enumerate(header) if n in RATING_NAMES), None)
    start = 0 if t_col is None else 1
    t_col = t_col or 0
    out = []
    for row in rows[start:]:
        if len(row) <= t_col or not row[t_col].strip():
            continue
        rating = None
        if r_col is not None and len(row) > r_col:
            try:
                rating = float(row[r_col].replace(",", "."))
            except ValueError:
                pass
        out.append((row[t_col].strip(), rating))
    return out


def analyze(reviews):
    return [(t, *predict(t)) for t in reviews]


def format_stats(counts: Counter, total: int) -> str:
    lines = []
    for label in ("positive", "neutral", "negative"):
        n = counts.get(label, 0)
        lines.append(f"{LABEL_RU.get(label, label)}: {n} ({n / total:.0%})")
    return "\n".join(lines)


@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Я анализирую тональность отзывов.\n"
        "Пришли текст отзыва или файл .txt / .csv с отзывами."
    )


@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "Команды:\n"
        "/start - начать работу\n"
        "/help - справка\n"
        "/stats - статистика по всем твоим отзывам\n\n"
        "Отправь текст отзыва или файл:\n"
        ".txt - один отзыв на строку\n"
        ".csv - столбец text (или отзыв) и, по желанию, rating (или оценка)"
    )


@router.message(Command("stats"))
async def cmd_stats(message: Message):
    counts = db.get_stats(message.from_user.id)
    total = sum(counts.values())
    if not total:
        await message.answer("История пока пуста. Отправь отзыв или файл.")
        return
    await message.answer(f"Всего проанализировано: {total}\n" + format_stats(counts, total))


@router.message(F.document)
async def handle_document(message: Message, bot: Bot):
    doc = message.document
    name = (doc.file_name or "").lower()
    if not name.endswith((".txt", ".csv")):
        await message.answer("Поддерживаются только файлы .txt и .csv.")
        return
    if doc.file_size and doc.file_size > MAX_FILE_SIZE:
        await message.answer("Файл слишком большой (максимум 1 МБ).")
        return

    buf = io.BytesIO()
    await bot.download(doc, destination=buf)
    parsed = parse_reviews(decode_bytes(buf.getvalue()), name)
    if not parsed:
        await message.answer("Не нашёл в файле отзывов.")
        return

    note = ""
    if len(parsed) > MAX_REVIEWS:
        parsed = parsed[:MAX_REVIEWS]
        note = f"\n\n(обработаны первые {MAX_REVIEWS} отзывов)"

    reviews = [t for t, _ in parsed]
    ratings = [r for _, r in parsed]

    await message.answer(f"Анализирую отзывов: {len(reviews)}...")
    results = await asyncio.to_thread(analyze, reviews)
    db.save_reviews(message.from_user.id, results)
    await message.answer(build_report(results, ratings, LABEL_RU) + note)


@router.message()
async def handle_text(message: Message):
    if not message.text:
        await message.answer("Пожалуйста, отправь текст или файл .txt / .csv.")
        return
    label, confidence = predict(message.text)
    db.save_reviews(message.from_user.id, [(message.text, label, confidence)])
    await message.answer(
        f"Тональность: {LABEL_RU.get(label, label)}\n"
        f"Уверенность: {confidence:.0%}"
    )