from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from ml.predict import predict

router = Router()


@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Я анализирую тональность отзывов.\n"
        "Пришли мне текст отзыва, и я скажу, он положительный, нейтральный или отрицательный."
    )


@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "Команды:\n"
        "/start - начать работу\n"
        "/help - справка\n\n"
        "Просто отправь текст отзыва, и я определю его тональность."
    )


@router.message()
async def handle_text(message: Message):
    if not message.text:
        await message.answer("Пожалуйста, отправь текстовое сообщение.")
        return
    label, confidence = predict(message.text)
    await message.answer(f"Тональность: {label}\nУверенность: {confidence:.0%}")