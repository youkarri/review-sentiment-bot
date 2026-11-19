"""Проверка аналитики на реальной модели: python -m ml.try_analytics"""
import pandas as pd
from ml.analytics import analyze, format_report, make_pie_chart

df = pd.read_csv("data/labeled.csv")

# файл-образец для демо бота (то, что пользователь пришлёт в Telegram)
df[["text"]].sample(100, random_state=1).to_csv("data/sample_reviews.csv", index=False)

result = analyze(df["text"].head(300).tolist())
print(format_report(result))
make_pie_chart(result, "data/pie.png")
print("\nДиаграмма сохранена: data/pie.png")
