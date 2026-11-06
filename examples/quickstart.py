"""Быстрый старт qresearch: полный отчёт в две строки.

Запустите в Google Colab (https://colab.research.google.com) или локально
после `pip install qresearch`:
    python examples/quickstart.py
"""
import qresearch as qr

# Полный отчёт по длинным ОФЗ и денежному рынку:
# таблицы 7-9, диагностика, портфель и графики.
result = qr.report(["RUGBITR5+", "RUSFARIND"], n_sims=10_000)

# Дальше можно работать с результатами как с обычными DataFrame:
# result["stats"]         - таблица 7
# result["annual"]        - таблица 8
# result["monte_carlo"]   - таблица 9 по каждому активу

# Отдельные метрики по акции Сбербанка:
print(qr.quick("SBER"))
print(qr.drawdown_info("SBER"))
print(qr.var("SBER", level=99))

# Шпаргалка по всем функциям:
qr.help()
