"""Быстрый старт qresearch: полный отчёт в две строки.

Запустите локально после `pip install qresearch`:

    python examples/quickstart.py
"""
import qresearch as qr

# Полный отчёт по акциям Сбербанка и Газпрома:
# описательные статистики, годовые доходности, диагностика,
# Монте-Карло, портфель и графики.
result = qr.report(["SBER", "GAZP"], n_sims=10_000)

# Дальше можно работать с результатами как с обычными DataFrame:
# result["stats"]       - описательные статистики
# result["annual"]      - годовые доходности
# result["monte_carlo"] - Монте-Карло по каждому активу

# Ключевые метрики без лишнего:
print(qr.quick(["SBER", "GAZP", "IMOEX"]))
print(qr.drawdown_info("SBER"))
print(qr.var("SBER", level=99))

# Шпаргалка по всем функциям:
qr.help()
