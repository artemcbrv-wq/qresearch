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

# Ключевые метрики таблицей-картинкой (plot=False - текстом, как раньше):
qr.quick(["SBER", "GAZP", "IMOEX"])

# Одно число одной командой - без лишних таблиц и выводов:
print(qr.cagr("SBER"))            # годовая доходность, например 0.0834
print(qr.volatility("SBER"))      # годовая волатильность
print(qr.sharpe("SBER"))           # Шарп

# Только Монте-Карло: таблица и картинка:
qr.plot_table(qr.monte_carlo_table("SBER", n_sims=10_000))
qr.plot_monte_carlo("SBER")

# Просадка с датами и VaR:
print(qr.drawdown_info("SBER"))
print(qr.var("SBER", level=99))

# Шпаргалка по всем функциям:
qr.help()
