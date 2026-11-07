"""qresearch - количественная аналитика финансовых рынков.

Библиотека для анализа доходностей, риска и портфелей без глубокого
знания Python. Достаточно двух команд::

    import qresearch as qr
    qr.report(["SBER", "GAZP"])   # полный отчёт

Загрузка данных
---------------
    a = qr.load("SBER")           # акция МосБиржи
    b = qr.load("IMOEX")          # индекс МосБиржи
    c = qr.load_file("цены.csv")  # свой CSV/Excel файл

Анализ одной командой (в любую функцию можно передавать тикеры напрямую):
    qr.quick("SBER")                     # ключевые метрики (таблица-картинка)
    qr.stats_table(["SBER", "GAZP"])     # все описательные статистики
    qr.annual_table("IMOEX")             # доходности по годам
    qr.diagnostics("SBER")               # нормальность, кластеризация, стационарность
    qr.monte_carlo_table("SBER")         # Монте-Карло
    qr.report("SBER")                    # всё сразу + графики

Одно число одной командой (без лишних таблиц и выводов):
    qr.cagr("SBER")          # годовая доходность, например 0.0834
    qr.volatility("SBER")    # годовая волатильность
    qr.sharpe("SBER")        # Шарп; ещё есть sortino, calmar
    qr.var("SBER", 95)       # VaR; ещё есть cvar, max_drawdown

Таблицы и Монте-Карло картинками matplotlib:
    qr.quick(["SBER", "GAZP"])                     # метрики таблицей-картинкой
    qr.plot_table(qr.stats_table("SBER"))          # любой DataFrame таблицей
    qr.plot_monte_carlo("SBER")                     # гистограмма симуляций года

Риск и портфели:
    qr.var("SBER", level=95)             # VaR (доля единицы)
    qr.cvar("SBER", level=99)            # CVaR
    qr.drawdown_info("SBER")             # просадка с датами
    p = qr.portfolio(["SBER", "GAZP"], weights=[0.7, 0.3])
    p.summary(); p.correlation(); p.risk_contribution(); p.min_variance()

Графики:
    qr.plot_growth(["SBER", "GAZP"])     # база 100, лог-шкала
    qr.plot_drawdown("SBER")             # просадки
    qr.plot_rolling_vol("SBER")          # скользящая волатильность
    qr.plot_histogram("SBER")            # гистограмма + нормаль + t
"""
from __future__ import annotations

from ._core import Asset, ensure_list, to_asset
from .data import INDEX_PRESETS, load, load_file, load_moex
from .diagnostics import (
    diagnostics,
    normality_tests,
    stationarity_test,
    volatility_clustering_test,
)
from .montecarlo import mc_summary, monte_carlo, monte_carlo_table
from .plots import (
    plot_all,
    plot_annual,
    plot_drawdown,
    plot_growth,
    plot_histogram,
    plot_monte_carlo,
    plot_qq,
    plot_returns,
    plot_rolling_vol,
    plot_table,
)
from .portfolio import Portfolio, portfolio
from .report import quick, report
from .risk import cvar, drawdown, drawdown_info, max_drawdown, rolling_vol, var
from .stats import (
    annual_returns,
    annual_stats,
    annual_table,
    calmar,
    cagr,
    daily_stats,
    sharpe,
    sortino,
    stats_table,
    volatility,
)

__version__ = "0.1.0"

__all__ = [
    # данные
    "load", "load_file", "load_moex", "INDEX_PRESETS",
    "Asset", "to_asset", "ensure_list",
    # статистики
    "daily_stats", "stats_table", "annual_returns", "annual_table",
    "annual_stats", "cagr", "volatility", "sharpe", "sortino", "calmar",
    # риск
    "var", "cvar", "drawdown", "drawdown_info", "max_drawdown",
    "rolling_vol",
    # диагностика
    "normality_tests", "volatility_clustering_test", "stationarity_test",
    "diagnostics",
    # Монте-Карло
    "monte_carlo", "mc_summary", "monte_carlo_table",
    # портфели
    "Portfolio", "portfolio",
    # графики
    "plot_growth", "plot_returns", "plot_histogram", "plot_qq",
    "plot_drawdown", "plot_rolling_vol", "plot_annual", "plot_all",
    "plot_table", "plot_monte_carlo",
    # отчёты
    "report", "quick",
]


def help():  # noqa: A001 - намеренно перекрывает встроенный help
    """Печатает шпаргалку по qresearch на русском."""
    print(__doc__)
