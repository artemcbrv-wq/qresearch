"""qresearch - количественная аналитика финансовых рынков для всех.

Библиотека для анализа доходностей, риска и портфелей без глубокого
знания Python. Достаточно двух команд::

    import qresearch as qr
    qr.report(["RUGBITR5+", "RUSFARIND"])   # полный отчёт

Быстрый старт
-------------
Скачивание данных (источник определяется автоматически):
    a = qr.load("SBER")         # акция МосБиржи
    b = qr.load("RUGBITR5+")    # индекс ОФЗ (МосБиржа)
    c = qr.load("AAPL")         # акция с Yahoo Finance
    d = qr.load_file("цены.csv")   # свой CSV/Excel файл

Анализ одной командой (в любую функцию можно передавать тикеры напрямую):
    qr.quick("SBER")                     # главные цифры
    qr.stats_table(["SBER", "GAZP"])     # все статистики (таблица 7)
    qr.annual_table("RUGBITR5+")         # доходности по годам (таблица 8)
    qr.diagnostics("RUGBITR5+")          # нормальность, кластеризация, ADF
    qr.monte_carlo_table("RUGBITR5+")    # Монте-Карло (таблица 9)
    qr.report("SBER")                    # всё сразу + графики

Риск и портфели:
    qr.var("SBER", level=95)             # VaR (доля единицы)
    qr.cvar("SBER", level=99)            # CVaR
    qr.drawdown_info("SBER")            # просадка с датами
    p = qr.portfolio(["SBER", "RUGBITR5+"], weights=[0.7, 0.3])
    p.summary(); p.correlation(); p.risk_contribution(); p.min_variance()

Графики:
    qr.plot_growth(["SBER", "GAZP"])     # база 100, лог-шкала
    qr.plot_drawdown("SBER")             # просадки
    qr.plot_rolling_vol("SBER")          # скользящая волатильность
    qr.plot_histogram("SBER")            # гистограмма + нормаль + t
"""
from __future__ import annotations

from ._core import Asset, ensure_list, to_asset
from .data import INDEX_PRESETS, load, load_file, load_moex, load_yahoo
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
    plot_qq,
    plot_returns,
    plot_rolling_vol,
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
    "load", "load_file", "load_moex", "load_yahoo", "INDEX_PRESETS",
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
    # отчёты
    "report", "quick",
]


def help():  # noqa: A001 - намеренно перекрывает встроенный help
    """Печатает шпаргалку по qresearch на русском."""
    print(__doc__)
