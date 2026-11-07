"""Отчёты одной командой: quick (ключевые метрики) и report (всё сразу)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._core import ensure_list, to_asset
from .diagnostics import diagnostics as _diagnostics
from .montecarlo import monte_carlo_table as _mc_table
from .plots import plot_all as _plot_all
from .portfolio import portfolio as _portfolio
from .risk import drawdown_info, max_drawdown
from .stats import annual_stats, annual_table, daily_stats, stats_table


def _fmt_frame(df) -> str:
    """Печать DataFrame с прочерком вместо NaN."""
    return df.to_string(na_rep="-")


def quick(x, plot: bool = True) -> pd.DataFrame:
    """Ключевые метрики по каждому активу: одна таблица.

    По умолчанию рисует таблицу метрик как картинку matplotlib (без
    текстового вывода). ``plot=False`` - прежнее поведение: печатать
    текстом. В любом случае возвращает DataFrame (колонки - активы).
    """
    from .plots import plot_table

    assets = [to_asset(t) for t in ensure_list(x)]
    rows = {}
    for a in assets:
        r = a.returns.to_numpy()
        p = a.prices.to_numpy()
        opy = a.opy
        rows[a.name] = {
            "Период": f"{a.dates[0].date()} - {a.dates[-1].date()}",
            "Наблюдений (дней)": len(r),
            "CAGR (геометрическая), %": round(
                float(((p[-1] / p[0]) ** (opy / len(p)) - 1) * 100), 2
            ),
            "Волатильность, %": round(
                float(r.std(ddof=1) * np.sqrt(opy) * 100), 2
            ),
            "Шарп (rf = 0), годовых": round(
                float((r.mean() * opy) / (r.std(ddof=1) * np.sqrt(opy))), 2
            ),
            "Макс. просадка, %": round(float(max_drawdown(a)) * 100, 2),
        }
    out = pd.DataFrame(rows)
    if plot:
        plot_table(out, title="qresearch: ключевые метрики")
    else:
        print("qresearch: ключевые метрики\n" + "-" * 60)
        print(_fmt_frame(out))
    return out


def report(x, days: int | None = None, n_sims: int = 20_000, seed: int = 42,
           block: int = 21, show_plots: bool = True) -> dict:
    """Полный отчёт: статистики, годовые доходности, диагностика, Монте-Карло, портфель и графики.

    Печатает все блоки отчёта и возвращает словарь с DataFrame
    для дальнейшей работы: ``stats``, ``annual``, ``diagnostics``,
    ``monte_carlo``, ``portfolio`` (если активов больше одного).
    """
    assets = [to_asset(t) for t in ensure_list(x)]
    out: dict = {}

    # ---------- Описательные статистики ----------
    out["stats"] = stats_table(assets)
    print("ОПИСАТЕЛЬНЫЕ СТАТИСТИКИ ДНЕВНЫХ ДОХОДНОСТЕЙ\n")
    print(_fmt_frame(out["stats"]))

    # ---------- Годовые доходности ----------
    out["annual"] = annual_table(assets)
    print("\nГОДОВЫЕ ДОХОДНОСТИ, %\n")
    print(_fmt_frame(out["annual"]))
    for a in assets:
        print(f"\n{a.name}: {_fmt_frame(annual_stats(a))}")

    # ---------- Диагностика модели ----------
    out["diagnostics"] = {a.name: _diagnostics(a) for a in assets}
    for name, blocks in out["diagnostics"].items():
        print(f"\nДиагностика модели: {name}\n" + "-" * 60)
        for title, series in blocks.items():
            print(f"{title}: {_fmt_frame(series)}")

    # ---------- Монте-Карло ----------
    print("\nМОНТЕ-КАРЛО: РАСПРЕДЕЛЕНИЕ ГОДОВОЙ ДОХОДНОСТИ\n")
    mc_by_asset = {}
    for a in assets:
        mc_by_asset[a.name] = _mc_table(
            a, days=days, n_sims=n_sims, seed=seed, block=block
        )
        print(f"\n{a.name}")
        print(_fmt_frame(mc_by_asset[a.name]))
    out["monte_carlo"] = mc_by_asset

    # ---------- Портфель (если активов больше одного) ----------
    if len(assets) > 1:
        pf = _portfolio(assets)
        out["portfolio"] = pf
        print("\nПОРТФЕЛЬ: риск и доходность с учётом корреляций\n")
        print("Корреляции дневных доходностей:")
        print(_fmt_frame(pf.correlation()))
        print("\nИтоги портфеля (равные веса, если веса не заданы):")
        print(_fmt_frame(pf.summary()))
        print("\nВклад активов в риск портфеля, %:")
        print(_fmt_frame(pf.risk_contribution()))
        print("\nВеса минимальной дисперсии (шорты возможны):")
        print(_fmt_frame(pf.min_variance()))
        print("\nМаксимальная просадка каждого актива:")
        for a in assets:
            info = drawdown_info(a)
            print(f"  {a.name}: {info['Макс. просадка, %']}% "
                  f"(пик {info['Пик']}, дно {info['Дно']}, "
                  f"восстановление {info['Восстановление']})")

    if show_plots:
        _plot_all(assets)
    return out
