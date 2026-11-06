"""Описательные статистики дневных доходностей (таблицы 7 и 8 отчёта)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._core import ensure_list, to_asset


def daily_stats(x) -> pd.Series:
    """Полный набор статистик одного ряда (строки таблицы 7 отчёта).

    Работает с любым входом: тикер, ``Asset``, Series/DataFrame.
    """
    a = to_asset(x)
    r = a.returns.to_numpy()
    p = a.prices.to_numpy()
    d = a.dates
    opy = a.opy

    s = {}
    s["Период"] = f"{d[0].date()} - {d[-1].date()}"
    s["Наблюдений (дней)"] = int(len(r))
    s["Средняя годовая (арифметическая), %"] = round(
        float(r.mean() * opy * 100), 3
    )
    s["CAGR (геометрическая), %"] = round(
        float(((p[-1] / p[0]) ** (opy / len(r)) - 1) * 100), 3
    )
    s["Волатильность годовая, %"] = round(
        float(r.std(ddof=1) * np.sqrt(opy) * 100), 3
    )
    i_min, i_max = int(r.argmin()), int(r.argmax())
    s["Минимум дневной, % (дата)"] = (
        f"{r[i_min] * 100:.2f} ({d[i_min + 1].strftime('%d.%m.%Y')})"
    )
    s["Максимум дневной, % (дата)"] = (
        f"{r[i_max] * 100:.2f} ({d[i_max + 1].strftime('%d.%m.%Y')})"
    )
    s["Асимметрия (skewness)"] = round(float(pd.Series(r).skew()), 3)
    s["Избыточный эксцесс"] = round(float(pd.Series(r).kurt()), 3)
    s["Доля отрицательных дней, %"] = round(float((r < 0).mean() * 100), 2)
    for q, tag in ((5, "95"), (1, "99")):
        v = float(np.percentile(r, q))
        s[f"VaR {tag}% исторический, %"] = round(v * 100, 3)
        s[f"CVaR {tag}% исторический, %"] = round(
            float(r[r <= v].mean() * 100), 3
        )
    s["Макс. скользящая волатильность (63 дн.), %"] = round(
        float(pd.Series(r).rolling(63).std().max() * np.sqrt(opy) * 100), 3
    )
    # Максимальная просадка и восстановление
    peak = np.maximum.accumulate(p)
    dd = p / peak - 1
    i_bot = int(dd.argmin())
    after = np.where(p[i_bot:] >= peak[i_bot])[0]
    if dd[i_bot] == 0:
        s["Максимальная просадка, % (дно)"] = "0"
        s["Восстановление просадки (дата)"] = "-"
    else:
        s["Максимальная просадка, % (дно)"] = (
            f"{dd[i_bot] * 100:.2f} ({d[i_bot].strftime('%d.%m.%Y')})"
        )
        s["Восстановление просадки (дата)"] = (
            d[i_bot + after[0]].strftime("%d.%m.%Y") if len(after) else "-"
        )
    s["Шарп (rf = 0), годовых"] = round(
        float((r.mean() * opy) / (r.std(ddof=1) * np.sqrt(opy))), 3
    )
    return pd.Series(s, name=a.name, dtype=object)


def stats_table(tickers) -> pd.DataFrame:
    """Таблица 7: статистики всех активов рядом (колонки - активы)."""
    assets = [to_asset(t) for t in ensure_list(tickers)]
    return pd.DataFrame({a.name: daily_stats(a) for a in assets})


# ---------------------------------------------------------------------- #
def cagr(x) -> float:
    """Годовой темп роста (геометрическая доходность), доля единицы."""
    a = to_asset(x)
    p = a.prices.to_numpy()
    return float((p[-1] / p[0]) ** (a.opy / len(p)) - 1)


def volatility(x) -> float:
    """Годовая волатильность дневных доходностей, доля единицы."""
    a = to_asset(x)
    return float(a.returns.std(ddof=1) * np.sqrt(a.opy))


def sharpe(x) -> float:
    """Коэффициент Шарпа (безрисковая ставка = 0)."""
    a = to_asset(x)
    r = a.returns.to_numpy()
    return float((r.mean() * a.opy) / (r.std(ddof=1) * np.sqrt(a.opy)))


def sortino(x) -> float:
    """Коэффициент Сортино: доходность к волатильности вниз."""
    a = to_asset(x)
    r = a.returns
    downside = r[r < 0].std(ddof=1)
    if not downside or downside == 0:
        return float("inf")
    return float((r.mean() * a.opy) / (downside * np.sqrt(a.opy)))


def max_drawdown_value(x) -> float:
    """Максимальная просадка как отрицательная доля (для calmar)."""
    p = to_asset(x).prices.to_numpy()
    peak = np.maximum.accumulate(p)
    return float((p / peak - 1).min())


def calmar(x) -> float:
    """Коэффициент Калмара: CAGR к модулю максимальной просадки."""
    mdd = max_drawdown_value(x)
    if mdd == 0:
        return float("inf")
    return cagr(x) / abs(mdd)


# ---------------------------------------------------------------------- #
def annual_returns(x) -> pd.Series:
    """Доходность каждого календарного года, % (произведение (1+r) - 1)."""
    a = to_asset(x)
    r = a.returns
    g = (1 + r).groupby(r.index.year).prod() - 1
    out = (g * 100).round(2)
    out.index.name = "Год"
    out.name = a.name
    return out


def annual_table(tickers) -> pd.DataFrame:
    """Таблица 8: годовые доходности всех активов рядом."""
    assets = [to_asset(t) for t in ensure_list(tickers)]
    return pd.DataFrame({a.name: annual_returns(a) for a in assets})


def annual_stats(x) -> pd.Series:
    """Статистики годовых доходностей по ПОЛНЫМ годам (таблица 8)."""
    a = to_asset(x)
    ann = annual_returns(a)
    # Неполный последний год не участвует в статистиках.
    end = a.dates[-1]
    if end.month < 12:
        ann = ann[ann.index != end.year]
    if ann.empty:
        return pd.Series(
            {"Полных лет": 0, "Средняя, %": float("nan"),
             "Ст. отклонение, %": float("nan"), "Убыточных лет": 0},
            name=a.name,
        )
    s = {
        "Полных лет": int(len(ann)),
        "Средняя, %": round(float(ann.mean()), 2),
        "Ст. отклонение, %": (
            round(float(ann.std(ddof=1)), 2) if len(ann) > 1 else float("nan")
        ),
        "Убыточных лет": int((ann < 0).sum()),
        "Лучшая, %": float(ann.max()),
        "Худшая, %": float(ann.min()),
    }
    return pd.Series(s, name=a.name)
