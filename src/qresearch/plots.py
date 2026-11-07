"""Графики matplotlib: одна функция - один готовый рисунок."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _sps

from ._core import ensure_list, to_asset
from .risk import drawdown, rolling_vol
from .stats import annual_returns

_COLORS = ("tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple",
           "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan")


def _axes(figsize=(10, 4.5)):
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_growth(x, log: bool = True):
    """Рост с базой 100; лог-шкала по умолчанию показывает процентные темпы."""
    assets = [to_asset(t) for t in ensure_list(x)]
    ax = _axes()
    for i, a in enumerate(assets):
        p = a.prices
        ax.plot(p.index, p / p.iloc[0] * 100,
                color=_COLORS[i % len(_COLORS)], lw=1.2, label=a.name)
    if log:
        ax.set_yscale("log")
    ax.set_title("Рост с базой 100 (лог-шкала)" if log else "Рост с базой 100")
    ax.set_ylabel("База 100")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def plot_returns(x):
    """Дневные доходности во времени."""
    assets = [to_asset(t) for t in ensure_list(x)]
    ax = _axes()
    for i, a in enumerate(assets):
        ax.plot(a.returns.index, a.returns * 100,
                color=_COLORS[i % len(_COLORS)], lw=0.7, alpha=0.8, label=a.name)
    ax.set_title("Дневные доходности, %")
    ax.set_ylabel("%")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def plot_histogram(x, bins: int = 80):
    """Гистограмма дневных доходностей с наложенными нормалью и t-распределением."""
    a = to_asset(x)
    r = a.returns.to_numpy()
    ax = _axes()
    ax.hist(r, bins=bins, density=True, color="tab:blue", alpha=0.5,
            label="Фактические доходности")
    grid = np.linspace(r.min(), r.max(), 400)
    ax.plot(grid, _sps.norm.pdf(grid, r.mean(), r.std(ddof=1)),
            "k--", lw=1.2, label="Нормальное распределение")
    try:
        df_t, loc_t, scale_t = _sps.t.fit(r)
        ax.plot(grid, _sps.t.pdf(grid, df_t, loc_t, scale_t),
                color="crimson", lw=1.2, label=f"t-распределение (df={df_t:.0f})")
    except Exception:  # t.fit редкие сбои не должны ломать график
        pass
    ax.set_title(f"Гистограмма дневных доходностей: {a.name}")
    ax.set_xlabel("Дневная доходность")
    ax.set_ylabel("Плотность")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def plot_qq(x):
    """QQ-график: квантили доходностей против нормального распределения."""
    a = to_asset(x)
    ax = _axes()
    _sps.probplot(a.returns.to_numpy(), dist="norm", plot=ax)
    ax.set_title(f"QQ-график против нормали: {a.name}")
    ax.get_lines()[0].set(markersize=3, color="tab:blue",
                          label="Квантили выборки")
    ax.get_lines()[1].set(color="crimson", lw=1.2, label="Теория")
    ax.legend()
    return ax


def plot_drawdown(x):
    """Просадки от исторических пиков."""
    a = to_asset(x)
    dd = drawdown(a)
    ax = _axes()
    ax.fill_between(dd.index, dd * 100, 0, color="crimson", alpha=0.5)
    ax.set_title(f"Просадки от пика: {a.name}")
    ax.set_ylabel("%")
    ax.grid(alpha=0.3)
    return ax


def plot_rolling_vol(x, window: int = 63):
    """Скользящая годовая волатильность (окно 63 дня = квартал)."""
    a = to_asset(x)
    v = rolling_vol(a, window=window)
    ax = _axes()
    ax.plot(v.index, v, color="tab:orange", lw=1.2)
    ax.set_title(f"Скользящая волатильность ({window} дн.): {a.name}")
    ax.set_ylabel("%, годовых")
    ax.grid(alpha=0.3)
    return ax


def plot_annual(x):
    """Годовые доходности столбиками (положительные и отрицательные)."""
    a = to_asset(x)
    ann = annual_returns(a)
    ax = _axes((10, 4.5))
    colors = ["tab:blue" if v >= 0 else "crimson" for v in ann.to_numpy()]
    ax.bar(ann.index.astype(str), ann.to_numpy(), color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"Годовые доходности: {a.name}")
    ax.set_ylabel("%")
    ax.grid(alpha=0.3, axis="y")
    return ax


def plot_all(x):
    """Все основные графики по каждому активу."""
    import matplotlib.pyplot as plt

    assets = [to_asset(t) for t in ensure_list(x)]
    for a in assets:
        plot_growth(a)
        plot_histogram(a)
        plot_qq(a)
        plot_drawdown(a)
        plot_rolling_vol(a)
        plot_annual(a)
    plt.tight_layout()
    return plt.show()
