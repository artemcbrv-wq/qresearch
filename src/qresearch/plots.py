"""Графики matplotlib в едином стиле: одна функция - один готовый рисунок.

Крупные фигуры, сетка с alpha=0.35, палитра #1f77b4 / #2ca02c /
#d62728, гистограммы с наложенными нормалью и t-распределением,
просадки с заливкой, многопанельные компоновки.

У каждого графика есть параметр ``save``:

- ``save=None`` - только показать в ноутбуке (по умолчанию);
- ``save=True`` - показать и сохранить PNG (dpi=150) в папку ``qr_plots``;
- ``save="папка"`` - показать и сохранить в указанную папку;
- ``save="файл.png"`` - показать и сохранить по точному пути.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats as _sps

from ._core import ensure_list, to_asset
from .montecarlo import monte_carlo as _monte_carlo
from .risk import drawdown, rolling_vol
from .stats import annual_returns

_COLORS = ("#1f77b4", "#2ca02c", "#6c757d", "#9467bd", "#d62728",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")
_FILL = "#a0d2eb"       # заливка под просадками
_TREND = "#1768ac"      # блочный бутстреп на гистограммах Монте-Карло


def _grid(ax):
    """Сетка в едином стиле библиотеки."""
    ax.grid(alpha=0.35)


def _maybe_save(fig, save, stem: str) -> None:
    """Сохраняет фигуру в PNG, если попросили (см. модуль docstring)."""
    if not save:
        return
    if save is True:
        os.makedirs("qr_plots", exist_ok=True)
        path = os.path.join("qr_plots", stem + ".png")
    else:
        path = str(save)
        if not path.endswith(".png"):
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, stem + ".png")
        else:
            folder = os.path.dirname(path)
            if folder:
                os.makedirs(folder, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def _stem(*names) -> str:
    """Имя файла из имён активов: growth_SBER_GAZP."""
    return "_".join(str(n) for n in names)


def _safe(stem: str) -> str:
    """Заменяет символы, недопустимые в именах файлов (для plot_table)."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)


def plot_growth(x, log: bool = True, save=None):
    """Накопленная динамика с базой 100 (лог-шкала по умолчанию).

    Лог-шкала показывает процентные темпы роста: одинаковый наклон
    линий означает одинаковую скорость роста в процентах.
    """
    import matplotlib.pyplot as plt

    assets = [to_asset(t) for t in ensure_list(x)]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, a in enumerate(assets):
        p = a.prices
        base = p / p.iloc[0] * 100
        ax.plot(base.index, base.values,
                color=_COLORS[i % len(_COLORS)], lw=1.6,
                label=f"{a.name} (старт {p.index[0].date()})")
    if log:
        ax.set_yscale("log")
    ax.set_ylabel("Значение, база = 100 на дату старта"
                  + (" (лог-шкала)" if log else ""))
    ax.set_title("Накопленная динамика (база 100)")
    _grid(ax)
    ax.legend(loc="upper left", fontsize=10)
    _maybe_save(fig, save, "growth_" + _stem(*(a.name for a in assets)))
    return ax


def plot_returns(x, save=None):
    """Дневные доходности во времени, %; по панели на актив.

    По форме линии видно кластеризацию волатильности: спокойные
    периоды чередуются с полосами больших движений.
    """
    import matplotlib.pyplot as plt

    assets = [to_asset(t) for t in ensure_list(x)]
    n = len(assets)
    if n == 1:
        fig, ax = plt.subplots(figsize=(12, 5))
        axes = [ax]
    else:
        fig, axes_all = plt.subplots(n, 1, figsize=(12, 3 * n + 1))
        axes = list(axes_all)
    for i, (ax, a) in enumerate(zip(axes, assets)):
        ax.plot(a.returns.index, a.returns * 100,
                color=_COLORS[i % len(_COLORS)], lw=0.7)
        ax.set_title(f"{a.name}, % за день")
        ax.set_ylabel("Доходность, %")
        _grid(ax)
    fig.tight_layout()
    _maybe_save(fig, save, "returns_" + _stem(*(a.name for a in assets)))
    return axes[0] if n == 1 else fig


def plot_histogram(x, bins: int = 120, save=None):
    """Гистограмма дневных доходностей, %: нормаль пунктиром + t (MLE).

    Ось X обрезана по перцентилям 0.1-99.9 с запасом 5%, чтобы хвост
    не растягивал картинку; в заголовке видно, сколько точек попало
    «вне оси». t-распределение подобрано методом максимального
    правдоподобия и показывает тяжесть хвостов: у реальных рядов
    пик острее и хвосты толще, чем у нормали.
    """
    import matplotlib.pyplot as plt

    a = to_asset(x)
    r = (a.returns * 100).to_numpy()
    lo, hi = np.percentile(r, [0.1, 99.9])
    pad = (hi - lo) * 0.05
    lo, hi = lo - pad, hi + pad
    outside = int(((r < lo) | (r > hi)).sum())

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.hist(r, bins=bins, density=True, color=_COLORS[0], alpha=0.75,
            label="Эмпирическое распределение")
    xs = np.linspace(lo, hi, 600)
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    ax.plot(xs, _sps.norm.pdf(xs, mu, sd), "k--", lw=1.5,
            label="Нормальное")
    try:
        nu, loc, scale = _sps.t.fit(r)
        ax.plot(xs, _sps.t.pdf(xs, nu, loc=loc, scale=scale),
                color="#d62728", lw=1.8, label=f"Стьюдента (ν={nu:.1f})")
    except Exception:  # редкие сбои t.fit не должны ломать график
        pass
    ax.set_xlim(lo, hi)
    ax.set_title(f"{a.name} (дней: {len(r)}; вне оси: {outside})")
    ax.set_xlabel("Дневная доходность, %")
    ax.set_ylabel("Плотность")
    _grid(ax)
    ax.legend(fontsize=9)
    _maybe_save(fig, save, "histogram_" + a.name)
    return ax


def plot_qq(x, save=None):
    """QQ-график: выборочные квантили против квантилей нормали, в %.

    Расчёт: probplot по дневным доходностям в процентах; по оси X -
    квантили нормали, подогнанной к ряду (slope * osm + intercept из
    probplot), то есть в тех же процентах, что и выборочные квантили
    по оси Y. Пределы осей одинаковые, поэтому «идеальная нормальность»
    - точная диагональ y = x из угла в угол, а отклонение точек от неё
    читается прямо в процентах: S-образные концы (точки выше прямой
    справа и ниже слева) - «тяжёлые хвосты», экстремальные дни случаются
    чаще, чем предсказывает нормальное распределение.

    Панель ~6.5 x 5.5; для списка активов - по панели на актив.
    Подписи осей - как в оригинальной работе.
    """
    import matplotlib.pyplot as plt

    assets = [to_asset(t) for t in ensure_list(x)]
    n = len(assets)
    if n == 1:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        axes = [ax]
    else:
        fig, axes_all = plt.subplots(1, n, figsize=(6.5 * n, 5.5))
        axes = list(axes_all)
    for i, (ax, a) in enumerate(zip(axes, assets)):
        vals = (a.returns * 100).to_numpy()
        (osm, osr), (slope, intercept, _) = _sps.probplot(vals, dist="norm")
        theor = slope * osm + intercept  # квантили подогнанной нормали, %
        ax.scatter(theor, osr, s=6, color=_COLORS[i % len(_COLORS)], alpha=0.6)
        lo = min(float(theor.min()), float(osr.min()))
        hi = max(float(theor.max()), float(osr.max()))
        pad = (hi - lo) * 0.05
        lims = [lo - pad, hi + pad]
        ax.plot(lims, lims, "k--", lw=1.2)  # диагональ y = x из угла в угол
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(f"{a.name} — QQ-график")
        ax.set_xlabel("Теоретические квантили N(0,1)")
        ax.set_ylabel("Квантили доходностей, %")
        ax.grid(alpha=0.35, linestyle="--")
    _maybe_save(fig, save, "qq_" + _stem(*(a.name for a in assets)))
    return axes[0] if n == 1 else fig


def plot_drawdown(x, save=None):
    """Просадки от исторического пика, %: линия с заливкой.

    В заголовке - максимальная просадка и дата дна.
    """
    import matplotlib.pyplot as plt

    a = to_asset(x)
    dd = drawdown(a) * 100
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dd.index, dd.values, color="#7f7f7f", lw=1.0)
    ax.fill_between(dd.index, dd.values, 0, color=_FILL, alpha=0.85)
    mdd = float(dd.min())
    bottom = dd.idxmin()
    ax.set_ylabel("Просадка от пика, %")
    ax.set_title(f"Просадки: {a.name} "
                 f"(максимальная {mdd:.1f}%, дно {bottom.date()})")
    _grid(ax)
    _maybe_save(fig, save, "drawdown_" + a.name)
    return ax


def plot_rolling_vol(x, window: int = 63, save=None):
    """Скользящая годовая волатильность, % (окно 63 дня = квартал)."""
    import matplotlib.pyplot as plt

    a = to_asset(x)
    v = rolling_vol(a, window=window)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(v.index, v.values, color=_COLORS[0], lw=1.0)
    ax.set_title(f"Скользящая волатильность (окно {window} дн.): {a.name}")
    ax.set_ylabel("Годовая волатильность, %")
    _grid(ax)
    _maybe_save(fig, save, "rolling_vol_" + a.name)
    return ax


def _year_incomplete(a, year) -> bool:
    """Год не закончился для этого актива."""
    end = a.dates[-1]
    return end.year == year and not (end.month == 12 and end.day == 31)


def plot_annual(x, save=None):
    """Годовые доходности: столбики по годам и их распределение.

    Две панели: слева годовые доходности всех активов столбиками по
    годам (``*`` - неполный год), справа - распределение годовых
    доходностей полных лет. Возвращает фигуру matplotlib.
    """
    import matplotlib.pyplot as plt

    assets = [to_asset(t) for t in ensure_list(x)]
    ann = {a.name: annual_returns(a) for a in assets}
    years = sorted(set().union(*(set(v.index) for v in ann.values())))
    if not years:
        raise ValueError("Нет данных для годовых доходностей.")
    last = years[-1]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.6, 1]}
    )
    pos = np.arange(len(years))
    width = 0.8 / len(assets)
    for i, a in enumerate(assets):
        vals = [ann[a.name].get(y, np.nan) for y in years]
        offset = (i - (len(assets) - 1) / 2) * width
        ax1.bar(pos + offset, vals, width=width * 0.92,
                color=_COLORS[i % len(_COLORS)], label=a.name)
    marked = [y for y in years
              if any(_year_incomplete(a, y) for a in assets)]
    ax1.set_xticks(pos)
    ax1.set_xticklabels(
        [str(y) + ("*" if y in marked else "") for y in years],
        rotation=45,
    )
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_ylabel("Годовая доходность, %")
    ax1.set_title("Годовые доходности (* - неполный год)")
    ax1.grid(alpha=0.35, axis="y")
    ax1.legend(fontsize=9)

    for i, a in enumerate(assets):
        s = ann[a.name]
        if _year_incomplete(a, last):
            s = s[s.index != last]
        if len(s):
            ax2.hist(s, bins=min(12, max(3, len(s))),
                     color=_COLORS[i % len(_COLORS)], alpha=0.6,
                     label=f"{a.name} ({len(s)} полных лет)")
    ax2.set_xlabel("Годовая доходность, %")
    ax2.set_ylabel("Число лет")
    ax2.set_title("Распределение годовых доходностей")
    ax2.grid(alpha=0.35, axis="y")
    ax2.legend(fontsize=9)
    fig.tight_layout()
    _maybe_save(fig, save, "annual_" + _stem(*(a.name for a in assets)))
    return fig


def plot_monte_carlo(x, n_sims: int = 20_000, days=None, seed: int = 42,
                     block: int = 21, bins: int = 60, save=None):
    """Смоделированные распределения годовой доходности, три метода.

    iid-бутстреп и блочный бутстреп - гистограммы (блочный даёт более
    тяжёлые хвосты, потому что сохраняет кластеризацию волатильности),
    параметрический метод - красная кривая нормальной плотности. По
    панели на актив. Возвращает фигуру matplotlib.
    """
    import matplotlib.pyplot as plt

    assets = [to_asset(t) for t in ensure_list(x)]
    n = len(assets)
    if n == 1:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        axes = [ax]
    else:
        fig, axes_all = plt.subplots(1, n, figsize=(14, 5.5))
        axes = list(axes_all)
    for ax, a in zip(axes, assets):
        sims = _monte_carlo(a, days=days, n_sims=n_sims, seed=seed,
                            block=block)
        iid = sims["iid-бутстреп"] * 100
        blk = sims[f"блочный бутстреп ({block} дн.)"] * 100
        par = sims["параметрический N(mu, sigma)"] * 100
        ax.hist(iid, bins=bins, density=True, color="#1f77b4",
                alpha=0.5, label="iid-бутстреп")
        ax.hist(blk, bins=bins, density=True, color=_TREND,
                alpha=0.55, label=f"блочный бутстреп ({block} дн.)")
        xs = np.linspace(min(iid.min(), blk.min(), par.min()),
                         max(iid.max(), blk.max(), par.max()), 400)
        ax.plot(xs, _sps.norm.pdf(xs, float(par.mean()), float(par.std(ddof=1))),
                color="#d62728", lw=1.8, label="параметрическая N(μ, σ)")
        ax.set_title(a.name)
        ax.set_xlabel("Годовая доходность, %")
        ax.set_ylabel("Плотность")
        _grid(ax)
        ax.legend(fontsize=9)
    fig.suptitle("Смоделированные распределения годовой доходности",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _maybe_save(fig, save,
                "monte_carlo_" + _stem(*(a.name for a in assets)))
    return fig


def plot_table(data, title: str = "", fontsize: int = 10, save=None):
    """Рисует DataFrame или Series как таблицу - картинку matplotlib.

    Пригодится для отчётов и скриншотов: вместо текстового вывода
    получается аккуратная таблица. NaN показывается прочерком::

        qr.plot_table(qr.stats_table("SBER"))
        qr.plot_table(qr.monte_carlo_table("SBER"))
        qr.plot_table(qr.stats_table("SBER"), save="отчёт/метрики.png")
    """
    import matplotlib.pyplot as plt

    if isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        df = data.copy()
    df = df.reset_index()  # индекс становится первой колонкой таблицы
    df.columns = [""] + [str(c) for c in df.columns[1:]]
    shown = df.where(df.notna(), "-")
    cells = shown.astype(str).values.tolist()
    n_rows, n_cols = len(cells), len(cells[0])

    width = min(max(4.0 + 2.6 * n_cols, 7.0), 16.0)
    height = min(1.6 + 0.5 * n_rows, 22.0)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    tbl = ax.table(cellText=cells, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_height(cell.get_height() * 1.5)
        if r == 0:
            cell.set_facecolor("#dce6f5")
            cell.set_text_props(fontweight="bold")
    try:
        tbl.auto_set_column_width(col=list(range(n_cols)))
    except Exception:  # старые matplotlib - ширины остаются равными
        pass
    _maybe_save(fig, save, _safe(title or "table"))
    return ax


def plot_all(x, save=None):
    """Все основные графики по каждому активу (кроме Монте-Карло).

    Рисует рост с базой 100, дневные доходности, а по каждому активу:
    гистограмму, QQ-график, просадки, скользящую волатильность и
    годовые доходности. ``save`` передаётся во все графики (см.
    docstring модуля :mod:`qresearch.plots`).
    """
    import matplotlib.pyplot as plt

    assets = [to_asset(t) for t in ensure_list(x)]
    plot_growth(assets, save=save)
    plot_returns(assets, save=save)
    for a in assets:
        plot_histogram(a, save=save)
        plot_qq(a, save=save)
        plot_drawdown(a, save=save)
        plot_rolling_vol(a, save=save)
        plot_annual(a, save=save)
    plt.tight_layout()
    return plt.show()
