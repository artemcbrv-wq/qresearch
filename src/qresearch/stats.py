"""Описательные статистики дневных и календарно-годовых доходностей."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _sps
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from ._core import ensure_list, to_asset
from .risk import drawdown_info


def daily_stats(x) -> pd.Series:
    """Полный набор описательных статистик дневных доходностей актива.

    Работает с любым входом: тикер, ``Asset``, Series/DataFrame.
    """
    a = to_asset(x)
    r = a.returns.to_numpy()
    p = a.prices.to_numpy()
    d = a.dates
    opy = a.opy

    s: dict = {}
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
        var = float(np.percentile(r, q))
        s[f"VaR {tag}% исторический, %"] = round(var * 100, 3)
        s[f"CVaR {tag}% исторический, %"] = round(
            float(r[r <= var].mean() * 100), 3
        )
    window = min(63, len(r))  # на короткой истории окно сжимается
    s["Макс. скользящая волатильность (63 дн.), %"] = round(
        float(pd.Series(r).rolling(window).std().max() * np.sqrt(opy) * 100), 3
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
    """Описательные статистики всех активов рядом (колонки - активы)."""
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
    if not np.isfinite(downside) or downside == 0:
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


def annual_table(tickers, mark_incomplete: bool = True) -> pd.DataFrame:
    """Годовые доходности всех активов рядом.

    С ``mark_incomplete=True`` неполный последний год помечается
    звёздочкой ``*`` в индексе: год ещё не закончился, и его доходность
    не с чем сравнивать. ``mark_incomplete=False`` убирает пометку и
    возвращает числовой индекс годов.
    """
    assets = [to_asset(t) for t in ensure_list(tickers)]
    df = pd.DataFrame({a.name: annual_returns(a) for a in assets})
    if mark_incomplete and len(df):
        last = int(df.index.max())
        incomplete = any(
            a.dates[-1].year == last
            and not (a.dates[-1].month == 12 and a.dates[-1].day == 31)
            for a in assets
        )
        if incomplete:
            df.index = pd.Index(
                [str(y) + ("*" if y == last else "") for y in df.index],
                name="Год",
            )
    return df


def annual_stats(x) -> pd.Series:
    """Статистики годовых доходностей по полным календарным годам.

    Неполный последний год не участвует в статистиках, как и первый
    год, если история начинается со второй его половины: год, покрытый
    данными наполовину, исказил бы и среднюю, и разброс.
    """
    a = to_asset(x)
    ann = annual_returns(a)
    start, end = a.dates[0], a.dates[-1]
    if not (end.month == 12 and end.day == 31):
        ann = ann[ann.index != end.year]
    if start.month > 6:
        ann = ann[ann.index != start.year]
    if ann.empty:
        return pd.Series(
            {"Полных лет": 0, "Средняя, %": float("nan"),
             "Ст. отклонение, %": float("nan"),
             "Число убыточных лет": 0},
            name=a.name,
        )
    s = {
        "Полных лет": int(len(ann)),
        "Средняя, %": round(float(ann.mean()), 2),
        "Ст. отклонение, %": (
            round(float(ann.std(ddof=1)), 2) if len(ann) > 1 else float("nan")
        ),
        "Минимум, %": round(float(ann.min()), 2),
        "25%, %": round(float(ann.quantile(0.25)), 2),
        "50%, %": round(float(ann.median()), 2),
        "75%, %": round(float(ann.quantile(0.75)), 2),
        "Максимум, %": round(float(ann.max()), 2),
        "Доля положительных лет": round(float((ann > 0).mean()), 2),
        "Число убыточных лет": int((ann < 0).sum()),
    }
    return pd.Series(s, name=a.name)


# ---------------------------------------------------------------------- #
def full_stats(x) -> pd.Series:
    """Полная статистика дневных доходностей одним списком строк.

    Дополняет :func:`daily_stats`: даты экстремумов отдельными строками,
    параметрические VaR с бэктестом превышений (факт/ожидалось),
    просадка с датами пика, дна и восстановления и p-value тестов
    Жарка-Бера, Дики-Фуллера и Льюнг-Бокса по доходностям и их
    квадратам (лаги 5 и 20). Проценты указаны в %, VaR - отрицательные.

    Требует историю длиннее 20 наблюдений (тесты ADF и Льюнг-Бокса).
    """
    a = to_asset(x)
    r = a.returns
    vals = r.to_numpy()
    n = len(vals)
    if n <= 20:
        raise ValueError(
            "История слишком короткая для full_stats: нужно больше 20 наблюдений."
        )
    opy = a.opy

    mean_ann = float(vals.mean() * opy)
    vol_ann = float(vals.std(ddof=1) * np.sqrt(opy))
    i_min, i_max = int(vals.argmin()), int(vals.argmax())

    var95h = float(np.percentile(vals, 5))
    var99h = float(np.percentile(vals, 1))
    cvar95h = float(vals[vals <= var95h].mean())
    cvar99h = float(vals[vals <= var99h].mean())
    mu, sd = float(vals.mean()), float(vals.std(ddof=1))
    var95p = mu - 1.6449 * sd
    var99p = mu - 2.3263 * sd
    exc95, exp95 = int((vals < var95p).sum()), int(round(n * 0.05))
    exc99, exp99 = int((vals < var99p).sum()), int(round(n * 0.01))

    window = min(63, n)
    roll_max = (
        float(pd.Series(vals).rolling(window).std().max()) * np.sqrt(opy)
    )
    info = drawdown_info(a)
    rec = info["Восстановление"]
    rec = rec if rec != "-" else "не восстановился"

    jb_p = float(_sps.jarque_bera(vals)[1])
    if np.all(vals == vals[0]):
        raise ValueError("Доходности постоянны, тест ADF неприменим.")
    adf_p = float(adfuller(vals, autolag="AIC")[1])
    lb_r = acorr_ljungbox(vals, lags=[5, 20], return_df=True)["lb_pvalue"]
    lb_r2 = acorr_ljungbox(vals ** 2, lags=[5, 20], return_df=True)["lb_pvalue"]

    s = {
        "Период": f"{a.dates[0].date()} - {a.dates[-1].date()}",
        "Наблюдений (дней)": n,
        "Средняя годовая (арифм.), %": round(mean_ann * 100, 2),
        "CAGR (геометрическая), %": round(cagr(a) * 100, 2),
        "Волатильность годовая, %": round(vol_ann * 100, 2),
        "Минимум дневной, %": round(float(vals.min()) * 100, 2),
        "Дата минимума": r.index[i_min].date(),
        "Максимум дневной, %": round(float(vals.max()) * 100, 2),
        "Дата максимума": r.index[i_max].date(),
        "Асимметрия (skewness)": round(float(pd.Series(vals).skew()), 2),
        "Эксцесс (избыточный)": round(float(pd.Series(vals).kurt()), 2),
        "Доля отрицательных дней, %": round(
            float((vals < 0).mean() * 100), 2
        ),
        "VaR 95% исторический, %": round(var95h * 100, 2),
        "VaR 99% исторический, %": round(var99h * 100, 2),
        "CVaR 95% исторический, %": round(cvar95h * 100, 2),
        "CVaR 99% исторический, %": round(cvar99h * 100, 2),
        "VaR 95% параметрический, %": round(var95p * 100, 2),
        "VaR 99% параметрический, %": round(var99p * 100, 2),
        "Превышений VaR95 (факт/ожидалось)": f"{exc95} / {exp95}",
        "Превышений VaR99 (факт/ожидалось)": f"{exc99} / {exp99}",
        "Макс. скользящая волатильность (63 дн.), %": round(
            roll_max * 100, 2
        ),
        "Макс. просадка (MDD), %": info["Макс. просадка, %"],
        "Пик MDD": info["Пик"],
        "Дно MDD": info["Дно"],
        "Восстановление MDD": rec,
        "Шарп (rf=0), годовых": round(mean_ann / vol_ann, 2),
        "Жарк-Бера p-value": round(jb_p, 4),
        "Дики-Фуллер (ADF) p-value": round(adf_p, 4),
        "Льюнг-Бокс p (доходности, лаги 5/20)":
            f"{lb_r.loc[5]:.2f} / {lb_r.loc[20]:.2f}",
        "Льюнг-Бокс p (квадраты, лаги 5/20)":
            f"{lb_r2.loc[5]:.2f} / {lb_r2.loc[20]:.2f}",
    }
    return pd.Series(s, name=a.name, dtype=object)


def full_stats_table(tickers) -> pd.DataFrame:
    """Полная статистика всех активов рядом (колонки - активы)."""
    assets = [to_asset(t) for t in ensure_list(tickers)]
    return pd.DataFrame({a.name: full_stats(a) for a in assets})


# ---------------------------------------------------------------------- #
def compare_annual(x, y):
    """Годовые доходности двух инструментов на общем периоде дат.

    Возвращает пару ``(DataFrame, корреляция)``: таблица годовых
    доходностей, %, по годам общего периода (``*`` - неполный год) и
    корреляция дневных доходностей на общем периоде.
    """
    a, b = to_asset(x), to_asset(y)
    common = a.returns.index.intersection(b.returns.index)
    if len(common) < 2:
        raise ValueError(
            f"У «{a.name}» и «{b.name}» нет общего периода истории."
        )
    ra, rb = a.returns.loc[common], b.returns.loc[common]
    corr = float(ra.corr(rb))
    ga = ((1 + ra).groupby(ra.index.year).prod() - 1) * 100
    gb = ((1 + rb).groupby(rb.index.year).prod() - 1) * 100
    years = sorted(set(ga.index) | set(gb.index))
    last = max(years)
    incomplete = common[-1] < pd.Timestamp(year=last, month=12, day=31)
    df = pd.DataFrame(
        {
            f"{a.name}, %": [ga.get(y, float("nan")) for y in years],
            f"{b.name}, %": [gb.get(y, float("nan")) for y in years],
        },
        index=pd.Index(
            [str(y) + ("*" if (y == last and incomplete) else "")
             for y in years],
            name="Год",
        ),
    )
    return df, corr


def sharpe_vs(x, benchmark) -> float:
    """Шарп актива относительно другого инструмента на общем периоде.

    Избыток средней годовой доходности актива над инструментом
    сравнения (например, денежным рынком), делённый на годовую
    волатильность самого актива.
    """
    a, b = to_asset(x), to_asset(benchmark)
    common = a.returns.index.intersection(b.returns.index)
    if len(common) < 2:
        raise ValueError(
            f"У «{a.name}» и «{b.name}» нет общего периода истории."
        )
    ra, rb = a.returns.loc[common], b.returns.loc[common]
    years = max((common[-1] - common[0]).days / 365.25, 1e-9)
    ann = len(common) / years
    excess = float(ra.mean() * ann - rb.mean() * ann)
    return excess / float(ra.std(ddof=1) * np.sqrt(ann))


def implied_rate(x) -> pd.Series:
    """Годовая ставка, %, восстановленная из капитализированного индекса.

    Для капитализированных индексов денежного рынка (типа RUSFARIND):
    доходность наблюдения аннуализируется по фактическому числу
    календарных дней до следующего наблюдения, потому что начисления
    идут по календарным дням, включая выходные и праздники.
    """
    a = to_asset(x)
    idx = a.prices.index
    gaps = idx.to_series().diff().dt.days.to_numpy()[1:]
    r = a.returns.to_numpy()
    rate = ((1 + r) ** (365.25 / gaps) - 1) * 100
    return pd.Series(rate, index=idx[1:], name=f"Ставка из {a.name}, %")
