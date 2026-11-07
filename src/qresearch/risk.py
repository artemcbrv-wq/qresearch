"""Риск-метрики: VaR, CVaR, просадки, скользящая волатильность."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _sps

from ._core import to_asset


def var(x, level: float = 95, method: str = "hist") -> float:
    """Value at Risk дневных доходностей, доля единицы (отрицательное).

    Параметры
    ----------
    level :
        Уровень доверия в процентах (например, 95).
    method :
        ``"hist"`` - исторический (перцентиль);
        ``"parametric"`` (псевдоним ``"normal"``) - нормальный N(mu, sigma).
    """
    a = to_asset(x)
    r = a.returns.to_numpy()
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    if method in ("hist", "historical", "h"):
        return float(np.percentile(r, 100 - level))
    if method in ("parametric", "normal", "n"):
        z = _sps.norm.ppf(level / 100)
        return float(mu - z * sd)
    raise ValueError(f"Неизвестный метод VaR: {method!r} (используйте hist/parametric).")


def cvar(x, level: float = 95, method: str = "hist") -> float:
    """CVaR (Expected Shortfall): средний убыток за VaR, доля единицы."""
    a = to_asset(x)
    r = a.returns.to_numpy()
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    if method in ("hist", "historical", "h"):
        v = float(np.percentile(r, 100 - level))
        tail = r[r <= v]
        return float(tail.mean()) if len(tail) else v
    if method in ("parametric", "normal", "n"):
        alpha = 1 - level / 100
        z = _sps.norm.ppf(level / 100)
        return float(mu - sd * _sps.norm.pdf(z) / alpha)
    raise ValueError(f"Неизвестный метод CVaR: {method!r} (используйте hist/parametric).")


def drawdown(x) -> pd.Series:
    """Текущая просадка от исторического пика, доля единицы (<= 0)."""
    a = to_asset(x)
    p = a.prices
    dd = p / p.cummax() - 1
    dd.name = "Просадка"
    return dd


def max_drawdown(x) -> float:
    """Максимальная историческая просадка, доля единицы (<= 0)."""
    return float(drawdown(x).min())


def drawdown_info(x) -> dict:
    """Максимальная просадка с датами пика, дна и восстановления."""
    a = to_asset(x)
    p = a.prices.to_numpy()
    d = a.dates
    peak = np.maximum.accumulate(p)
    dd = p / peak - 1
    i_bot = int(dd.argmin())
    if dd[i_bot] == 0:
        return {
            "Макс. просадка, %": 0.0,
            "Пик": "-",
            "Дно": "-",
            "Восстановление": "-",
        }
    i_peak = int((p[: i_bot + 1] == peak[i_bot]).argmax())
    after = np.where(p[i_bot:] >= peak[i_bot])[0]
    return {
        "Макс. просадка, %": round(float(dd[i_bot]) * 100, 2),
        "Пик": d[i_peak].strftime("%d.%m.%Y"),
        "Дно": d[i_bot].strftime("%d.%m.%Y"),
        "Восстановление": (
            d[i_bot + after[0]].strftime("%d.%m.%Y") if len(after) else "-"
        ),
    }


def rolling_vol(x, window: int = 63) -> pd.Series:
    """Скользящая годовая волатильность, % (окно 63 дня = квартал)."""
    a = to_asset(x)
    out = (a.returns.rolling(window).std() * np.sqrt(a.opy) * 100).round(2)
    out.name = f"Волатильность ({window} дн.), %"
    return out


def var_backtest(x, level: float = 95, method: str = "parametric") -> pd.Series:
    """Бэктест VaR: сколько раз дневной убыток превысил VaR.

    Сравнивает фактическое число превышений с ожидаемым
    (``n * (1 - уровень/100)``). У хорошо откалиброванной модели факт
    близок к ожидению: систематически больше - модель недооценивает
    риск, сильно меньше - перестраховывается::

        qr.var_backtest("SBER", level=95, method="parametric")
    """
    a = to_asset(x)
    r = a.returns.to_numpy()
    v = var(a, level=level, method=method)
    breaches = int((r < v).sum())
    expected = int(round(len(r) * (1 - level / 100)))
    return pd.Series(
        {
            "VaR, %": round(v * 100, 2),
            "Превышений (факт)": breaches,
            "Ожидалось": expected,
            "Доля превышений, %": round(breaches / len(r) * 100, 2),
        },
        name=a.name,
    )
