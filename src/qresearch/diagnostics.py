"""Диагностика модели: нормальность, кластеризация волатильности, стационарность."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _sps

from ._core import to_asset

_ADF_CRIT_5 = -2.86  # приближённое критическое значение MacKinnon (5%)


def normality_tests(x) -> pd.Series:
    """Тесты нормальности дневных доходностей: Жарк-Бера и Шапиро-Уилка."""
    a = to_asset(x)
    r = a.returns.to_numpy()
    jb_stat, jb_p = _sps.jarque_bera(r)
    sample = r if len(r) <= 5000 else r[:5000]  # Шапиро-Уилка: предел 5000
    sw_stat, sw_p = _sps.shapiro(sample)
    rejected = bool(jb_p < 0.05 or sw_p < 0.05)
    return pd.Series(
        {
            "Жарк-Бера: статистика": round(float(jb_stat), 2),
            "Жарк-Бера: p-value": round(float(jb_p), 4),
            "Шапиро-Уилка: статистика": round(float(sw_stat), 3),
            "Шапиро-Уилка: p-value": round(float(sw_p), 4),
            "Вывод": (
                "нормальность отвергается" if rejected
                else "нормальность не отвергается"
            ),
        },
        name=a.name,
    )


def volatility_clustering_test(x, lags=(1, 5, 10, 20)) -> pd.Series:
    """Кластеризация волатильности: Ljung-Box по квадратам доходностей.

    Если «плохие дни ходят группами», то квадраты доходностей
    автокоррелируют, и тест её обнаруживает.
    """
    a = to_asset(x)
    r = a.returns.to_numpy()
    y = (r - r.mean()) ** 2
    n = len(y)

    def rho(k: int) -> float:
        return float(np.corrcoef(y[:-k], y[k:])[0, 1])

    q_stat = n * (n + 2) * sum(rho(k) ** 2 / (n - k) for k in lags)
    p_value = float(_sps.chi2.sf(q_stat, len(lags)))
    return pd.Series(
        {
            "Q (Ljung-Box)": round(float(q_stat), 2),
            "p-value": round(p_value, 4),
            "Вывод": (
                "кластеризация волатильности есть" if p_value < 0.05
                else "кластеризации не обнаружено"
            ),
        },
        name=a.name,
    )


def stationarity_test(x) -> pd.Series:
    """Стационарность доходностей: тест Дики-Фуллера (AR(1) без константы).

    Упрощённая версия ADF: t-статистика наклона в регрессии
    r_t на r_{t-1}, сравнивается с критическим значением 5%.
    """
    a = to_asset(x)
    r = a.returns.to_numpy()
    slope, _ = np.polyfit(r[:-1], r[1:], 1)
    resid = r[1:] - (slope * r[:-1])
    sxx = float(np.sum((r[:-1] - r[:-1].mean()) ** 2))
    if sxx == 0:
        raise ValueError("Доходности постоянны, тест неприменим.")
    se = float(np.sqrt(resid @ resid / (len(r) - 2) / sxx))
    t_stat = float((slope - 1) / se)
    return pd.Series(
        {
            "t-статистика (ADF)": round(t_stat, 2),
            "Критическое значение (5%)": _ADF_CRIT_5,
            "Вывод": (
                "стационарен (выброс гипотезы случайного блуждания)"
                if t_stat < _ADF_CRIT_5
                else "не отвергаем случайное блуждание"
            ),
        },
        name=a.name,
    )


def diagnostics(x, print_: bool = False) -> dict:
    """Все три диагностики сразу.

    Возвращает словарь из трёх Series: нормальность, кластеризация
    волатильности, стационарность.
    """
    a = to_asset(x)
    out = {
        "Нормальность": normality_tests(a),
        "Кластеризация волатильности": volatility_clustering_test(a),
        "Стационарность": stationarity_test(a),
    }
    if print_:
        print(f"\nДиагностика модели: {a.name}\n" + "-" * 60)
        for title, series in out.items():
            print(f"\n{title}")
            print(series.to_string())
    return out
