"""Диагностика ряда доходностей: нормальность, автокорреляция, кластеризация волатильности, стационарность.

Тесты Льюнг-Бокса и Дики-Фуллера считаются через statsmodels и
возвращают p-value: вероятность увидеть такую статистику, если
нулевая гипотеза верна. Вывод при 5% уровне значимости.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _sps
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from ._core import to_asset


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


def _lb(vals: np.ndarray, lags) -> dict:
    """Q-статистики и p-value Льюнг-Бокса: {лаг: (Q, p)}."""
    lags = sorted(lags)
    if len(vals) <= max(lags):
        raise ValueError(
            f"История слишком короткая для теста Льюнг-Бокса: нужно "
            f"больше {max(lags)} наблюдений, есть {len(vals)}."
        )
    df = acorr_ljungbox(vals, lags=lags, return_df=True)
    return {
        lag: (float(df.loc[lag, "lb_stat"]), float(df.loc[lag, "lb_pvalue"]))
        for lag in lags
    }


def autocorrelation_test(x, lags=(5, 20)) -> pd.Series:
    """Автокорреляция самих доходностей: Льюнг-Бокс по заданным лагам.

    Заметная автокорреляция означает, что «вчерашний день помогает
    предсказать сегодняшний»: наблюдения зависимы, и модели,
    предполагающие независимость (например, iid-бутстреп), занижают риск.
    Это отдельное свойство, не путать с кластеризацией волатильности
    (автокорреляцией квадратов).
    """
    a = to_asset(x)
    res = _lb(a.returns.to_numpy(), lags)
    s: dict = {}
    for lag, (q, p) in res.items():
        s[f"Q (лаг {lag})"] = round(q, 2)
        s[f"p-value (лаг {lag})"] = round(p, 4)
    p_min = min(p for _, p in res.values())
    s["Вывод"] = (
        "автокорреляция доходностей есть" if p_min < 0.05
        else "автокорреляция не обнаружена"
    )
    return pd.Series(s, name=a.name)


def volatility_clustering_test(x, lags=(5, 20)) -> pd.Series:
    """Кластеризация волатильности: Льюнг-Бокс по квадратам доходностей.

    Если «плохие дни ходят группами», квадраты доходностей
    автокоррелируют, и тест это обнаруживает.
    """
    a = to_asset(x)
    r = a.returns.to_numpy()
    res = _lb((r - r.mean()) ** 2, lags)
    s: dict = {}
    for lag, (q, p) in res.items():
        s[f"Q (лаг {lag})"] = round(q, 2)
        s[f"p-value (лаг {lag})"] = round(p, 4)
    p_min = min(p for _, p in res.values())
    s["Вывод"] = (
        "кластеризация волатильности есть" if p_min < 0.05
        else "кластеризация не обнаружена"
    )
    return pd.Series(s, name=a.name)


def stationarity_test(x) -> pd.Series:
    """Стационарность доходностей: тест Дики-Фуллера (ADF, statsmodels).

    H0 - единичный корень (случайное блуждание). Лаг подбирается
    автоматически (критерий AIC). p-value < 0.05 означает, что ряд
    стационарен: случайное блуждание отвергается.
    """
    a = to_asset(x)
    r = a.returns.to_numpy()
    if len(r) < 20:
        raise ValueError(
            "История слишком короткая для теста ADF: нужно не меньше 20 наблюдений."
        )
    if np.all(r == r[0]):
        raise ValueError("Доходности постоянны, тест ADF неприменим.")
    res = adfuller(r, autolag="AIC")
    t_stat, p_value = float(res[0]), float(res[1])
    crit5 = float(res[4]["5%"])
    return pd.Series(
        {
            "t-статистика (ADF)": round(t_stat, 2),
            "p-value (ADF)": round(p_value, 4),
            "Критическое значение (5%)": round(crit5, 2),
            "Вывод": (
                "стационарен (выброс гипотезы случайного блуждания)"
                if p_value < 0.05
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
