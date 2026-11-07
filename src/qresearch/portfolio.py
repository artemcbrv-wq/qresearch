"""Портфельная аналитика: риск и доходность с учётом корреляций."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._core import ensure_list, to_asset


class Portfolio:
    """Портфель из нескольких активов с общими датами наблюдений.

    Параметры
    ----------
    assets :
        Список тикеров/``Asset`` (или один объект).
    weights :
        Веса активов; по умолчанию равные. Отрицательные веса запрещены,
        веса автоматически нормируются к сумме 1.
    """

    def __init__(self, assets, weights=None):
        self.assets = [to_asset(a) for a in ensure_list(assets)]
        if not self.assets:
            raise ValueError("Портфель пуст: передайте хотя бы один актив.")
        self.names = [a.name for a in self.assets]
        rets = pd.concat(
            {a.name: a.returns for a in self.assets}, axis=1, join="inner"
        ).dropna()
        if rets.empty:
            raise ValueError("У активов нет общих дат для анализа портфеля.")
        self.rets = rets
        n = len(self.assets)
        if weights is None:
            w = np.full(n, 1.0 / n)
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape != (n,):
                raise ValueError(
                    f"Передано {len(w)} весов для {n} активов."
                )
            if (w < 0).any():
                raise ValueError("Отрицательные веса (шорты) запрещены.")
            if w.sum() <= 0:
                raise ValueError("Сумма весов должна быть положительной.")
            w = w / w.sum()
        self.weights = pd.Series(w, index=self.names, name="Вес")

    # ------------------------------------------------------------------ #
    def portfolio_returns(self) -> np.ndarray:
        """Дневные доходности портфеля (общие даты активов)."""
        return self.rets.to_numpy() @ self.weights.to_numpy()

    @property
    def opy(self) -> float:
        """Наблюдений в год по общему периоду портфеля."""
        days = (self.rets.index[-1] - self.rets.index[0]).days
        years = max(days / 365.25, 1e-9)
        return len(self.rets) / years

    def summary(self) -> pd.Series:
        """Доходность (CAGR), волатильность и Шарп портфеля."""
        pr = self.portfolio_returns()
        eq = np.cumprod(1 + pr)
        years = max(
            (self.rets.index[-1] - self.rets.index[0]).days / 365.25, 1e-9
        )
        cagr = float(eq[-1] ** (1 / years) - 1)
        vol = float(pr.std(ddof=1) * np.sqrt(self.opy))
        return pd.Series(
            {
                "Доходность (CAGR), %": round(cagr * 100, 2),
                "Волатильность, %": round(vol * 100, 2),
                "Шарп (rf = 0), годовых": round(cagr / vol, 2) if vol > 0 else np.inf,
            },
            name="Портфель",
        )

    def correlation(self) -> pd.DataFrame:
        """Матрица корреляций дневных доходностей."""
        return self.rets.corr().round(3)

    def risk_contribution(self) -> pd.Series:
        """Вклад каждого актива в риск портфеля, % (сумма = 100)."""
        cov = self.rets.cov().to_numpy()
        w = self.weights.to_numpy()
        pv = float(w @ cov @ w)
        rc = (cov @ w) * w / pv * 100
        return pd.Series(rc, index=self.names, name="Вклад в риск, %").round(2)

    def min_variance(self) -> pd.Series:
        """Веса портфеля минимальной дисперсии (аналитически).

        В отличие от обычных весов, здесь допускаются отрицательные
        значения (хеджирование), сумма равна 1.
        """
        cov = self.rets.cov().to_numpy()
        cov = cov + np.eye(len(cov)) * 1e-12  # численная стабильность
        inv = np.linalg.inv(cov)
        ones = np.ones(len(cov))
        w = inv @ ones
        w = w / w.sum()
        return pd.Series(w, index=self.names, name="Мин. дисперсия")


def portfolio(assets, weights=None) -> Portfolio:
    """Создаёт :class:`Portfolio` из тикеров или активов."""
    return Portfolio(assets, weights=weights)
