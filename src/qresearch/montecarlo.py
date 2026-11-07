"""Монте-Карло: моделируем годовую доходность тремя способами.

1. iid-бутстреп - каждый день года берём наугад из истории независимо;
2. блочный бутстреп - наугад берём целые куски по ``block`` дней подряд,
   так сохраняется кластеризация волатильности;
3. параметрический - генерируем дни из нормального N(mu, sigma).

Число дней в смоделированном году по умолчанию берётся фактическое -
сколько наблюдений в году у самого актива (например, 251 у индексов
МосБиржи, 246 у индексов денежного рынка), а не круглые 252.

Для каждого метода генератор создаётся заново с тем же seed, поэтому
результаты воспроизводятся цифра в цифру.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._core import to_asset


def monte_carlo(x, days: int | None = None, n_sims: int = 20_000,
                seed: int = 42, block: int = 21) -> dict:
    """Три способа смоделировать год.

    Параметры
    ----------
    days :
        Число торговых дней в смоделированном году. ``None`` (по
        умолчанию) - фактическое число наблюдений в году по истории
        актива: ``round(opy)``.
    n_sims :
        Число симуляций на метод.
    seed :
        Сид генератора (воспроизводимость).
    block :
        Длина блока блочного бутстрепа, дней (21 - торговый месяц).

    Возвращает словарь: имя метода -> массив годовых доходностей
    (доля единицы) длиной ``n_sims``.
    """
    a = to_asset(x)
    r = a.returns.to_numpy()
    n = len(r)
    n_days = int(round(a.opy)) if days is None else int(days)
    out: dict = {}

    # --- Метод 1: iid-бутстреп (дни независимо и с возвращением) --- #
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_sims, n_days))
    out["iid-бутстреп"] = np.prod(1 + r[idx], axis=1) - 1

    # --- Метод 2: блочный бутстреп (цельные куски по block дней) --- #
    rng = np.random.default_rng(seed)
    if n > block:
        n_blocks = int(np.ceil(n_days / block))
        starts = rng.integers(0, n - block, size=(n_sims, n_blocks))
        idx = starts[:, :, None] + np.arange(block)[None, None, :]
        idx = idx.reshape(n_sims, n_blocks * block)[:, :n_days]
    else:  # история короткая, блоки не помещаются - откат к iid
        idx = rng.integers(0, n, size=(n_sims, n_days))
    out[f"блочный бутстреп ({block} дн.)"] = np.prod(1 + r[idx], axis=1) - 1

    # --- Метод 3: параметрическая нормальная симуляция N(mu, sigma) --- #
    rng = np.random.default_rng(seed)
    sims = rng.normal(r.mean(), r.std(ddof=1), size=(n_sims, n_days))
    out["параметрический N(mu, sigma)"] = np.prod(1 + sims, axis=1) - 1
    return out


def mc_summary(sims) -> pd.Series:
    """Сводка по симуляциям: вероятность убытка, среднее, сигма и перцентили, %."""
    sims = np.asarray(sims)
    return pd.Series(
        {
            "P(убыток), %": float((sims < 0).mean() * 100),
            "Средняя, %": float(sims.mean() * 100),
            "σ, %": float(sims.std(ddof=1) * 100),
            "P1, %": float(np.percentile(sims, 1) * 100),
            "P5, %": float(np.percentile(sims, 5) * 100),
            "P50, %": float(np.percentile(sims, 50) * 100),
            "P95, %": float(np.percentile(sims, 95) * 100),
            "P99, %": float(np.percentile(sims, 99) * 100),
        }
    )


def monte_carlo_table(x, days: int | None = None, n_sims: int = 20_000,
                      seed: int = 42, block: int = 21) -> pd.DataFrame:
    """Сводная таблица Монте-Карло по всем трём методам."""
    sims = monte_carlo(x, days=days, n_sims=n_sims, seed=seed, block=block)
    return pd.DataFrame(
        {k: mc_summary(v) for k, v in sims.items()}
    ).T.round(2)
