"""Ядро qresearch: класс Asset, нормализация и очистка данных.

Контракт всей библиотеки: аргумент-актив в любой функции можно передать
как тикер (строку), pandas-объект или готовый :class:`Asset`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Единственные имена колонок внутри библиотеки.
DATE_COL = "TRADEDATE"
PRICE_COL = "CLOSE"

_DATE_CANDIDATES = (
    "tradedate", "date", "datetime", "timestamp", "time", "day",
    "дата", "торговая дата",
)
_PRICE_CANDIDATES = (
    "close", "adj close", "adj_close", "price", "last", "waprice",
    "value", "цена", "цена закрытия", "значение", "уровень",
)


class Asset:
    """Один временной ряд цен с вычисляемыми свойствами.

    Параметры
    ----------
    data :
        ``pd.Series`` с датами в индексе, либо ``pd.DataFrame`` /
        ``dict`` с колонками даты и цены (имена угадываются автоматически).
    name :
        Человекочитаемое имя актива.

    Атрибуты
    --------
    prices : pd.Series
        Цены (индекс - даты, отсортированы, дубликаты удалены).
    returns : pd.Series
        Дневные простые доходности (первая отброшена как NaN).
    dates : pd.DatetimeIndex
        Даты наблюдений.
    opy : float
        Наблюдений в год: длина истории, делённая на число лет.
    """

    __slots__ = ("name", "_prices", "_returns")

    def __init__(self, data, name: str | None = None):
        df = _clean(_normalize(data))
        if df.empty:
            raise ValueError(
                "Не удалось распознать данные: нужны колонка с датой и "
                "колонка с ценой (например, TRADEDATE и CLOSE)."
            )
        if name is None:
            name = "Актив"
        self.name = str(name)
        prices = pd.Series(
            df[PRICE_COL].astype(float).to_numpy(),
            index=pd.DatetimeIndex(df[DATE_COL]),
            name=self.name,
        ).sort_index()
        self._prices = prices[~prices.index.duplicated(keep="last")]
        self._returns: pd.Series | None = None

    # ------------------------------------------------------------------ #
    @property
    def prices(self) -> pd.Series:
        return self._prices

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self._prices.index

    @property
    def returns(self) -> pd.Series:
        if self._returns is None:
            self._returns = self._prices.pct_change().dropna()
        return self._returns

    @property
    def opy(self) -> float:
        """Наблюдений в год (длина истории / число лет)."""
        days = (self.dates[-1] - self.dates[0]).days
        years = max(days / 365.25, 1e-9)
        return len(self._prices) / years

    def __len__(self) -> int:
        return len(self._prices)

    def __repr__(self) -> str:
        return (
            f"<Asset {self.name}: {len(self._prices)} наблюдений, "
            f"{self.dates[0].date()} - {self.dates[-1].date()}>"
        )


# ---------------------------------------------------------------------- #
def _find_col(columns, candidates):
    """Ищет колонку по кандидатам без учёта регистра; None, если нет."""
    lookup = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    return None


def _normalize(data) -> pd.DataFrame:
    """Приводит любые входные данные к DataFrame с TRADEDATE и CLOSE."""
    if isinstance(data, pd.Series):
        return pd.DataFrame({DATE_COL: data.index, PRICE_COL: data.to_numpy()})
    if isinstance(data, pd.DataFrame):
        date_col = _find_col(data.columns, _DATE_CANDIDATES)
        price_col = _find_col(data.columns, _PRICE_CANDIDATES)
        if date_col is None or price_col is None:
            # запасной вариант: первая колонка - дата, вторая - цена
            cols = list(data.columns)
            if len(cols) < 2:
                raise ValueError(
                    "В таблице меньше двух колонок: нужны дата и цена."
                )
            date_col, price_col = cols[0], cols[1]
        return pd.DataFrame({DATE_COL: data[date_col], PRICE_COL: data[price_col]})
    if isinstance(data, dict):
        return _normalize(pd.DataFrame(data))
    raise TypeError(
        f"Неподдерживаемый тип данных: {type(data)!r}. Ожидается pd.Series, "
        "pd.DataFrame, dict или тикер (строка)."
    )


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка: даты, положительные цены, сортировка, без дубликатов."""
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, PRICE_COL])
    df = df[df[PRICE_COL] > 0]
    df = df.sort_values(DATE_COL).drop_duplicates(subset=DATE_COL, keep="last")
    return df.reset_index(drop=True)


def to_asset(x, name: str | None = None) -> Asset:
    """Превращает тикер/Series/DataFrame/Asset в :class:`Asset`.

    Строка загружается с МосБиржи через :func:`qresearch.data.load`.
    ``Asset`` возвращается как есть.
    """
    if isinstance(x, Asset):
        return x
    if isinstance(x, str):
        from . import data as _data

        return _data.load(x, name=name)
    if isinstance(x, (pd.Series, pd.DataFrame, dict)):
        return Asset(x, name=name)
    raise TypeError(f"Не удалось превратить в Asset объект типа {type(x)!r}.")


def ensure_list(x) -> list:
    """Оборачивает одиночный объект в список; списки проходят как есть."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]
