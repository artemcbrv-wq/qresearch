"""Загрузка данных: МосБиржа (ISS API) и локальные CSV/Excel файлы.

Тикеры акций и индексов МосБиржи загружаются с публичного API iss.moex.com.
Свои данные читаются из файлов через :func:`load_file`. Загруженные активы
кэшируются в памяти, поэтому повторные вызовы не ходят в сеть.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from ._core import Asset, DATE_COL, PRICE_COL

_MOEX_BASE = "https://iss.moex.com/iss"
_CACHE: dict = {}

# Известные индексы МосБиржи: тикер -> (доска торгов, дата начала истории).
_INDEX_BOARDS = {
    "IMOEX": ("RTSI", "2010-09-22"),
    "RTSI": ("RTSI", "2010-09-22"),
    "MOEXBC": ("RTSI", "2010-12-30"),
}

# Человекочитаемые псевдонимы индексов (можно передавать в любую функцию).
INDEX_PRESETS = {
    "Индекс МосБиржи": "IMOEX",
    "Индекс РТС": "RTSI",
    "MOEX Blue Chips": "MOEXBC",
}

_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251")
_SEPARATORS = (",", ";", "\t")


def _resolve_alias(ticker: str) -> str:
    """Заменяет человекочитательный псевдоним на настоящий тикер."""
    if not isinstance(ticker, str):
        raise TypeError(f"Тикер должен быть строкой, получен {type(ticker)!r}.")
    for alias, code in INDEX_PRESETS.items():
        if ticker.strip().lower() == alias.lower():
            return code
    return ticker.strip()


def _get_json(url: str) -> dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": "qresearch/0.1.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _iss_block(js: dict, name: str) -> pd.DataFrame:
    """Разворачивает блок ISS-ответа (columns + data) в DataFrame."""
    block = js[name]
    cols = [c["name"] for c in block["columns"]]
    return pd.DataFrame(block["data"], columns=cols)


def _moex_resolve(ticker: str):
    """Определяет (engine, market, board) тикера МосБиржи через /iss/securities."""
    if ticker in _INDEX_BOARDS:
        return "stock", "index", _INDEX_BOARDS[ticker][0]
    js = _get_json(
        f"{_MOEX_BASE}/securities/{urllib.parse.quote(ticker)}.json"
    )
    df = _iss_block(js, "securities")
    if df.empty:
        raise ValueError(f"Тикер «{ticker}» не найден на МосБирже.")
    row = df.iloc[0]
    group = str(row.get("GROUP", "")).lower()
    board = str(row.get("PRIMARYBOARDID") or row.get("BOARDID") or "")
    if "index" in group:
        return "stock", "index", board or "RTSI"
    if "shares" in group:
        return "stock", "shares", board or "TQBR"
    if "bond" in group:
        return "stock", "bonds", board or "TQOB"
    return "stock", "shares", board or "TQBR"


def load_moex(ticker: str, start=None, end=None, board=None, name=None) -> Asset:
    """История тикера с публичного API МосБиржи (ISS) постранично.

    Параметры
    ----------
    ticker :
        Тикер МосБиржи (например, ``"SBER"``, ``"IMOEX"``) или псевдоним
        из :data:`INDEX_PRESETS`.
    start, end :
        Границы периода в формате ``"ГГГГ-ММ-ДД"``.
    board :
        Код доски торгов; по умолчанию определяется автоматически.
    """
    ticker = _resolve_alias(ticker)
    engine, market, board_id = _moex_resolve(ticker)
    if board is not None:
        board_id = board
    if start is None:
        start = _INDEX_BOARDS.get(ticker, ("", "2010-01-01"))[1]

    base = (
        f"{_MOEX_BASE}/history/engines/{engine}/markets/{market}"
        f"/boards/{board_id}/securities/{urllib.parse.quote(ticker)}.json"
    )
    frames: list[pd.DataFrame] = []
    cursor = 0
    while True:
        params = urllib.parse.urlencode(
            {"from": start, "till": end or "", "start": cursor}
        )
        js = _get_json(f"{base}?{params}")
        page = _iss_block(js, "history")
        if page.empty:
            break
        frames.append(page)
        cursor += len(page)
        cur = js.get("history.cursor", {}).get("data", [])
        total = int(cur[0][1]) if cur and len(cur[0]) > 1 else 0
        if total and cursor >= total:
            break
        if len(page) < 100:  # страница меньше максимальной - данные кончились
            break

    if not frames:
        raise ValueError(
            f"МосБиржа не вернула данных для «{ticker}». Проверьте тикер, "
            "доску торгов и период."
        )
    rows = pd.concat(frames, ignore_index=True)
    df = pd.DataFrame({DATE_COL: rows["TRADEDATE"], PRICE_COL: rows["CLOSE"]})
    return Asset(df, name=name or ticker)


def load_file(path, date_col=None, price_col=None, name=None) -> Asset:
    """Загружает цены из вашего CSV или Excel файла.

    Разделитель и кодировка CSV определяются автоматически. Если колонки
    называются нестандартно, укажите их явно через ``date_col``/``price_col``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {p}")
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        df = None
        errors = []
        for enc in _ENCODINGS:
            for sep in _SEPARATORS:
                try:
                    df = pd.read_csv(p, sep=sep, encoding=enc)
                    if df.shape[1] >= 2:
                        break
                except (UnicodeDecodeError, pd.errors.ParserError) as e:
                    errors.append(str(e))
            if df is not None and df.shape[1] >= 2:
                break
        if df is None or df.shape[1] < 2:
            raise ValueError(
                f"Не удалось прочитать «{p.name}» как таблицу с датой и "
                f"ценой. Детали: {errors[-1] if errors else 'пустой файл'}"
            )
    if date_col is not None and price_col is not None:
        df = pd.DataFrame({DATE_COL: df[date_col], PRICE_COL: df[price_col]})
    elif date_col is not None or price_col is not None:
        raise ValueError("Укажите одновременно date_col и price_col.")
    return Asset(df, name=name or p.stem)


def _cache_put(key, asset):
    if len(_CACHE) > 128:  # простая защита от разрастания
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = asset


def load(x, start=None, end=None, name=None) -> Asset:
    """Загружает актив с МосБиржи по тикеру (с кэшированием в памяти).

    Работает и с ``Asset``/``pd.Series``/``pd.DataFrame`` (проходят
    насквозь без изменений), и с псевдонимами из :data:`INDEX_PRESETS`.
    Для своих данных из файла используйте :func:`load_file`.
    """
    from ._core import to_asset

    if isinstance(x, Asset):
        return x
    if not isinstance(x, str):
        return to_asset(x, name=name)
    ticker = _resolve_alias(x)
    key = (ticker, str(start), str(end))
    if key in _CACHE:
        return _CACHE[key]
    try:
        asset = load_moex(ticker, start=start, end=end, name=name)
    except Exception as e:
        raise ValueError(
            f"Не удалось загрузить «{ticker}» с МосБиржи. Проверьте тикер "
            "и доступность iss.moex.com, либо загрузите свои данные из "
            f"файла через qr.load_file(...). Причина: {e}"
        ) from e
    _cache_put(key, asset)
    return asset
