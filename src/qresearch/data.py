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

# Браузерные заголовки: защита ISS отклоняет запросы с небраузерным
# User-Agent (например, "qresearch/0.1.0"), возвращая JSON без данных.
_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}

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

# Запасные доски торгов на случай, если определить основную не удалось.
_FALLBACK_BOARDS = (
    ("stock", "shares", "TQBR"),   # акции
    ("stock", "index", "RTSI"),    # индексы
    ("stock", "bonds", "TQOB"),    # облигации
)

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
    """GET с браузерными заголовками; ответ обязан быть JSON-словарём."""
    req = urllib.request.Request(url, headers=_HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        snippet = raw[:200].replace("\n", " ")
        raise ValueError(
            "МосБиржа вернула не-JSON ответ (вероятно, сработала защита "
            f"сайта). Начало ответа: {snippet!r}"
        ) from e


def _iss_block(js: dict, name: str) -> pd.DataFrame:
    """Разворачивает блок ISS-ответа (columns + data) в DataFrame."""
    block = js.get(name)
    if not isinstance(block, dict) or "columns" not in block:
        snippet = json.dumps(js, ensure_ascii=False)[:300]
        raise ValueError(
            f"Неожиданный ответ МосБиржи: нет блока «{name}». "
            f"Начало ответа: {snippet!r}"
        )
    cols = [c["name"] for c in block["columns"]]
    return pd.DataFrame(block["data"], columns=cols)


class _EmptyHistory(Exception):
    """По этой доске торгов истории нет (пробуем следующую доску)."""


def _fetch_history(ticker: str, engine: str, market: str, board: str,
                   start, end) -> pd.DataFrame:
    """Постранично скачивает историю с одной доски торгов."""
    base = (
        f"{_MOEX_BASE}/history/engines/{engine}/markets/{market}"
        f"/boards/{board}/securities/{urllib.parse.quote(ticker)}.json"
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
        cur = js.get("history.cursor")
        data = cur.get("data", []) if isinstance(cur, dict) else []
        total = int(data[0][1]) if data and len(data[0]) > 1 else 0
        if total and cursor >= total:
            break
        if len(page) < 100:  # страница меньше максимальной - данные кончились
            break
    if not frames:
        raise _EmptyHistory(f"{board}: история пуста")
    return pd.concat(frames, ignore_index=True)


def _moex_candidates(ticker: str) -> list:
    """Возможные (engine, market, board) для тикера, от лучших к запасным."""
    if ticker in _INDEX_BOARDS:
        return [("stock", "index", _INDEX_BOARDS[ticker][0])]
    try:
        js = _get_json(
            f"{_MOEX_BASE}/securities/{urllib.parse.quote(ticker)}.json"
            "?iss.only=securities&iss.meta=off"
        )
        df = _iss_block(js, "securities")
        if not df.empty:
            row = df.iloc[0]
            group = str(row.get("GROUP", "")).lower()
            board = str(row.get("PRIMARYBOARDID") or row.get("BOARDID") or "")
            if "index" in group:
                return [("stock", "index", board or "RTSI")]
            if "bond" in group:
                return [("stock", "bonds", board or "TQOB")]
            return [("stock", "shares", board or "TQBR")]
    except Exception:
        pass  # разведка не удалась: переберём типовые доски торгов
    return list(_FALLBACK_BOARDS)


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
        Код доски торгов; по умолчанию определяется автоматически
        (если не удалось - перебираются типовые доски).
    """
    ticker = _resolve_alias(ticker)
    if start is None:
        start = _INDEX_BOARDS.get(ticker, ("", "2010-01-01"))[1]

    candidates = _moex_candidates(ticker)
    if board is not None:
        candidates = [(candidates[0][0], candidates[0][1], board)]

    errors: list[str] = []
    for engine, market, board_id in candidates:
        try:
            rows = _fetch_history(ticker, engine, market, board_id, start, end)
        except _EmptyHistory as e:
            errors.append(str(e))
            continue
        except Exception as e:
            errors.append(f"{board_id}: {e}")
            continue
        df = pd.DataFrame({DATE_COL: rows["TRADEDATE"], PRICE_COL: rows["CLOSE"]})
        return Asset(df, name=name or ticker)

    raise ValueError(
        f"МосБиржа не вернула данных для «{ticker}». Попытки: "
        + "; ".join(errors)
    )


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
            f"Не удалось загрузить «{ticker}»: {e}. "
            "Свои данные можно загрузить через qr.load_file(...)."
        ) from e
    _cache_put(key, asset)
    return asset
