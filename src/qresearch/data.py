"""Загрузка данных: МосБиржа (ISS API) и локальные CSV/Excel файлы.

Тикеры акций и индексов МосБиржи загружаются с публичного API iss.moex.com
(открытый доступ, без регистрации и ключей, рыночные данные с задержкой
около 15 минут). Свои данные читаются из файлов через :func:`load_file`.
Загруженные активы кэшируются в памяти, поэтому повторные вызовы не ходят
в сеть.
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
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

# Известные индексы МосБиржи: тикер -> (доска торгов, дата начала истории).
# Индексы живут на рынке index: основная доска SNDX, индексы РТС - RTSI.
_INDEX_BOARDS = {
    "IMOEX": ("SNDX", "2010-09-22"),
    "RTSI": ("RTSI", "2010-09-22"),
    "MOEXBC": ("SNDX", "2010-12-30"),
}

# Человекочитаемые псевдонимы индексов (можно передавать в любую функцию).
INDEX_PRESETS = {
    "Индекс МосБиржи": "IMOEX",
    "Индекс РТС": "RTSI",
    "MOEX Blue Chips": "MOEXBC",
}

# Запасные доски торгов, если определить основную не удалось:
# TQBR - акции (основной режим Т+), TQTF - фонды и ETF,
# SNDX - индексы, RTSI - индексы РТС, TQOB - ОФЗ, TQCB - облигации.
_FALLBACK_BOARDS = (
    ("stock", "shares", "TQBR"),
    ("stock", "shares", "TQTF"),
    ("stock", "index", "SNDX"),
    ("stock", "index", "RTSI"),
    ("stock", "bonds", "TQOB"),
    ("stock", "bonds", "TQCB"),
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


def _iss_block(js, name: str, required: bool = True) -> pd.DataFrame:
    """Разворачивает блок ISS-ответа (columns + data) в DataFrame.

    ISS менял формат ответа: раньше колонки приходили списком словарей
    ``[{"name": ...}]``, теперь метаданные вынесены в отдельную секцию
    ``metadata``, а ``columns`` - простой список строк. Поддерживаем оба
    варианта, а также строки-записи и словари-записи в ``data``.
    """
    block = js.get(name) if isinstance(js, dict) else None
    if not isinstance(block, dict):
        if not required:
            return pd.DataFrame()
        snippet = json.dumps(js, ensure_ascii=False, default=str)[:300]
        raise ValueError(
            f"Неожиданный ответ МосБиржи: нет блока «{name}». "
            f"Начало ответа: {snippet!r}"
        )
    raw = block.get("columns")
    if isinstance(raw, list):
        cols = [c["name"] if isinstance(c, dict) else str(c) for c in raw]
    elif isinstance(raw, str):
        cols = [c.strip() for c in raw.split(",") if c.strip()]
    else:
        cols = []
    if not cols:
        raise ValueError(f"В блоке «{name}» нет списка колонок: {str(block)[:200]}")
    data = block.get("data")
    if not isinstance(data, list) or not data:
        return pd.DataFrame(columns=cols)
    rows = [r for r in data if isinstance(r, (list, dict))]
    if not rows:
        return pd.DataFrame(columns=cols)
    if isinstance(rows[0], dict):
        df = pd.DataFrame(rows)
        keep = [c for c in cols if c in df.columns] or list(df.columns)
        return df[keep]
    width = min(len(cols), min(len(r) for r in rows))
    return pd.DataFrame([r[:width] for r in rows], columns=cols[:width])


def _pick(df: pd.DataFrame, *names):
    """Возвращает первую найденную колонку df (без учёта регистра) или None."""
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        col = lower.get(n.lower())
        if col is not None:
            return df[col]
    return None


class _EmptyHistory(Exception):
    """По этой доске торгов истории нет (пробуем следующую)."""


def _fetch_pages(url: str, base_params: dict, block: str,
                 cursor_block: str) -> pd.DataFrame:
    """Постранично скачивает все строки одного блока ISS-ответа.

    Пагинация - параметром ``start``, полный объём подсказывает служебный
    блок ``*.cursor`` (колонки INDEX, TOTAL, PAGESIZE).
    """
    frames: list = []
    cursor = 0
    for _ in range(100):  # не больше 100 страниц = 10 000 строк
        params = dict(base_params)
        params["start"] = cursor
        js = _get_json(f"{url}?{urllib.parse.urlencode(params)}")
        page = _iss_block(js, block)
        if page.empty:
            break
        clean = _page_dates_prices(page)
        if not clean.empty:
            frames.append(clean)  # чистые страницы: без пустых и NA-колонок
        cursor += len(page)
        cur = _iss_block(js, cursor_block, required=False)
        if not cur.empty and cur.shape[1] > 1:
            try:
                total = int(cur.iloc[0, 1])
            except (TypeError, ValueError):
                total = 0
            if total and cursor >= total:
                break
        if len(page) < 100:  # страница меньше максимальной - данные кончились
            break
    if not frames:
        raise _EmptyHistory(f"«{block}»: история пуста")
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=DATE_COL).reset_index(drop=True)
    if out.empty:
        raise _EmptyHistory("нет строк с ценой закрытия")
    return out


def _page_dates_prices(page: pd.DataFrame) -> pd.DataFrame:
    """Оставляет из одной страницы ISS-ответа дату и цену закрытия."""
    dates = _pick(page, "TRADEDATE", "end", "begin")
    prices = _pick(page, "CLOSE", "lastprice")
    if dates is None or prices is None:
        raise _EmptyHistory("в ответе нет колонок с датой и ценой закрытия")
    df = pd.DataFrame(
        {
            DATE_COL: dates.astype(str).str.slice(0, 10),
            PRICE_COL: pd.to_numeric(prices, errors="coerce"),
        }
    )
    return df.dropna().drop_duplicates(subset=DATE_COL)


def _fetch_history(ticker: str, engine: str, market: str, board: str,
                   start, end) -> pd.DataFrame:
    """История торгов с одной доски: /iss/history/engines/.../securities/X.json"""
    url = (
        f"{_MOEX_BASE}/history/engines/{engine}/markets/{market}"
        f"/boards/{board}/securities/{urllib.parse.quote(ticker)}.json"
    )
    params = {"from": start, "iss.only": "history,history.cursor"}
    if end:
        params["till"] = end
    rows = _fetch_pages(url, params, "history", "history.cursor")
    return rows


def _fetch_candles(ticker: str, engine: str, market: str, start, end,
                   board=None) -> pd.DataFrame:
    """Дневные свечи (interval=24): /iss/engines/.../securities/X/candles.json"""
    url = f"{_MOEX_BASE}/engines/{engine}/markets/{market}"
    if board:
        url += f"/boards/{board}"
    url += f"/securities/{urllib.parse.quote(ticker)}/candles.json"
    params = {"interval": 24, "from": start, "iss.only": "candles,candles.cursor"}
    if end:
        params["till"] = end
    rows = _fetch_pages(url, params, "candles", "candles.cursor")
    return rows


def _moex_candidates(ticker: str) -> list:
    """Возможные (engine, market, board) для тикера, от лучших к запасным."""
    if ticker in _INDEX_BOARDS:
        first = _INDEX_BOARDS[ticker][0]
        second = "RTSI" if first != "RTSI" else "SNDX"
        return [("stock", "index", first), ("stock", "index", second)]
    group, board = "", ""
    try:
        js = _get_json(
            f"{_MOEX_BASE}/securities/{urllib.parse.quote(ticker)}.json"
            "?iss.only=securities"
        )
        df = _iss_block(js, "securities")
        if not df.empty:
            g = _pick(df, "group")
            b = _pick(df, "primaryboardid", "boardid")
            if g is not None:
                group = str(g.iloc[0]).lower()
            if b is not None:
                board = str(b.iloc[0])
    except Exception:
        pass  # разведка не удалась: переберём типовые доски торгов
    if "index" in group:
        base, boards = ("stock", "index"), [board or "SNDX", "SNDX", "RTSI"]
    elif "bond" in group:
        base, boards = ("stock", "bonds"), [board or "TQOB", "TQOB", "TQCB"]
    elif "fund" in group or "etf" in group or "ppif" in group:
        base, boards = ("stock", "shares"), [board or "TQTF", "TQTF", "TQBR"]
    elif group:
        base, boards = ("stock", "shares"), [board or "TQBR", "TQBR", "TQTF"]
    else:
        return list(_FALLBACK_BOARDS)
    seen, out = set(), []
    for b in boards:
        if b and b not in seen:
            seen.add(b)
            out.append((base[0], base[1], b))
    return out


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
    if board is not None and candidates:
        candidates = [(candidates[0][0], candidates[0][1], board)]

    errors: list[str] = []
    for engine, market, board_id in candidates:
        try:
            df = _fetch_history(ticker, engine, market, board_id, start, end)
        except Exception as e:
            errors.append(f"{board_id}: {e}")
            continue
        return Asset(df, name=name or ticker)

    # Последний шанс: дневные свечи без привязки к доске торгов.
    for engine, market in (("stock", "shares"), ("stock", "index"), ("stock", "bonds")):
        try:
            df = _fetch_candles(ticker, engine, market, start, end)
        except Exception as e:
            errors.append(f"свечи/{market}: {e}")
            continue
        return Asset(df, name=name or ticker)

    raise ValueError(
        f"МосБиржа не вернула данных для «{ticker}» (доски: "
        f"{', '.join(c[2] for c in candidates)}). Попытки: " + "; ".join(errors)
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
