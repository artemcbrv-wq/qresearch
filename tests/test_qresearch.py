"""Тесты qresearch на синтетических данных (интернет не нужен).

Генерируем случайные блуждания цен и проверяем весь конвейер
библиотеки: статистики, риск, диагностику, Монте-Карло и портфели.
"""
import numpy as np
import pandas as pd
import pytest

import qresearch as qr
from qresearch._core import Asset, DATE_COL, PRICE_COL


def make_asset(seed=7, n=1000, name="SYNTH", start="2020-01-01",
               drift=0.0004, vol=0.01) -> Asset:
    """Синтетический актив: геометрическое случайное блуждание цен."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n)
    rets = rng.normal(drift, vol, n)
    prices = 100 * np.cumprod(1 + rets)
    df = pd.DataFrame({DATE_COL: dates, PRICE_COL: prices})
    return Asset(df, name=name)


# --------------------------------------------------------------------- #
# Ядро: Asset и нормализация данных
# --------------------------------------------------------------------- #
def test_asset_basics():
    a = make_asset()
    assert len(a.prices) == 1000
    assert len(a.returns) == 999
    assert a.name == "SYNTH"
    assert a.dates.is_monotonic_increasing
    assert a.opy > 200  # дневные данные - примерно 250 наблюдений в год


def test_asset_from_series():
    s = pd.Series(
        [100.0, 101.0, 100.5], index=pd.bdate_range("2024-01-01", periods=3)
    )
    a = Asset(s, name="S")
    assert abs(a.prices.iloc[-1] - 100.5) < 1e-9
    assert a.returns.iloc[0] == 0.01


def test_asset_rejects_garbage():
    with pytest.raises(ValueError):
        Asset({"foo": [1, 2, 3]})


def test_ensure_list():
    assert qr.ensure_list("SBER") == ["SBER"]
    assert qr.ensure_list(["A", "B"]) == ["A", "B"]
    assert qr.ensure_list(("A",)) == ["A"]


def test_load_file_csv(tmp_path):
    p = tmp_path / "prices.csv"
    df = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=50),
        "close": 100.0 + np.arange(50),
    })
    df.to_csv(p, index=False)
    a = qr.load_file(str(p))
    assert len(a.prices) == 50
    assert a.dates.is_monotonic_increasing
    # явные имена колонок дают тот же результат
    b = qr.load_file(str(p), date_col="date", price_col="close")
    assert abs(b.prices.iloc[-1] - 149.0) < 1e-9


def test_load_unknown_ticker_raises():
    with pytest.raises(ValueError):
        qr.load("NO_SUCH_TICKER_XYZ")


# --------------------------------------------------------------------- #
# Статистики
# --------------------------------------------------------------------- #
def test_daily_stats():
    s = qr.daily_stats(make_asset())
    assert s["Наблюдений (дней)"] == 999
    assert "CAGR (геометрическая), %" in s.index
    assert "Шарп (rf = 0), годовых" in s.index
    # у синтетики с drift=0.04%/день CAGR около 10% годовых
    assert 0 < s["CAGR (геометрическая), %"] < 40
    assert s["Доля отрицательных дней, %"] < 60


def test_daily_stats_short_history():
    # короче окна скользящей волатильности - не должно быть NaN
    s = qr.daily_stats(make_asset(n=40))
    assert "Макс. скользящая волатильность (63 дн.), %" in s.index
    assert s["Макс. скользящая волатильность (63 дн.), %"] > 0


def test_stats_table():
    df = qr.stats_table(
        [make_asset(name="A"), make_asset(seed=1, name="B")]
    )
    assert df.shape[1] == 2
    assert {"A", "B"}.issubset(df.columns)
    assert df.loc["Наблюдений (дней)"].tolist() == [999, 999]


def test_annual_returns():
    ann = qr.annual_returns(make_asset(seed=3, n=1500))
    assert ann.index.min() == 2020
    assert ann.index.max() <= 2026
    # процентные значения разумного масштаба
    assert ann.abs().max() < 200


# --------------------------------------------------------------------- #
# Риск: VaR, CVaR, просадки
# --------------------------------------------------------------------- #
def test_var_cvar():
    a = make_asset(seed=11, vol=0.02)
    v95 = qr.var(a, level=95)
    c99 = qr.cvar(a, level=99)
    v99 = qr.var(a, level=99)
    assert -0.5 < v95 < 0  # исторический VaR отрицателен и разумно мал
    assert c99 <= v99 <= v95  # хвост хуже VaR того же уровня
    vpar = qr.var(a, level=95, method="parametric")
    assert vpar < 0


def test_drawdown():
    a = make_asset(seed=5, vol=0.03)
    dd = qr.drawdown(a)
    assert (dd <= 0).all()
    mdd = qr.max_drawdown(a)
    assert mdd < 0
    assert abs(mdd - dd.min()) < 1e-12
    info = qr.drawdown_info(a)
    assert set(info) == {"Макс. просадка, %", "Пик", "Дно", "Восстановление"}
    assert info["Макс. просадка, %"] == round(mdd * 100, 2)


# --------------------------------------------------------------------- #
# Диагностика
# --------------------------------------------------------------------- #
def test_normality_on_gaussian_data():
    a = make_asset(seed=0, drift=0.0, vol=0.01)
    res = qr.normality_tests(a)
    # для честно нормальных данных p-value не микроскопический
    assert res["Жарк-Бера: p-value"] > 0.001
    # асимметрия около нуля
    assert abs(qr.daily_stats(a)["Асимметрия (skewness)"]) < 0.5


def test_diagnostics_blocks():
    a = make_asset(seed=2, vol=0.02)
    d = qr.diagnostics(a)
    assert set(d) == {"Нормальность", "Кластеризация волатильности",
                      "Стационарность"}
    assert d["Стационарность"]["t-статистика (ADF)"] < 0


def test_stationarity_on_random_walk():
    # дневные доходности - стационарный ряд: ADF должен быть глубже критического
    a = make_asset(seed=8, n=2000)
    t = qr.stationarity_test(a)["t-статистика (ADF)"]
    assert t < -10  # доходности сильно стационарнее случайного блуждания


# --------------------------------------------------------------------- #
# Монте-Карло
# --------------------------------------------------------------------- #
def test_monte_carlo_deterministic():
    a = make_asset(seed=2)
    s1 = qr.monte_carlo(a, n_sims=200, seed=42)
    s2 = qr.monte_carlo(a, n_sims=200, seed=42)
    assert len(s1) == 3
    for k in s1:
        assert s1[k].shape == (200,)
        assert np.array_equal(s1[k], s2[k])  # одинаковый seed - тот же результат


def test_monte_carlo_table():
    a = make_asset(seed=4)
    tab = qr.monte_carlo_table(a, n_sims=500, seed=42)
    assert tab.shape[0] == 3
    assert "P(убыток), %" in tab.columns
    assert (tab["P(убыток), %"] >= 0).all()
    assert (tab["P(убыток), %"] <= 100).all()
    s = qr.mc_summary(np.array([-0.01, 0.02, -0.03, 0.05]))
    assert 0 <= s["P(убыток), %"] <= 100


# --------------------------------------------------------------------- #
# Портфели
# --------------------------------------------------------------------- #
def test_portfolio():
    a = make_asset(seed=1, name="A")
    b = make_asset(seed=2, name="B")
    p = qr.portfolio([a, b], weights=[0.7, 0.3])
    assert abs(p.weights.sum() - 1) < 1e-9
    summ = p.summary()
    assert summ["Волатильность, %"] > 0
    corr = p.correlation()
    assert corr.shape == (2, 2)
    assert (np.diag(corr) == 1.0).all()
    rc = p.risk_contribution()
    assert abs(rc.sum() - 100) < 1.0
    mv = p.min_variance()
    assert abs(mv.sum() - 1) < 1e-6
