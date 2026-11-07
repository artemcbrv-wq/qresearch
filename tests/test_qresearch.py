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


# --------------------------------------------------------------------- #
# Визуализация: таблицы-картинки и Монте-Карло графиком
# --------------------------------------------------------------------- #
def test_plot_table_renders_dataframe():
    a = make_asset(seed=9, name="V")
    ax = qr.plot_table(qr.stats_table(a))
    assert ax is not None
    import matplotlib.pyplot as plt
    plt.close("all")


def test_plot_table_renders_series_with_title():
    ax = qr.plot_table(qr.monte_carlo_table(make_asset(seed=9), n_sims=100),
                       title="Монте-Карло")
    assert ax is not None
    import matplotlib.pyplot as plt
    plt.close("all")


def test_plot_monte_carlo_figure():
    a = make_asset(seed=5)
    ax = qr.plot_monte_carlo(a, n_sims=200, seed=1)
    assert ax is not None
    import matplotlib.pyplot as plt
    plt.close("all")


def test_quick_plot_returns_dataframe():
    df = qr.quick(make_asset(seed=3, name="Q"), plot=False)
    assert "CAGR (геометрическая), %" in df.index
    df2 = qr.quick(make_asset(seed=3, name="Q"), plot=True)
    assert df2.shape == df.shape
    import matplotlib.pyplot as plt
    plt.close("all")


def test_single_metrics_return_numbers():
    a = make_asset(seed=11, drift=0.0004, vol=0.02)
    assert isinstance(qr.cagr(a), float)
    assert qr.volatility(a) > 0
    assert abs(qr.sharpe(a) - (qr.cagr(a) / qr.volatility(a))) < 0.5
    assert qr.max_drawdown(a) <= 0
    assert -1 < qr.var(a, level=95) < 0


# --------------------------------------------------------------------- #
# Новые расчёты: бэктест VaR, автокорреляция, полная статистика
# --------------------------------------------------------------------- #
def test_var_backtest():
    a = make_asset(seed=13)
    bt = qr.var_backtest(a, level=95, method="parametric")
    assert bt["Ожидалось"] == int(round(len(a.returns) * 0.05))
    assert 0 <= bt["Превышений (факт)"] <= len(a.returns)
    assert bt["VaR, %"] < 0


def test_autocorrelation_test_pvalue():
    s = qr.autocorrelation_test(make_asset(seed=2))
    assert 0 <= s["p-value (лаг 5)"] <= 1
    assert 0 <= s["p-value (лаг 20)"] <= 1
    assert "Вывод" in s.index


def test_stationarity_has_pvalue():
    s = qr.stationarity_test(make_asset(seed=8, n=2000))
    assert 0 <= s["p-value (ADF)"] <= 1
    assert s["t-статистика (ADF)"] < 0


def test_full_stats_and_table():
    a = make_asset(seed=4)
    s = qr.full_stats(a)
    assert "Превышений VaR95 (факт/ожидалось)" in s.index
    assert "Льюнг-Бокс p (квадраты, лаги 5/20)" in s.index
    assert "Дно MDD" in s.index
    tab = qr.full_stats_table([a])
    assert tab.shape[1] == 1
    assert tab.shape[0] == len(s)


def test_annual_stats_quartiles():
    s = qr.annual_stats(make_asset(seed=3, n=1500))
    for key in ("Минимум, %", "25%, %", "50%, %", "75%, %", "Максимум, %",
                "Доля положительных лет", "Число убыточных лет"):
        assert key in s.index
    assert s["Минимум, %"] <= s["Максимум, %"]


def test_annual_table_marks_incomplete_year():
    a = make_asset(seed=6, n=300, start="2025-01-01")
    df = qr.annual_table(a)
    assert str(df.index[-1]).endswith("*")
    df2 = qr.annual_table(a, mark_incomplete=False)
    assert not str(df2.index[-1]).endswith("*")


def test_compare_annual_and_sharpe_vs():
    a = make_asset(seed=1, name="A")
    b = make_asset(seed=2, name="B")
    df, corr = qr.compare_annual(a, b)
    assert df.shape[1] == 2
    assert -1.0 <= corr <= 1.0
    assert np.isfinite(qr.sharpe_vs(a, b))


def test_implied_rate():
    a = make_asset(seed=9, drift=0.0004, vol=0.001)
    rate = qr.implied_rate(a)
    assert len(rate) == len(a.returns)
    assert float(rate.mean()) > 0


def test_monte_carlo_actual_days_per_year():
    a = make_asset(seed=5, n=756)  # около трёх лет: opy около 252
    sims = qr.monte_carlo(a, n_sims=100, seed=1)
    assert sims["iid-бутстреп"].shape == (100,)
    assert sims["параметрический N(mu, sigma)"].shape == (100,)


def test_plots_restyled_and_save(tmp_path):
    a = make_asset(seed=11, n=300)
    fig = qr.plot_annual(a, save=str(tmp_path / "annual.png"))
    assert (tmp_path / "annual.png").exists()
    fig_mc = qr.plot_monte_carlo(a, n_sims=200, seed=3,
                                 save=str(tmp_path / "mc.png"))
    assert (tmp_path / "mc.png").exists()
    assert len(fig.axes) == 2      # две панели: бары + распределение
    assert len(fig_mc.axes) == 1   # один актив - одна панель
    qr.plot_table(qr.daily_stats(a), save=str(tmp_path / "table.png"))
    assert (tmp_path / "table.png").exists()



def test_plot_qq_matches_target(tmp_path):
    a = make_asset(seed=12, n=400)
    ax = qr.plot_qq(a, save=str(tmp_path / "qq.png"))
    assert (tmp_path / "qq.png").exists()
    # метод эталона: подписи дословно, панель 6.5x5.5
    assert ax.get_xlabel() == "Теоретические квантили N(0,1)"
    assert ax.get_ylabel() == "Квантили доходностей, %"
    w, h = ax.figure.get_size_inches()
    assert abs(w - 6.5) < 0.01 and abs(h - 5.5) < 0.01
    # обе оси в % с одинаковыми пределами: диагональ y = x под 45 градусов
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert abs((x1 - x0) - (y1 - y0)) < 1e-9
    assert len(ax.lines) == 1  # пунктирная диагональ
    fig = qr.plot_qq([make_asset(seed=1, name="A"), make_asset(seed=2, name="B")],
                     save=str(tmp_path / "qq2.png"))
    assert (tmp_path / "qq2.png").exists()
    assert len(fig.axes) == 2  # по панели на актив
    import matplotlib.pyplot as plt
    plt.close("all")
