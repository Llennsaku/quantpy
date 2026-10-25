"""A-share dividend + low-volatility portfolio backtest.

AkShare is used for adjusted/unadjusted prices, implemented dividends and the
CSI 300 benchmark.  Data is treated point-in-time: a dividend enters the
signal only after its ex-date.

Examples
--------
python dividend_strategy.py all
python dividend_strategy.py backtest --start 20180101 --end 20261023
python dividend_strategy.py optimize --train-end 20221231 --valid-end 20241231
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tushare as ts
import akshare as ak
from dotenv import load_dotenv


# A deliberately explicit research universe.  It avoids silently pretending
# that today's constituents are historical constituents, but still has a
# survivorship/selection bias which is disclosed in the generated report.
UNIVERSE = {
    "000333.SZ": "美的集团",
    "000651.SZ": "格力电器",
    "000895.SZ": "双汇发展",
    "002304.SZ": "洋河股份",
    "600000.SH": "浦发银行",
    "600009.SH": "上海机场",
    "600019.SH": "宝钢股份",
    "600025.SH": "华能水电",
    "600028.SH": "中国石化",
    "600036.SH": "招商银行",
    "600104.SH": "上汽集团",
    "600176.SH": "中国巨石",
    "600177.SH": "雅戈尔",
    "600188.SH": "兖矿能源",
    "600350.SH": "山东高速",
    "600377.SH": "宁沪高速",
    "600585.SH": "海螺水泥",
    "600660.SH": "福耀玻璃",
    "600674.SH": "川投能源",
    "600690.SH": "海尔智家",
    "600741.SH": "华域汽车",
    "600886.SH": "国投电力",
    "600900.SH": "长江电力",
    "600926.SH": "杭州银行",
    "601006.SH": "大秦铁路",
    "601088.SH": "中国神华",
    "601166.SH": "兴业银行",
    "601225.SH": "陕西煤业",
    "601288.SH": "农业银行",
    "601318.SH": "中国平安",
    "601328.SH": "交通银行",
    "601398.SH": "工商银行",
    "601601.SH": "中国太保",
    "601668.SH": "中国建筑",
    "601818.SH": "光大银行",
    "601857.SH": "中国石油",
    "601919.SH": "中远海控",
    "601939.SH": "建设银行",
    "601988.SH": "中国银行",
    "601998.SH": "中信银行",
}

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "dividend_cache"
RESULT_DIR = ROOT / "dividend_results"


@dataclass(frozen=True)
class StrategyConfig:
    top_n: int = 12
    rebalance_months: int = 3
    volatility_days: int = 120
    dividend_window_days: int = 730
    min_dividend_years: int = 2
    yield_weight: float = 0.50
    low_vol_weight: float = 0.30
    consistency_weight: float = 0.20
    commission_rate: float = 0.0001
    transfer_fee_rate: float = 0.00001
    stamp_duty_rate: float = 0.0005
    min_commission: float = 5.0
    initial_cash: float = 1_000_000.0


def tushare_client():
    load_dotenv(ROOT / ".env")
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError(".env 中缺少 TUSHARE_TOKEN")
    # Passing the token directly avoids tushare writing tk.csv into user home.
    return ts.pro_api(token)


def _retry(call: Callable[[], pd.DataFrame], attempts: int = 5) -> pd.DataFrame:
    for attempt in range(attempts):
        try:
            return call()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _download_one(code: str, start: str, end: str, force: bool = False) -> str:
    CACHE_DIR.mkdir(exist_ok=True)
    paths = {
        "daily": CACHE_DIR / f"{code}_daily.csv",
        "adj": CACHE_DIR / f"{code}_adj.csv",
        "dividend": CACHE_DIR / f"{code}_dividend.csv",
    }
    if not force and all(path.exists() for path in paths.values()):
        return f"{code}: cached"

    bare_symbol = code.split(".")[0]
    symbol = ("sh" if code.endswith(".SH") else "sz") + bare_symbol
    raw_source = _retry(
        lambda: ak.stock_zh_a_daily(
            symbol=symbol, start_date=start, end_date=end, adjust=""
        )
    )
    adjusted_source = _retry(
        lambda: ak.stock_zh_a_daily(
            symbol=symbol, start_date=start, end_date=end, adjust="qfq"
        )
    )
    dividend_source = _retry(lambda: ak.stock_fhps_detail_em(symbol=bare_symbol))
    if raw_source.empty or adjusted_source.empty:
        raise RuntimeError(f"{code} 没有日线数据")
    # Use positions instead of Chinese labels so the script also works in a
    # Windows terminal whose legacy code page renders those labels incorrectly.
    daily = pd.DataFrame(
        {"trade_date": raw_source.iloc[:, 0], "close": raw_source.iloc[:, 4]}
    )
    adj = pd.DataFrame(
        {"trade_date": adjusted_source.iloc[:, 0], "adj_close": adjusted_source.iloc[:, 4]}
    )
    if dividend_source.empty:
        dividend = pd.DataFrame(
            columns=["ts_code", "end_date", "ann_date", "div_proc", "cash_div_tax", "record_date", "ex_date"]
        )
    else:
        dividend = pd.DataFrame(
            {
                "ts_code": code,
                "end_date": dividend_source.iloc[:, 0],
                "ann_date": dividend_source.iloc[:, 18],
                "div_proc": dividend_source.iloc[:, 17],
                # Eastmoney reports yuan per 10 shares; normalize to per share.
                "cash_div_tax": pd.to_numeric(dividend_source.iloc[:, 5], errors="coerce") / 10,
                "record_date": dividend_source.iloc[:, 15],
                "ex_date": dividend_source.iloc[:, 16],
            }
        )
    daily.to_csv(paths["daily"], index=False, encoding="utf-8-sig")
    adj.to_csv(paths["adj"], index=False, encoding="utf-8-sig")
    dividend.to_csv(paths["dividend"], index=False, encoding="utf-8-sig")
    return f"{code}: {len(daily)} bars, {len(dividend)} dividend records"


def download_data(start: str, end: str, force: bool = False, workers: int = 2) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    print(f"Downloading {len(UNIVERSE)} stocks ({start} - {end})...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, code, start, end, force): code
            for code in UNIVERSE
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                print(future.result())
            except Exception as exc:
                print(f"WARN {code}: {exc}")

    benchmark_path = CACHE_DIR / "000300.SH_index.csv"
    if force or not benchmark_path.exists():
        source = _retry(lambda: ak.stock_zh_index_daily(symbol="sh000300"))
        benchmark = source[["date", "close"]].rename(columns={"date": "trade_date"})
        benchmark = benchmark[
            (pd.to_datetime(benchmark.trade_date) >= pd.Timestamp(start))
            & (pd.to_datetime(benchmark.trade_date) <= pd.Timestamp(end))
        ]
        benchmark.to_csv(benchmark_path, index=False, encoding="utf-8-sig")


def _read_csv(path: Path, date_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"缺少缓存 {path.name}；请先运行 download 或 all")
    frame = pd.read_csv(path, dtype={column: str for column in date_columns})
    for column in date_columns:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def load_market_data(start: str, end: str):
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    raw_close, adjusted_close, dividends = {}, {}, []

    for code in UNIVERSE:
        daily_path = CACHE_DIR / f"{code}_daily.csv"
        adj_path = CACHE_DIR / f"{code}_adj.csv"
        div_path = CACHE_DIR / f"{code}_dividend.csv"
        if not (daily_path.exists() and adj_path.exists() and div_path.exists()):
            continue
        daily = _read_csv(daily_path, ["trade_date"])
        adj = _read_csv(adj_path, ["trade_date"])
        merged = daily[["trade_date", "close"]].merge(
            adj[["trade_date", "adj_close"]], on="trade_date", how="inner"
        ).sort_values("trade_date")
        merged = merged[(merged.trade_date >= start_ts) & (merged.trade_date <= end_ts)]
        if merged.empty:
            continue
        raw_close[code] = merged.set_index("trade_date")["close"]
        adjusted_close[code] = merged.set_index("trade_date")["adj_close"]

        div = _read_csv(div_path, ["end_date", "ann_date", "record_date", "ex_date"])
        if not div.empty:
            div["ts_code"] = code
            dividends.append(div)

    if not adjusted_close:
        raise RuntimeError("没有可回测的股票缓存")
    raw = pd.DataFrame(raw_close).sort_index().ffill(limit=5)
    adjusted = pd.DataFrame(adjusted_close).sort_index().ffill(limit=5)
    divs = pd.concat(dividends, ignore_index=True) if dividends else pd.DataFrame()
    return raw, adjusted, clean_dividends(divs)


def clean_dividends(dividends: pd.DataFrame) -> pd.DataFrame:
    if dividends.empty:
        return pd.DataFrame(columns=["ts_code", "end_date", "ex_date", "cash_div_tax"])
    frame = dividends.copy()
    frame["cash_div_tax"] = pd.to_numeric(frame["cash_div_tax"], errors="coerce")
    frame = frame.dropna(subset=["ts_code", "ex_date", "cash_div_tax"])
    frame = frame[frame.cash_div_tax > 0]
    if "div_proc" in frame:
        implemented = frame["div_proc"].astype(str).str.contains("实施", na=False)
        # Some old records have a blank process but a real ex-date.
        frame = frame[implemented | frame["div_proc"].isna()]
    frame = frame.sort_values(["ts_code", "end_date", "ex_date", "ann_date"])
    return frame.drop_duplicates(["ts_code", "end_date", "ex_date"], keep="last")


def rebalance_dates(index: pd.DatetimeIndex, months: int) -> pd.DatetimeIndex:
    marker = index.to_period("M")
    first_of_month = pd.Series(index=index, data=marker).groupby(marker).head(1).index
    anchor_month = first_of_month[0].month
    return pd.DatetimeIndex(
        [date for date in first_of_month if (date.month - anchor_month) % months == 0]
    )


def select_portfolio(
    signal_date: pd.Timestamp,
    raw_close: pd.DataFrame,
    daily_returns: pd.DataFrame,
    dividends: pd.DataFrame,
    config: StrategyConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    price = raw_close.loc[:signal_date].iloc[-1]
    volatility = daily_returns.loc[:signal_date].tail(config.volatility_days).std() * np.sqrt(252)
    cutoff = signal_date - pd.Timedelta(days=config.dividend_window_days)
    known = dividends[(dividends.ex_date <= signal_date) & (dividends.ex_date > cutoff)]
    cash = known.groupby("ts_code")["cash_div_tax"].sum()
    years = known.assign(year=known.ex_date.dt.year).groupby("ts_code")["year"].nunique()

    factors = pd.concat(
        [price.rename("price"), cash.rename("cash"), years.rename("years"), volatility.rename("vol")],
        axis=1,
    )
    # Annualize cash paid over the selected trailing window.  This avoids
    # displaying a two-year cumulative cash yield as if it were a one-year
    # dividend yield.
    factors["yield"] = factors.cash / factors.price * (365 / config.dividend_window_days)
    factors = factors.replace([np.inf, -np.inf], np.nan).dropna()
    factors = factors[(factors.price > 0) & (factors["yield"] > 0)]
    factors = factors[factors.years >= config.min_dividend_years]
    if factors.empty:
        return pd.Series(dtype=float), factors
    factors["yield"] = factors["yield"].clip(upper=factors["yield"].quantile(0.95))
    factors["score"] = (
        config.yield_weight * factors["yield"].rank(pct=True)
        + config.low_vol_weight * factors.vol.rank(ascending=False, pct=True)
        + config.consistency_weight * factors.years.rank(pct=True)
    )
    selected = factors.nlargest(min(config.top_n, len(factors)), "score").copy()
    inverse_vol = 1 / selected.vol.clip(lower=0.05)
    weights = inverse_vol / inverse_vol.sum()
    return weights, selected.sort_values("score", ascending=False)


def transaction_cost(turnover: float, portfolio_value: float, positions: int, config: StrategyConfig) -> float:
    if turnover <= 0 or portfolio_value <= 0:
        return 0.0
    # One-way turnover is half the absolute weight change. Approximate buys and
    # sells as half each; stamp duty applies only to sells.
    traded = turnover * portfolio_value
    commission = max(traded * config.commission_rate, config.min_commission * max(positions, 1))
    transfer = traded * config.transfer_fee_rate
    stamp = traded * 0.5 * config.stamp_duty_rate
    return (commission + transfer + stamp) / portfolio_value


def run_backtest(start: str, end: str, config: StrategyConfig):
    raw, adjusted, dividends = load_market_data(start, end)
    returns = adjusted.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0)
    dates = returns.index
    rebalances = set(rebalance_dates(dates, config.rebalance_months))
    weights = pd.Series(dtype=float)
    portfolio_returns = []
    turnover_rows, holdings_rows = [], []
    value = config.initial_cash

    for i, date in enumerate(dates):
        if i == 0:
            portfolio_returns.append(0.0)
            continue
        if date in rebalances or (weights.empty and i == 1):
            signal_date = dates[i - 1]
            new_weights, factors = select_portfolio(
                signal_date, raw, returns, dividends, config
            )
            aligned = weights.reindex(weights.index.union(new_weights.index), fill_value=0)
            target = new_weights.reindex(aligned.index, fill_value=0)
            turnover = float((target - aligned).abs().sum() / 2)
            cost = transaction_cost(turnover, value, len(new_weights), config)
            turnover_rows.append({"date": date, "turnover": turnover, "cost_rate": cost})
            for code, weight in new_weights.items():
                holdings_rows.append(
                    {
                        "rebalance_date": date,
                        "signal_date": signal_date,
                        "ts_code": code,
                        "name": UNIVERSE.get(code, ""),
                        "weight": weight,
                        "yield": factors.loc[code, "yield"],
                        "volatility": factors.loc[code, "vol"],
                        "dividend_years": factors.loc[code, "years"],
                        "score": factors.loc[code, "score"],
                    }
                )
            weights = new_weights
        else:
            cost = 0.0

        day_return = float((returns.loc[date].reindex(weights.index).fillna(0) * weights).sum())
        net_return = day_return - cost
        portfolio_returns.append(net_return)
        value *= 1 + net_return

    strategy = pd.Series(portfolio_returns, index=dates, name="strategy")
    benchmark_file = CACHE_DIR / "000300.SH_index.csv"
    benchmark = _read_csv(benchmark_file, ["trade_date"]).sort_values("trade_date")
    benchmark = benchmark.set_index("trade_date")["close"].pct_change(fill_method=None)
    benchmark = benchmark.reindex(dates).fillna(0).rename("benchmark")
    return strategy, benchmark, pd.DataFrame(holdings_rows), pd.DataFrame(turnover_rows)


def performance(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna()
    if len(returns) < 2:
        return {key: float("nan") for key in ("annual_return", "volatility", "sharpe", "max_drawdown", "calmar")}
    equity = (1 + returns).cumprod()
    years = len(returns) / 252
    annual_return = equity.iloc[-1] ** (1 / years) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() else np.nan
    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min()
    calmar = annual_return / abs(max_drawdown) if max_drawdown else np.nan
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "annual_return": float(annual_return),
        "volatility": float(volatility),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "calmar": float(calmar),
    }


def evaluate_period(returns: pd.Series, start: str, end: str) -> dict[str, float]:
    return performance(returns.loc[pd.Timestamp(start):pd.Timestamp(end)])


def optimize(
    start: str,
    train_end: str,
    valid_end: str,
    end: str,
    base: StrategyConfig,
) -> tuple[StrategyConfig, pd.DataFrame, dict]:
    rows = []
    configs = [
        replace(base, top_n=top_n, rebalance_months=months, volatility_days=vol_days)
        for top_n in (8, 12, 16)
        for months in (3, 6)
        for vol_days in (60, 120)
    ]
    for number, config in enumerate(configs, 1):
        returns, _, _, turnovers = run_backtest(start, end, config)
        train = evaluate_period(returns, start, train_end)
        validation_start = (pd.Timestamp(train_end) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        valid = evaluate_period(returns, validation_start, valid_end)
        mean_turnover = float(turnovers.turnover.mean()) if not turnovers.empty else 0.0
        # Validation is weighted more heavily, while the minimum Sharpe term
        # penalizes parameter sets that only work in one period.
        robust_score = 0.35 * train["sharpe"] + 0.65 * valid["sharpe"]
        robust_score += 0.20 * min(train["sharpe"], valid["sharpe"])
        robust_score -= 0.05 * mean_turnover
        rows.append(
            {
                "top_n": config.top_n,
                "rebalance_months": config.rebalance_months,
                "volatility_days": config.volatility_days,
                "train_sharpe": train["sharpe"],
                "valid_sharpe": valid["sharpe"],
                "train_return": train["annual_return"],
                "valid_return": valid["annual_return"],
                "mean_turnover": mean_turnover,
                "robust_score": robust_score,
            }
        )
        print(f"[{number:02d}/{len(configs)}] {asdict(config)} score={robust_score:.3f}")

    table = pd.DataFrame(rows).sort_values("robust_score", ascending=False)
    best_row = table.iloc[0]
    best = replace(
        base,
        top_n=int(best_row.top_n),
        rebalance_months=int(best_row.rebalance_months),
        volatility_days=int(best_row.volatility_days),
    )
    strategy, benchmark, holdings, turnovers = run_backtest(start, end, best)
    test_start = (pd.Timestamp(valid_end) + pd.Timedelta(days=1)).strftime("%Y%m%d")
    result = {
        "config": asdict(best),
        "periods": {
            "train": [start, train_end],
            "validation": [(pd.Timestamp(train_end) + pd.Timedelta(days=1)).strftime("%Y%m%d"), valid_end],
            "test": [test_start, end],
        },
        "strategy": {
            "full": performance(strategy),
            "test": evaluate_period(strategy, test_start, end),
        },
        "benchmark": {
            "full": performance(benchmark),
            "test": evaluate_period(benchmark, test_start, end),
        },
    }
    save_results(strategy, benchmark, holdings, turnovers, table, result)
    return best, table, result


def save_results(strategy, benchmark, holdings, turnovers, optimization, summary) -> None:
    RESULT_DIR.mkdir(exist_ok=True)
    pd.concat([strategy, benchmark], axis=1).to_csv(
        RESULT_DIR / "daily_returns.csv", encoding="utf-8-sig"
    )
    holdings.to_csv(RESULT_DIR / "holdings.csv", index=False, encoding="utf-8-sig")
    turnovers.to_csv(RESULT_DIR / "turnover.csv", index=False, encoding="utf-8-sig")
    optimization.to_csv(RESULT_DIR / "optimization.csv", index=False, encoding="utf-8-sig")
    with open(RESULT_DIR / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    equity = pd.concat([(1 + strategy).cumprod(), (1 + benchmark).cumprod()], axis=1)
    equity.columns = ["Dividend quality + low vol", "CSI 300"]
    ax = equity.plot(figsize=(11, 6), grid=True, title="Dividend Strategy Backtest")
    ax.set_ylabel("Net value")
    ax.figure.tight_layout()
    ax.figure.savefig(RESULT_DIR / "equity_curve.png", dpi=160)
    plt.close(ax.figure)


def print_summary(result: dict) -> None:
    print("\nBest config:", json.dumps(result["config"], ensure_ascii=False))
    for scope in ("full", "test"):
        strategy = result["strategy"][scope]
        benchmark = result["benchmark"][scope]
        print(
            f"{scope:>4} | strategy CAGR {strategy['annual_return']:.2%}, "
            f"Sharpe {strategy['sharpe']:.2f}, MDD {strategy['max_drawdown']:.2%} | "
            f"CSI300 CAGR {benchmark['annual_return']:.2%}, "
            f"Sharpe {benchmark['sharpe']:.2f}, MDD {benchmark['max_drawdown']:.2%}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="A股红利质量低波策略")
    parser.add_argument("command", choices=["download", "backtest", "optimize", "all"], nargs="?", default="all")
    parser.add_argument("--start", default="20160101")
    parser.add_argument("--train-end", default="20211231")
    parser.add_argument("--valid-end", default="20231231")
    parser.add_argument("--end", default="20261023")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    base = StrategyConfig()
    if args.command in {"download", "all"}:
        download_data(args.start, args.end, args.force, args.workers)
    if args.command == "backtest":
        strategy, benchmark, holdings, turnovers = run_backtest(args.start, args.end, base)
        summary = {
            "config": asdict(base),
            "strategy": {"full": performance(strategy)},
            "benchmark": {"full": performance(benchmark)},
        }
        save_results(strategy, benchmark, holdings, turnovers, pd.DataFrame(), summary)
        print_summary({**summary, "strategy": {"full": performance(strategy), "test": performance(strategy)}, "benchmark": {"full": performance(benchmark), "test": performance(benchmark)}})
    elif args.command in {"optimize", "all"}:
        _, _, result = optimize(args.start, args.train_end, args.valid_end, args.end, base)
        print_summary(result)


if __name__ == "__main__":
    main()
