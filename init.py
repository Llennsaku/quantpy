# %%
import tushare as ts
import pandas as pd
import backtrader as bt
import datetime
import os
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，不弹窗，直接保存图片
from dotenv import load_dotenv

# ==========================================
# 1. 安全加载 Tushare Token（从 .env 文件读取）
# ==========================================
# 请复制 .env.example 为 .env，并填入你的真实 token
# .env 文件已在 .gitignore 中，不会上传到 GitHub
load_dotenv()

token = os.getenv('TUSHARE_TOKEN')
if not token:
    raise ValueError("❌ 未找到 TUSHARE_TOKEN！请复制 .env.example 为 .env 并填入真实 token")
ts.set_token(token)
pro = ts.pro_api()
# %%
# ==========================================
# 2. 定义获取数据的函数（关键步骤）
# ==========================================
def get_tushare_data(code, start_date, end_date):
    """
    使用 pro.daily() 获取日线数据，清洗后对接 Backtrader。
    优先读取本地 CSV 缓存，避免每次重复下载。
    注意：daily 返回的是未复权数据，如需复权可升级权限后用 pro_bar。
    """
    # 生成缓存文件名：如 cache_600519.SH_20200101_20231231.csv
    cache_dir = "stock_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{code}_{start_date}_{end_date}.csv")

    # 如果缓存文件存在，直接读取
    if os.path.exists(cache_file):
        print(f"📂 从缓存加载: {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    # 缓存不存在，从 Tushare 下载
    print(f"⬇️ 正在从 Tushare 下载 {code} 的数据...")
    df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)

    if df is None or df.empty:
        print("❌ 错误：未获取到数据，请检查 Token 或股票代码是否正确。")
        return None

    # --- 数据清洗，对接 Backtrader ---
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    df.sort_index(ascending=True, inplace=True)

    # daily 接口列名: open, high, low, close, vol
    df = df[['open', 'high', 'low', 'close', 'vol']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.astype(float)

    # 保存到 CSV 缓存
    df.to_csv(cache_file)
    print(f"💾 已保存缓存: {cache_file}")

    return df


# ==========================================
# 3.5 A股真实佣金类（不免五 + 印花税）
# ==========================================
class AShareCommission(bt.CommInfoBase):
    """
    我的券商佣金规则：
    - 佣金费率 0.0001（万一），买入/卖出都收
    - 最低佣金 5 元（不免五）
    - 印花税 0.001（千一），仅卖出时收取
    - 过户费 0.00001（0.001%），买入/卖出都收
    """
    params = (
        ('commission', 0.0001),    # 万一佣金费率
        ('stamp_duty', 0.001),     # 千一印花税（仅卖出）
        ('transfer_fee', 0.00001), # 0.001% 过户费（买卖双向）
        ('min_commission', 5.0),   # 最低佣金 5 元
    )

    def _getcommission(self, size, price, pseudoexec):
        """计算单边佣金"""
        turnover = abs(size) * price  # 成交金额
        comm = turnover * self.p.commission
        if comm < self.p.min_commission:
            comm = self.p.min_commission  # 不免五，最低5元
        return comm

    def get_margin(self, price):
        return price  # A股全额保证金（即股票价格）

    def getsize(self, price, cash):
        """
        计算用给定现金最多能买多少股。
        买入成本 = 股数 × 价格 + 佣金（不免五）
        注意：A股买入必须是 100 的整数倍
        """
        if price <= 0:
            return 0
        # 预留佣金，佣金不足5元按5元算
        remaining = cash - self.p.min_commission
        if remaining <= 0:
            return 0
        shares = int(remaining / price)
        # 凑整为100的倍数（1手=100股）
        shares = (shares // 100) * 100
        # 验证：实际佣金是否够
        actual_comm = shares * price * self.p.commission
        if actual_comm < self.p.min_commission:
            actual_comm = self.p.min_commission
        if shares * price + actual_comm > cash:
            shares -= 100  # 退一手保安全
        return max(shares, 0)

    def _getstampduty(self, size, price):
        """印花税，仅卖出（size < 0）时收取"""
        if size < 0:
            return abs(size) * price * self.p.stamp_duty
        return 0.0

    def _gettransferfee(self, size, price):
        """过户费，买入卖出双向收取"""
        return abs(size) * price * self.p.transfer_fee

    def getvaluesize(self, size, price):
        return abs(size) * price

    def getoperationcost(self, size, price):
        """总交易成本 = 佣金 + 印花税 + 过户费"""
        comm = self._getcommission(size, price, pseudoexec=True)
        stamp = self._getstampduty(size, price)
        transfer = self._gettransferfee(size, price)
        return comm + stamp + transfer


# ==========================================
# 4. 策略部分（双均线）
# ==========================================
class DoubleMAStrategy(bt.Strategy):
    params = (('short_period', 5), ('long_period', 20))

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'[{dt.isoformat()}] {txt}')

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.short_ma = bt.indicators.SMA(self.datas[0], period=self.params.short_period)
        self.long_ma = bt.indicators.SMA(self.datas[0], period=self.params.long_period)
        self.order = None

    def notify_order(self, order):
        """订单状态变化时回调——关键：将 self.order 重置为 None"""
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None  # 订单完成/取消后，允许下一次交易

    def next(self):
        if self.order: return  # 有未完成订单则等待

        # 策略逻辑：
        if not self.position:  # 空仓
            if self.short_ma[0] > self.long_ma[0]:  # 金叉
                self.log(f'买入信号: 价格 {self.dataclose[0]:.2f}')
                # 用 95% 的资金全仓买入（避免因四舍五入导致资金不足）
                self.order = self.order_target_percent(target=0.95)
        else:  # 持仓
            if self.short_ma[0] < self.long_ma[0]:  # 死叉
                self.log(f'卖出信号: 价格 {self.dataclose[0]:.2f}')
                self.order = self.sell(size=self.position.size)  # 卖出全部持仓


# ==========================================
# 5. 封装单股票回测函数
# ==========================================
def run_single_backtest(code, start_date, end_date, init_cash):
    """
    封装单只股票的回测 + 今日信号判断。
    返回: (最终资金, 收益率%, 信号文字)
    """
    # --- 获取数据 ---
    df = get_tushare_data(code, start_date, end_date)
    if df is None:
        return None, None, "❌ 数据获取失败"

    # --- 回测 ---
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df, name=code)
    cerebro.adddata(data)
    cerebro.addstrategy(DoubleMAStrategy)
    cerebro.broker.setcash(init_cash)
    cerebro.broker.addcommissioninfo(AShareCommission())

    cerebro.run()
    final_value = cerebro.broker.getvalue()
    ret_pct = (final_value / init_cash - 1) * 100

    # --- 画图 ---
    os.makedirs("charts", exist_ok=True)
    plot_path = f"charts/{code}_{start_date}_{end_date}.png"
    figs = cerebro.plot(style='candlestick')
    if figs and figs[0]:
        figs[0][0].savefig(plot_path, dpi=150, bbox_inches='tight')

    # --- 今日信号 ---
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()

    if len(df) < 21:
        signal = "⚠️ 数据不足"
    else:
        prev = df.iloc[-2]
        last = df.iloc[-1]
        if prev['ma5'] <= prev['ma20'] and last['ma5'] > last['ma20']:
            signal = "🔔 金叉！建议买入"
        elif prev['ma5'] >= prev['ma20'] and last['ma5'] < last['ma20']:
            signal = "🔔 死叉！建议卖出"
        elif last['ma5'] > last['ma20']:
            signal = "🟢 多头持有"
        else:
            signal = "🔴 空仓观望"

    return final_value, ret_pct, signal


# ==========================================
# 6. 主程序：遍历多只股票
# ==========================================
if __name__ == '__main__':
    # 📋 你的自选股票池（后缀 .SH 或 .SZ）
    STOCK_LIST = [
        '002606.SZ',
        '601319.SH',
        '002286.SZ',
    ]

    START_DATE = '20260101'
    END_DATE   = '20260529'
    INIT_CASH  = 100000.0
    PER_STOCK  = INIT_CASH / len(STOCK_LIST)

    print("=" * 60)
    print(f"📊 多股票双均线回测 ({START_DATE} ~ {END_DATE})")
    print(f"   总资金 {INIT_CASH:,.0f}，每只分配 {PER_STOCK:,.0f}")
    print("=" * 60)

    total_final = 0.0
    for code in STOCK_LIST:
        final_val, ret, signal = run_single_backtest(code, START_DATE, END_DATE, PER_STOCK)
        if final_val is not None:
            total_final += final_val
            print(f"  {code:12s} | 最终 {final_val:10.2f} | 收益 {ret:+6.2f}% | {signal}")
        else:
            print(f"  {code:12s} | {signal}")

    print("-" * 60)
    total_ret = (total_final / INIT_CASH - 1) * 100
    print(f"  组合总计 | 最终 {total_final:10.2f} | 收益 {total_ret:+6.2f}%")