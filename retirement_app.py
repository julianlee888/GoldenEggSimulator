# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime
import os
from matplotlib import font_manager as fm

# --- 1. 頁面基本設定 ---
st.set_page_config(
    page_title="退休提領回測工具",
    page_icon="💰",
    layout="wide"
)

# --- 2. 字型設定功能 (針對 Streamlit Cloud 優化) ---
@st.cache_resource
def install_chinese_font():
    """
    下載並設定中文字型 (快取資源，避免每次重跑)
    """
    font_path = 'NotoSansCJKtc-Regular.otf'
    font_url = 'https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf'
    
    if not os.path.exists(font_path):
        try:
            import urllib.request
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(font_url, font_path)
        except Exception as e:
            st.warning(f"字型下載失敗: {e} (將使用預設字型)")
            return None

    try:
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    except Exception as e:
        st.warning(f"字型設定失敗: {e}")
    
    plt.rcParams['axes.unicode_minus'] = False 

install_chinese_font()
plt.style.use('ggplot')

# --- 3. 核心邏輯類別 (RetirementSimulator) ---
class RetirementSimulator:
    def __init__(self, stock_symbol, bond_symbol, cash_symbol, start_date, end_date):
        self.stock_symbol = stock_symbol
        self.bond_symbol = bond_symbol
        self.cash_symbol = cash_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.returns = pd.DataFrame()
        self.cpi_annual = None
        self.is_data_valid = False
        self.error_msg = ""

    def download_data(self):
        tickers = [self.stock_symbol, self.bond_symbol, self.cash_symbol]
        real_tickers = [t for t in tickers if t != 'CASH0']
        
        # 下載市場數據
        if real_tickers:
            try:
                data = yf.download(real_tickers, start=self.start_date, end=self.end_date, progress=False, auto_adjust=False)
                if 'Adj Close' in data:
                    df = data['Adj Close'].copy()
                else:
                    df = data.copy()
            except Exception as e:
                self.error_msg = f"下載數據失敗: {e}"
                return

            if df.empty:
                self.error_msg = "無法取得數據，請檢查日期範圍或代碼。"
                return

            # 檢查缺失代碼
            downloaded_cols = df.columns.tolist() if isinstance(df.columns, pd.Index) else []
            missing = [t for t in real_tickers if t not in downloaded_cols]
            if missing:
                self.error_msg = f"找不到以下標的: {missing}"
                return
        else:
            # 如果全是 CASH0，建立一個空的 DataFrame 結構
            try:
                temp = yf.download("SPY", start=self.start_date, end=self.end_date, progress=False)
                df = pd.DataFrame(index=temp.index)
            except:
                self.error_msg = "無法建立時間軸 (請至少包含一個真實市場標的或確保網路連線)"
                return

        # 處理 CASH0
        if 'CASH0' in tickers:
            df['CASH0'] = 100.0

        # 轉換月報酬
        df_monthly = df.resample('ME').last()
        self.returns = df_monthly.pct_change().dropna()
        self.prices = df_monthly.dropna()
        
        if self.prices.empty:
            self.error_msg = "數據處理後為空 (可能期間太短)。"
            return
            
        self.is_data_valid = True

    def download_cpi(self):
        try:
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date) if self.end_date else datetime.datetime.now()
            cpi_data = web.DataReader('CPIAUCSL', 'fred', start, end)
            self.cpi_annual = cpi_data.resample('YE').last().pct_change()
            self.cpi_annual.columns = ['inflation_rate']
            mean_inflation = self.cpi_annual['inflation_rate'].mean()
            self.cpi_annual['inflation_rate'] = self.cpi_annual['inflation_rate'].fillna(mean_inflation)
        except:
            self.cpi_annual = None

    def get_annual_returns_df(self):
        if self.returns.empty: return pd.DataFrame()
        return self.returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)

    def run_simulation(self, initial_portfolio, withdrawal_rate, stock_pct, bond_pct, cash_pct, use_fixed_inflation, fixed_inflation_rate):
        if not self.is_data_valid: return {}

        # 正規化比例
        total = stock_pct + bond_pct + cash_pct
        if not np.isclose(total, 1.0):
            stock_pct /= total
            bond_pct /= total
            cash_pct /= total

        annual_returns = self.returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
        years_retired = len(annual_returns)
        if years_retired < 1: return {}

        # 初始化變數
        start_year = annual_returns.index[0].year
        current_balance = initial_portfolio
        current_withdrawal = initial_portfolio * withdrawal_rate
        history = [current_balance] # 包含期初
        
        # 新增：詳細收支記錄
        records = []
        cumulative_withdrawal = 0.0

        failed = False
        failure_year = None

        # 為了讓 history 長度對應正確，我們記錄「該年度結束時」的餘額
        # 第一筆 history 是初始本金，不算在 loop 裡
        # run_simulation 的 history 列表邏輯：index 0 是初始，index 1 是第 1 年結束...
        
        # 重置 history，我們只存期末餘額以便畫圖 (或者保留期初)
        # 這裡為了畫圖方便，保留原邏輯：history[0] = 期初, history[i] = 第 i 年期末
        history = [initial_portfolio]

        for date, row in annual_returns.iterrows():
            year = date.year
            
            # 記錄當年度計畫提領金額
            this_year_withdrawal = current_withdrawal
            
            # 1. 提領 (年初提領)
            current_balance -= current_withdrawal
            
            if current_balance <= 0:
                current_balance = 0
                failed = True
                failure_year = year - start_year + 1
                
                # 破產該年，實際能領的只有剩下的錢 (雖然邏輯上是失敗，但記錄上就記原本想領的或實際領的)
                # 這裡記錄「計畫提領」比較能看出原本想領多少
                cumulative_withdrawal += this_year_withdrawal
                
                records.append({
                    '年份': year,
                    '期末餘額': 0,
                    '當年度提領': this_year_withdrawal,
                    '累計提領': cumulative_withdrawal
                })
                history.append(0)
                break
            
            # 成功提領
            cumulative_withdrawal += this_year_withdrawal
            
            # 2. 投資
            ret = (row.get(self.stock_symbol, 0) * stock_pct +
                   row.get(self.bond_symbol, 0) * bond_pct +
                   row.get(self.cash_symbol, 0) * cash_pct)
            current_balance *= (1 + ret)
            history.append(current_balance)
            
            # 記錄
            records.append({
                '年份': year,
                '期末餘額': current_balance,
                '當年度提領': this_year_withdrawal,
                '累計提領': cumulative_withdrawal
            })

            # 3. 通膨調整 (為下一年準備)
            if use_fixed_inflation:
                inflation = fixed_inflation_rate
            else:
                inflation = 0.03
                if self.cpi_annual is not None:
                    try:
                        val = self.cpi_annual.loc[self.cpi_annual.index.year == year, 'inflation_rate'].values[0]
                        inflation = val
                    except: pass
            current_withdrawal *= (1 + inflation)

        # 補齊剩餘年份的 0 (若提早破產)
        # 需要補齊 history 和 records
        last_recorded_year = records[-1]['年份'] if records else start_year - 1
        
        while len(history) < years_retired + 1:
            history.append(0)
            last_recorded_year += 1
            # 破產後提領為 0
            records.append({
                '年份': last_recorded_year,
                '期末餘額': 0,
                '當年度提領': 0,
                '累計提領': cumulative_withdrawal # 累計不再增加
            })

        # 建立詳細 DataFrame
        detailed_df = pd.DataFrame(records)
        # 設定年份為索引，雖然介面上可能直接顯示 Column 比較好看，這裡保留年份為欄位
        # detailed_df.set_index('年份', inplace=True)

        # 計算指標
        history_np = np.array(history)
        running_max = np.maximum.accumulate(history_np)
        running_max[running_max == 0] = 1
        drawdowns = (running_max - history_np) / running_max
        mdd = drawdowns.max()
        mdd_idx = drawdowns.argmax()
        mdd_year = start_year + mdd_idx - 1 # history index 0 is start, index 1 is year 1 end
        if mdd_year < start_year: mdd_year = start_year # fallback
        
        # cagr 計算 (使用最後一年非 0 餘額比較合理，或者直接用終值)
        # 若破產，終值為 0，cagr 為 -1
        final_balance_val = history[-1]
        cagr = (final_balance_val / initial_portfolio) ** (1/years_retired) - 1 if final_balance_val > 0 else -1.0

        return {
            'success': not failed,
            'failure_year': failure_year,
            'final_balance': final_balance_val,
            'cagr': cagr,
            'mdd': mdd,
            'mdd_year': mdd_year,
            'history': history,
            'detailed_df': detailed_df,
            'years': years_retired
        }

# --- 4. Streamlit 介面邏輯 ---

# 側邊欄：參數設定
st.sidebar.header("⚙️ 參數設定")

with st.sidebar.expander("1. 資金與期間", expanded=True):
    start_capital = st.number_input("初始本金", value=10000000, step=100000)
    withdrawal_rate = st.slider("初始提領率 (%)", 1.0, 10.0, 4.0, 0.1) / 100.0
    
    # 日期選擇
    col_d1, col_d2 = st.columns(2)
    start_d = col_d1.date_input("開始日期", datetime.date(1986, 1, 1))
    end_d = col_d2.date_input("結束日期", datetime.date.today())

with st.sidebar.expander("2. 通膨設定", expanded=False):
    use_fixed_infl = st.toggle("使用固定通膨率", value=True)
    fixed_infl_rate = st.slider("固定通膨率 (%)", 0.0, 10.0, 3.0, 0.5) / 100.0
    if not use_fixed_infl:
        st.caption("將使用 FRED (CPIAUCSL) 歷史數據")

with st.sidebar.expander("3. 投資標的代碼", expanded=False):
    st.caption("輸入 'CASH0' 可模擬零息現金")
    sym_stock = st.text_input("股票代碼", "VFINX")
    sym_bond = st.text_input("債券代碼", "VUSTX")
    sym_cash = st.text_input("現金代碼", "VFISX")

st.sidebar.subheader("投資組合比例設定")
# Helper for portfolio inputs
def portfolio_input(idx, def_s, def_b, def_c):
    st.sidebar.markdown(f"**組合 {idx}**")
    c1, c2, c3 = st.sidebar.columns(3)
    s = c1.number_input(f"股%", value=def_s, key=f"s{idx}", step=5)
    b = c2.number_input(f"債%", value=def_b, key=f"b{idx}", step=5)
    c = c3.number_input(f"現%", value=def_c, key=f"c{idx}", step=5)
    total = s + b + c
    if total != 100:
        st.sidebar.warning(f"總和: {total}% (將自動正規化)")
    return s/100, b/100, c/100

p1 = portfolio_input(1, 100, 0, 0)
p2 = portfolio_input(2, 50, 50, 0)
p3 = portfolio_input(3, 50, 0, 50)

# 主畫面
st.title("📈 退休提領回測工具 (Web版)")
st.markdown("基於 Bengen 4% 法則與 Trinity Study 邏輯的互動式模擬器。")

# --- 5. 執行模擬 ---

# 使用快取載入數據，避免每次調整參數都重新下載
@st.cache_data(ttl=3600) # 快取 1 小時
def load_market_data(s, b, c, start, end):
    sim = RetirementSimulator(s, b, c, start, end)
    sim.download_data()
    sim.download_cpi()
    return sim

if st.button("開始回測", type="primary"):
    with st.spinner("正在下載歷史數據並計算中..."):
        sim = load_market_data(sym_stock, sym_bond, sym_cash, start_d, end_d)
        
        if not sim.is_data_valid:
            st.error(sim.error_msg)
        else:
            # 顯示基本資訊
            annual_df = sim.get_annual_returns_df()
            total_years = len(annual_df)
            
            st.success(f"數據下載成功！期間: {sim.start_date} 至 {sim.end_date} (共 {total_years} 年)")
            
            # Tab 分頁
            tab1, tab2, tab3, tab4 = st.tabs(["📊 資產走勢圖", "📋 詳細統計數據", "📅 市場年度報酬", "📄 詳細收支表"])
            
            results = {}
            configs = [("組合 1", p1), ("組合 2", p2), ("組合 3", p3)]
            
            # 執行計算
            for name, (s, b, c) in configs:
                # 產生動態名稱
                parts = []
                if s>0: parts.append(f"股{s:.0%}")
                if b>0: parts.append(f"債{b:.0%}")
                if c>0: parts.append(f"現{c:.0%}")
                full_name = " + ".join(parts)
                
                res = sim.run_simulation(start_capital, withdrawal_rate, s, b, c, use_fixed_infl, fixed_infl_rate)
                if res:
                    results[full_name] = res

            # --- Tab 1: 圖表 ---
            with tab1:
                if results:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['#3498db', '#2ecc71', '#e74c3c']
                    
                    for i, (name, res) in enumerate(results.items()):
                        history = np.array(res['history'])
                        years = range(len(history))
                        color = colors[i % len(colors)]
                        ax.plot(years, history/1000000, label=name, linewidth=2.5, color=color)
                        ax.scatter(years[-1], history[-1]/1000000, s=50, color=color)
                    
                    ax.set_title(f"資產淨值走勢 ({total_years}年期間)", fontsize=14)
                    ax.set_xlabel("經過年數")
                    ax.set_ylabel("資產餘額 (百萬)")
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                else:
                    st.warning("無有效模擬結果")

            # --- Tab 2: 統計卡片 ---
            with tab2:
                for name, res in results.items():
                    with st.container():
                        st.subheader(name)
                        c1, c2, c3, c4 = st.columns(4)
                        
                        is_success = res['success']
                        final_bal = res['final_balance']
                        cagr = res['cagr']
                        mdd = res['mdd']
                        
                        c1.metric("模擬結果", "成功" if is_success else f"第 {res['failure_year']} 年破產", 
                                  delta_color="normal" if is_success else "inverse")
                        c2.metric("期末資產", f"${final_bal:,.0f}")
                        c3.metric("CAGR (年化)", f"{cagr:.2%}")
                        c4.metric("最大回撤 (MDD)", f"{mdd:.1%}", help=f"發生於第 {res['mdd_year']} 年")
                        st.divider()

            # --- Tab 3: 原始數據表 ---
            with tab3:
                st.markdown("### 各資產年度報酬率")
                # 格式化表格
                fmt_df = annual_df.style.format("{:.2%}")
                def color_negative_red(val):
                    color = 'red' if val < 0 else 'green'
                    return f'color: {color}'
                fmt_df = fmt_df.map(color_negative_red)
                st.dataframe(fmt_df, use_container_width=True)

            # --- Tab 4: 詳細收支表 (新增) ---
            with tab4:
                st.markdown("### 年度詳細收支表")
                for name, res in results.items():
                    with st.expander(f"{name} - 詳細數據", expanded=False):
                        df_detail = res['detailed_df']
                        # 格式化 DataFrame 顯示
                        # 設定年份為索引以便顯示
                        df_show = df_detail.set_index('年份')
                        st.dataframe(
                            df_show.style.format({
                                '期末餘額': '${:,.0f}', 
                                '當年度提領': '${:,.0f}', 
                                '累計提領': '${:,.0f}'
                            }),
                            use_container_width=True
                        )

else:
    st.info("👈 請在左側調整參數，並點擊「開始回測」按鈕")
