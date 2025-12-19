# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import io
import requests
import json
import base64
import time
from matplotlib import font_manager as fm
from streamlit_oauth import OAuth2Component

# --- 1. 頁面基本設定 ---
st.set_page_config(
    page_title="退休提領回測工具",
    page_icon="💰",
    layout="wide"
)

# --- 2. 工具函式：字型 ---
@st.cache_resource
def install_chinese_font():
    font_path = 'NotoSansCJKtc-Regular.otf'
    font_url = 'https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf'
    if not os.path.exists(font_path):
        try:
            import urllib.request
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(font_url, font_path)
        except:
            return None
    try:
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    except:
        pass
    plt.rcParams['axes.unicode_minus'] = False 

install_chinese_font()
plt.style.use('ggplot')

# --- 3. 工具函式：Firebase 寫入 ---
def save_lead_to_firebase(email):
    """將使用者的 Email 寫入 Firestore"""
    try:
        api_key = st.secrets["FIREBASE_WEB_API_KEY"]
        project_id = st.secrets["FIREBASE_PROJECT_ID"]
        doc_id = base64.b64encode(email.encode()).decode()
        url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents/marketing_leads/{doc_id}?key={api_key}"
        payload = {
            "fields": {
                "email": {"stringValue": email},
                "source": {"stringValue": "google_oauth_login"},
                "last_login": {"timestampValue": datetime.datetime.utcnow().isoformat() + "Z"}
            }
        }
        requests.patch(url, json=payload)
    except Exception as e:
        print(f"Firebase write error: {e}")

# --- 4. 核心邏輯類別 (模擬器) ---
class RetirementSimulator:
    def __init__(self, stock_symbol, bond_symbol, cash_symbol, start_date, end_date):
        # 自動轉大寫並去除空白，解決 0050.tw 找不到的問題
        self.stock_symbol = stock_symbol.upper().strip()
        self.bond_symbol = bond_symbol.upper().strip()
        self.cash_symbol = cash_symbol.upper().strip()
        
        self.request_start_date = pd.to_datetime(start_date)
        self.request_end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
        self.actual_start_date = None
        self.actual_end_date = None
        self.returns = pd.DataFrame()
        self.cpi_annual = None
        self.is_data_valid = False
        self.error_msg = ""

    def download_data(self):
        tickers = [self.stock_symbol, self.bond_symbol, self.cash_symbol]
        # 過濾掉 CASH0，只下載真實存在的標的
        real_tickers = [t for t in tickers if t != 'CASH0']
        
        if real_tickers:
            try:
                # 下載數據
                data = yf.download(real_tickers, start=self.request_start_date, end=self.request_end_date, progress=False, auto_adjust=False)
                
                # 處理資料結構：單一股票 vs 多檔股票
                # 多檔股票會回傳 MultiIndex DataFrame，單檔股票可能回傳一般 DataFrame
                if 'Adj Close' in data:
                    df = data['Adj Close'].copy()
                else:
                    df = data.copy() # Fallback

                # 關鍵修正：如果只下載一檔股票，yfinance 有時回傳的是 Series 或沒有 column name 的 DataFrame
                # 我們強制將其轉換為以 ticker 為 column name 的 DataFrame
                if len(real_tickers) == 1:
                    ticker = real_tickers[0]
                    if isinstance(df, pd.Series):
                        df = df.to_frame(name=ticker)
                    elif isinstance(df, pd.DataFrame):
                        # 如果是單欄 DataFrame 但欄位名不是 ticker (例如 'Adj Close')
                        if ticker not in df.columns:
                            df.columns = [ticker]
            
            except Exception as e:
                self.error_msg = f"下載數據失敗: {e}"
                return

            if df.empty:
                self.error_msg = "無法取得數據，請檢查日期範圍或代碼。"
                return

            # 檢查缺失代碼 (比對時確保都用大寫)
            downloaded_cols = [str(c).upper() for c in df.columns]
            missing = [t for t in real_tickers if t not in downloaded_cols]
            
            # 有時候 yfinance 即使下載失敗也不會報錯，只會少欄位
            if missing:
                # 再次嘗試寬容檢查 (有些代碼可能有後綴差異)
                really_missing = []
                for t in missing:
                    # 如果找不到完全匹配，看看是否有包含關係
                    if not any(t in col for col in downloaded_cols):
                        really_missing.append(t)
                
                if really_missing:
                    self.error_msg = f"找不到以下標的: {really_missing} (請確認 Yahoo Finance 代碼正確，台股請加 .TW)"
                    return
        else:
            try:
                # 如果全是 CASH0，用 SPY 抓時間軸
                temp = yf.download("SPY", start=self.request_start_date, end=self.request_end_date, progress=False)
                df = pd.DataFrame(index=temp.index)
            except:
                self.error_msg = "無法建立時間軸"
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
        self.actual_start_date = self.prices.index[0]
        self.actual_end_date = self.prices.index[-1]

    def download_cpi(self):
        try:
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
            cpi_data = pd.read_csv(url, index_col='DATE', parse_dates=True)
            cpi_data = cpi_data.loc[self.request_start_date - pd.Timedelta(days=365) : self.request_end_date + pd.Timedelta(days=365)]
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

        total = stock_pct + bond_pct + cash_pct
        if not np.isclose(total, 1.0):
            stock_pct /= total
            bond_pct /= total
            cash_pct /= total

        monthly_returns = self.returns
        if monthly_returns.empty: return {}

        start_date = monthly_returns.index[0]
        current_balance = initial_portfolio
        current_annual_withdrawal = initial_portfolio * withdrawal_rate
        
        cumulative_withdrawal = 0.0
        failed = False
        failure_year = None
        history = [initial_portfolio]
        yearly_records = {} 
        withdrawal_month = start_date.month

        for date, row in monthly_returns.iterrows():
            year = date.year
            month = date.month
            
            actual_withdrawal_this_month = 0
            if month == withdrawal_month:
                if date != start_date:
                    if use_fixed_inflation:
                        inflation = fixed_inflation_rate
                    else:
                        inflation = 0.03
                        if self.cpi_annual is not None:
                            try:
                                target_year = year - 1
                                if target_year in self.cpi_annual.index.year:
                                    inflation = self.cpi_annual.loc[self.cpi_annual.index.year == target_year, 'inflation_rate'].values[0]
                            except: pass
                    current_annual_withdrawal *= (1 + inflation)
                
                actual_withdrawal_this_month = current_annual_withdrawal
                current_balance -= actual_withdrawal_this_month
                cumulative_withdrawal += actual_withdrawal_this_month
            
            if current_balance <= 0:
                current_balance = 0
                if not failed:
                    failed = True
                    failure_year = year - start_date.year + 1
            
            if current_balance > 0:
                # 使用 get(key, 0) 避免如果某個代碼下載失敗導致報錯，預設報酬為 0
                ret = (row.get(self.stock_symbol, 0) * stock_pct +
                       row.get(self.bond_symbol, 0) * bond_pct +
                       row.get(self.cash_symbol, 0) * cash_pct)
                current_balance *= (1 + ret)
            
            history.append(current_balance)
            
            yearly_records[year] = {
                '年份': year,
                '期末餘額': current_balance,
                '當年度提領': current_annual_withdrawal, 
                '累計提領': cumulative_withdrawal
            }

        detailed_df = pd.DataFrame(list(yearly_records.values()))
        
        history_np = np.array(history)
        running_max = np.maximum.accumulate(history_np)
        running_max[running_max == 0] = 1
        drawdowns = (running_max - history_np) / running_max
        mdd = drawdowns.max()
        mdd_idx = drawdowns.argmax()
        mdd_year = start_date.year + (mdd_idx // 12)
        
        final_balance_val = history[-1]
        years_duration = len(monthly_returns) / 12
        if years_duration < 1: years_duration = 1
        
        cagr = (final_balance_val / initial_portfolio) ** (1/years_duration) - 1 if final_balance_val > 0 else -1.0

        return {
            'success': not failed,
            'failure_year': failure_year,
            'final_balance': final_balance_val,
            'cagr': cagr,
            'mdd': mdd,
            'mdd_year': mdd_year,
            'history': history,
            'detailed_df': detailed_df,
            'years': years_duration
        }

def to_excel(results_dict, annual_returns_df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        annual_returns_df.to_excel(writer, sheet_name='市場年度報酬')
        for name, res in results_dict.items():
            sheet_name = name[:30]
            summary_data = {
                '項目': ['成功與否', '破產年份', '期末資產', 'CAGR', 'MDD', 'MDD發生年'],
                '數值': [
                    "成功" if res['success'] else "失敗",
                    res['failure_year'] if not res['success'] else "-",
                    res['final_balance'],
                    res['cagr'],
                    res['mdd'],
                    res['mdd_year']
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name=sheet_name, startrow=0, index=False)
            res['detailed_df'].to_excel(writer, sheet_name=sheet_name, startrow=8, index=False)
    return output.getvalue()

# --- 5. 主程式介面邏輯 (登入牆) ---

if "user_email" not in st.session_state:
    st.session_state["user_email"] = None

# --- 畫面 A: 尚未登入 ---
if not st.session_state["user_email"]:
    st.title("🔒 退休提領回測工具")
    st.markdown("### 請登入以使用完整功能")
    st.markdown("本工具提供強大的歷史回測功能，協助您規劃退休金流。請使用 Google 帳號登入以開始使用。")
    
    try:
        oauth2 = OAuth2Component(
            st.secrets["GOOGLE_CLIENT_ID"], 
            st.secrets["GOOGLE_CLIENT_SECRET"],
            "https://accounts.google.com/o/oauth2/v2/auth",
            "https://oauth2.googleapis.com/token"
        )
        
        result = oauth2.authorize_button(
            name="使用 Google 帳號登入",
            icon="https://www.google.com.tw/favicon.ico",
            redirect_uri=st.secrets["GOOGLE_REDIRECT_URI"],
            scope="openid email profile",
            key="google_auth_btn"
        )
        
        if result:
            id_token = result["token"]["id_token"]
            payload = id_token.split('.')[1]
            payload += '=' * (-len(payload) % 4)
            decoded = json.loads(base64.b64decode(payload).decode('utf-8'))
            email = decoded.get("email")
            
            if email:
                st.session_state["user_email"] = email
                save_lead_to_firebase(email)
                st.success(f"登入成功！歡迎 {email}")
                time.sleep(1)
                st.rerun()
                
    except Exception as e:
        st.error(f"登入設定錯誤: {e}")
        st.info("請檢查 Secrets 設定是否正確")

# --- 畫面 B: 已登入 (顯示計算機) ---
else:
    with st.sidebar:
        st.write(f"👤 **{st.session_state['user_email']}**")
        if st.button("登出"):
            st.session_state["user_email"] = None
            st.rerun()
        st.divider()

    st.sidebar.header("⚙️ 參數設定")
    with st.sidebar.expander("1. 資金與期間", expanded=True):
        start_capital = st.number_input("初始本金", value=10000000, step=100000)
        withdrawal_rate = st.slider("初始提領率 (%)", 1.0, 10.0, 4.0, 0.1) / 100.0
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

    st.title("📈 退休提領回測工具 (Web版)")
    st.markdown("基於 Bengen 4% 法則與 Trinity Study 邏輯的互動式模擬器。")

    @st.cache_data(ttl=3600)
    def load_market_data(s, b, c, start, end):
        sim = RetirementSimulator(s, b, c, start, end)
        sim.download_data()
        sim.download_cpi()
        return sim

    if st.button("開始回測", type="primary"):
        with st.spinner("正在下載歷史數據並計算中..."):
            sim = load_market_data(sym_stock, sym_bond, sym_cash, str(start_d), str(end_d))
            
            if not sim.is_data_valid:
                st.error(sim.error_msg)
            else:
                annual_df = sim.get_annual_returns_df()
                total_years = len(annual_df)
                actual_start_str = sim.actual_start_date.strftime('%Y-%m-%d')
                actual_end_str = sim.actual_end_date.strftime('%Y-%m-%d')
                
                st.success(f"數據下載成功！實際數據期間: {actual_start_str} 至 {actual_end_str} (共 {total_years} 年)")
                if sim.request_start_date < sim.actual_start_date:
                    st.info(f"💡 提示：您請求的開始日期 ({start_d}) 早於數據上市日期，已自動調整為實際最早可用日期。")

                tab1, tab2, tab3, tab4 = st.tabs(["📊 資產走勢圖", "📋 詳細統計數據", "📅 市場年度報酬", "📄 詳細收支表"])
                
                results = {}
                configs = [("組合 1", p1), ("組合 2", p2), ("組合 3", p3)]
                
                for name, (s, b, c) in configs:
                    parts = []
                    if s>0: parts.append(f"股{s:.0%}")
                    if b>0: parts.append(f"債{b:.0%}")
                    if c>0: parts.append(f"現{c:.0%}")
                    full_name = " + ".join(parts)
                    res = sim.run_simulation(start_capital, withdrawal_rate, s, b, c, use_fixed_infl, fixed_infl_rate)
                    if res:
                        results[full_name] = res

                with tab1:
                    if results:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ['#3498db', '#2ecc71', '#e74c3c']
                        for i, (name, res) in enumerate(results.items()):
                            history = np.array(res['history'])
                            months_axis = np.arange(len(history)) / 12
                            color = colors[i % len(colors)]
                            ax.plot(months_axis, history/1000000, label=name, linewidth=2.5, color=color)
                            ax.scatter(months_axis[-1], history[-1]/1000000, s=50, color=color)
                        ax.set_title(f"資產淨值走勢 ({total_years}年期間)", fontsize=14)
                        ax.set_xlabel("經過年數")
                        ax.set_ylabel("資產餘額 (百萬)")
                        ax.legend()
                        ax.grid(True, linestyle='--', alpha=0.7)
                        st.pyplot(fig)
                    else:
                        st.warning("無有效模擬結果")

                with tab2:
                    for name, res in results.items():
                        with st.container():
                            st.subheader(name)
                            c1, c2, c3, c4 = st.columns(4)
                            is_success = res['success']
                            final_bal = res['final_balance']
                            cagr = res['cagr']
                            mdd = res['mdd']
                            c1.metric("模擬結果", "成功" if is_success else f"第 {res['failure_year']} 年破產", delta_color="normal" if is_success else "inverse")
                            c2.metric("期末資產", f"${final_bal:,.0f}")
                            c3.metric("CAGR (年化)", f"{cagr:.2%}")
                            c4.metric("最大回撤 (MDD)", f"{mdd:.1%}", help=f"發生於第 {res['mdd_year']} 年左右")
                            st.divider()

                with tab3:
                    st.markdown("### 各資產年度報酬率")
                    fmt_df = annual_df.style.format("{:.2%}")
                    def color_negative_red(val):
                        color = 'red' if val < 0 else 'green'
                        return f'color: {color}'
                    fmt_df = fmt_df.map(color_negative_red)
                    st.dataframe(fmt_df, use_container_width=True)

                with tab4:
                    if st.session_state["user_email"]:
                        st.markdown("### 年度詳細收支表")
                        for name, res in results.items():
                            with st.expander(f"{name} - 詳細數據", expanded=False):
                                df_detail = res['detailed_df']
                                df_show = df_detail.set_index('年份')
                                st.dataframe(df_show.style.format({'期末餘額': '${:,.0f}', '當年度提領': '${:,.0f}', '累計提領': '${:,.0f}'}), use_container_width=True)
                    else:
                        st.warning("🔒 此功能僅限會員使用，請在左側 Google 登入。")
                        st.info("登入後即可解鎖「詳細收支表」與「Excel 報告下載」功能！")

                st.divider()
                if results:
                    if st.session_state["user_email"]:
                        excel_data = to_excel(results, annual_df)
                        st.download_button(
                            label="📥 下載完整 Excel 報告",
                            data=excel_data,
                            file_name='retirement_simulation_report.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                    else:
                        st.button("🔒 登入後下載完整 Excel 報告", disabled=True)
    else:
        st.info("👈 請在左側調整參數，並點擊「開始回測」按鈕")
