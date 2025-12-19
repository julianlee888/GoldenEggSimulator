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
    page_title="金蛋模擬器",
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
        
        # 使用 REST API 寫入 (不需要複雜驗證，因為我們已經在 Rules 開放寫入)
        # 使用 email 作為文件 ID，避免重複寫入
        doc_id = base64.b64encode(email.encode()).decode() # 簡單編碼當ID
        
        url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents/marketing_leads/{doc_id}?key={api_key}"
        
        payload = {
            "fields": {
                "email": {"stringValue": email},
                "source": {"stringValue": "google_oauth_login"},
                "last_login": {"timestampValue": datetime.datetime.utcnow().isoformat() + "Z"}
            }
        }
        
        # 使用 PATCH (如果存在就更新時間，不存在就建立)
        requests.patch(url, json=payload)
    except Exception as e:
        # 寫入失敗不影響使用者使用，默默紀錄就好
        print(f"Firebase write error: {e}")

# --- 4. 核心邏輯類別 (模擬器) ---
class RetirementSimulator:
    def __init__(self, stock_symbol, bond_symbol, cash_symbol, start_date, end_date):
        self.stock_symbol = stock_symbol
        self.bond_symbol = bond_symbol
        self.cash_symbol = cash_symbol
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
        real_tickers = [t for t in tickers if t != 'CASH0']
        
        if real_tickers:
            try:
                data = yf.download(real_tickers, start=self.request_start_date, end=self.request_end_date, progress=False, auto_adjust=False)
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

            downloaded_cols = df.columns.tolist() if isinstance(df.columns, pd.Index) else []
            missing = [t for t in real_tickers if t not in downloaded_cols]
            if missing:
                self.error_msg = f"找不到以下標的: {missing}"
                return
        else:
            try:
                temp = yf.download("SPY", start=self.request_start_date, end=self.request_end_date, progress=False)
                df = pd.DataFrame(index=temp.index)
            except:
                self.error_msg = "無法建立時間軸"
                return

        if 'CASH0' in tickers:
            df['CASH0'] = 100.0

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

        annual_returns = self.returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
        years_retired = len(annual_returns)
        if years_retired < 1: return {}

        start_year = annual_returns.index[0].year
        current_balance = initial_portfolio
        current_withdrawal = initial_portfolio * withdrawal_rate
        
        records = []
        cumulative_withdrawal = 0.0
        failed = False
        failure_year = None
        history = [initial_portfolio]

        for date, row in annual_returns.iterrows():
            year = date.year
            this_year_withdrawal = current_withdrawal
            current_balance -= current_withdrawal
            
            if current_balance <= 0:
                current_balance = 0
                failed = True
                failure_year = year - start_year + 1
                cumulative_withdrawal += this_year_withdrawal
                records.append({'年份': year, '期末餘額': 0, '當年度提領': this_year_withdrawal, '累計提領': cumulative_withdrawal})
                history.append(0)
                break
            
            cumulative_withdrawal += this_year_withdrawal
            ret = (row.get(self.stock_symbol, 0) * stock_pct +
                   row.get(self.bond_symbol, 0) * bond_pct +
                   row.get(self.cash_symbol, 0) * cash_pct)
            current_balance *= (1 + ret)
            history.append(current_balance)
            
            records.append({'年份': year, '期末餘額': current_balance, '當年度提領': this_year_withdrawal, '累計提領': cumulative_withdrawal})

            if use_fixed_inflation:
                inflation = fixed_inflation_rate
            else:
                inflation = 0.03
                if self.cpi_annual is not None:
                    try:
                        if year in self.cpi_annual.index.year:
                            val = self.cpi_annual.loc[self.cpi_annual.index.year == year, 'inflation_rate'].values[0]
                            inflation = val
                    except: pass
            current_withdrawal *= (1 + inflation)

        last_recorded_year = records[-1]['年份'] if records else start_year - 1
        while len(history) < years_retired + 1:
            history.append(0)
            last_recorded_year += 1
            records.append({'年份': last_recorded_year, '期末餘額': 0, '當年度提領': 0, '累計提領': cumulative_withdrawal})

        detailed_df = pd.DataFrame(records)
        history_np = np.array(history)
        running_max = np.maximum.accumulate(history_np)
        running_max[running_max == 0] = 1
        drawdowns = (running_max - history_np) / running_max
        mdd = drawdowns.max()
        mdd_idx = drawdowns.argmax()
        mdd_year = start_year + mdd_idx - 1 
        if mdd_year < start_year: mdd_year = start_year 
        
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
    st.title("🔒退休提領回測工具")
    st.markdown("請登入以使用完整功能")
    st.markdown("本工具提供強大的歷史回測功能，協助您規劃退休金流。請使用Google帳號登入以開始使用")
    
    try:
        # 設定 OAuth 元件 (修正：初始化時不放入 scope)
        oauth2 = OAuth2Component(
            st.secrets["GOOGLE_CLIENT_ID"], 
            st.secrets["GOOGLE_CLIENT_SECRET"],
            "https://accounts.google.com/o/oauth2/v2/auth",
            "https://oauth2.googleapis.com/token"
        )
        
        # 顯示登入按鈕 (修正：scope 參數移至此處)
        result = oauth2.authorize_button(
            name="使用 Google 帳號登入",
            icon="https://www.google.com.tw/favicon.ico",
            redirect_uri=st.secrets["GOOGLE_REDIRECT_URI"],
            scope="openid email profile",
            key="google_auth_btn"
        )
        
        if result:
            # 解析 Email
            id_token = result["token"]["id_token"]
            payload = id_token.split('.')[1]
            payload += '=' * (-len(payload) % 4)
            decoded = json.loads(base64.b64decode(payload).decode('utf-8'))
            email = decoded.get("email")
            
            if email:
                st.session_state["user_email"] = email
                # 寫入資料庫
                save_lead_to_firebase(email)
                st.success(f"登入成功！歡迎 {email}")
                time.sleep(1)
                st.rerun()
                
    except Exception as e:
        st.error(f"登入設定錯誤: {e}")
        st.info("請檢查 Secrets 設定是否正確")

# --- 畫面 B: 已登入 (顯示計算機) ---
else:
    # 側邊欄：使用者資訊
    with st.sidebar:
        st.write(f"👤 **{st.session_state['user_email']}**")
        if st.button("登出"):
            st.session_state["user_email"] = None
            st.rerun()
        st.divider()

    # 側邊欄：參數設定
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
        st.caption("輸入YAHOO FINANCE股票代碼，'CASH0'可模擬零息現金")
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

    st.title("📈金蛋模擬器")
    st.markdown("以Bengen 4%法則與Trinity Study為基礎的退休金流模擬器，僅供教育用途")

    # 載入數據函式 (放在這裡確保只在登入後執行)
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
                            c4.metric("最大回撤 (MDD)", f"{mdd:.1%}", help=f"發生於第 {res['mdd_year']} 年")
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
                    st.markdown("### 年度詳細收支表")
                    for name, res in results.items():
                        with st.expander(f"{name} - 詳細數據", expanded=False):
                            df_detail = res['detailed_df']
                            df_show = df_detail.set_index('年份')
                            st.dataframe(df_show.style.format({'期末餘額': '${:,.0f}', '當年度提領': '${:,.0f}', '累計提領': '${:,.0f}'}), use_container_width=True)

                # 下載按鈕 (現在人人可見，因為已經登入才能進來)
                st.divider()
                if results:
                    excel_data = to_excel(results, annual_df)
                    st.download_button(
                        label="📥 下載完整 Excel 報告",
                        data=excel_data,
                        file_name='retirement_simulation_report.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
    else:
        st.info("👈 請在左側調整參數，並點擊「開始回測」按鈕")
