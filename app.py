import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. OLDAL BEÁLLÍTÁSAI ÉS STÍLUS
# ==========================================
st.set_page_config(
    page_title="ICT Backtester | Szakdolgozat", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. ÜZLETI LOGIKA (OSZTÁLYOK)
# ==========================================
class ICTStrategy:
    def __init__(self, data):
        self.data = data.copy()
        
    def generate_signals(self):
        self.data['Signal'] = 0
        long_condition = ((self.data['Sweep_Bullish'] == True) | (self.data['Sweep_Bullish'].shift(1) == True)) & (self.data['Break_Up'] == True) & (self.data['Close'] > self.data['VWAP'])
        short_condition = ((self.data['Sweep_Bearish'] == True) | (self.data['Sweep_Bearish'].shift(1) == True)) & (self.data['Break_Down'] == True) & (self.data['Close'] < self.data['VWAP'])
        
        self.data.loc[long_condition, 'Signal'] = 1
        self.data.loc[short_condition, 'Signal'] = -1
        return self.data

class Portfolio:
    def __init__(self, data, initial_capital, contract_multiplier, commission):
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.multiplier = contract_multiplier 
        self.commission = commission 
        
    def simulate_trades(self):
        self.data['Position'] = self.data['Signal'].shift(1).fillna(0)
        self.data['Price_Change'] = self.data['Close'] - self.data['Close'].shift(1)
        self.data['PnL'] = self.data['Position'] * self.data['Price_Change'] * self.multiplier
        self.data['Trade_Executed'] = self.data['Position'].diff().abs() > 0
        self.data.loc[self.data['Trade_Executed'], 'PnL'] -= self.commission
        self.data['PnL'] = self.data['PnL'].fillna(0)
        self.data['Equity'] = self.initial_capital + self.data['PnL'].cumsum()
        return self.data

# ==========================================
# 3. ADATLETÖLTÉS (Gyorsítótárazva - Caching)
# ==========================================
@st.cache_data(show_spinner=False)
def load_and_prepare_data(ticker):
    df = yf.download(ticker, interval="1h", period="1y")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.dropna(inplace=True)
    
    # Indikátorok számolása
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    daily_groups = df.groupby(df.index.date)
    df['Cum_Vol'] = daily_groups['Volume'].cumsum()
    df['Cum_Vol_Price'] = (df['Typical_Price'] * df['Volume']).groupby(df.index.date).cumsum()
    df['VWAP'] = df['Cum_Vol_Price'] / df['Cum_Vol']

    asian_session = df.between_time('00:00', '08:00')
    daily_asian_high = asian_session.groupby(asian_session.index.date)['High'].max()
    daily_asian_low = asian_session.groupby(asian_session.index.date)['Low'].min()
    
    df['Asian_High'] = df.index.date
    df['Asian_High'] = df['Asian_High'].map(daily_asian_high)
    df['Asian_Low'] = df.index.date
    df['Asian_Low'] = df['Asian_Low'].map(daily_asian_low)

    window = 5
    df['Local_Max'] = df['High'].rolling(window=window).max().shift(1)
    df['Local_Min'] = df['Low'].rolling(window=window).min().shift(1)
    df['Break_Up'] = (df['Close'] > df['Local_Max']) & (df['Close'].shift(1) <= df['Local_Max'].shift(1))
    df['Break_Down'] = (df['Close'] < df['Local_Min']) & (df['Close'].shift(1) >= df['Local_Min'].shift(1))

    df['Sweep_Bullish'] = (df['Low'] < df['Asian_Low']) & (df['Close'] > df['Asian_Low']) & (df['Open'] > df['Asian_Low'])
    df['Sweep_Bearish'] = (df['High'] > df['Asian_High']) & (df['Close'] < df['Asian_High']) & (df['Open'] < df['Asian_High'])
    return df

# ==========================================
# 4. WEBOLDAL FELÜLET (UI)
# ==========================================
st.title("🏛️ Algoritmikus Kereskedési és Backtesting Keretrendszer")
st.markdown("Szakdolgozati demonstrációs szoftver | **ICT (Inner Circle Trader) koncepciók validálása határidős piacokon**")
st.divider()

# Oldalsáv (Sidebar)
st.sidebar.header("⚙️ Szimulációs Paraméterek")
ticker = st.sidebar.text_input("Instrumentum (Ticker)", value="MNQ=F")
kezdoto = st.sidebar.number_input("Kezdőtőke (USD)", min_value=1000.0, value=10000.0, step=1000.0)
multiplier = st.sidebar.number_input("Kontraktus szorzó (MNQ=2)", value=2.0)
commission = st.sidebar.number_input("Jutalék / Csúszás (USD)", value=1.0, step=0.5)

if st.sidebar.button("🚀 Szimuláció Futtatása", type="primary"):
    with st.spinner('Adatelemzés és stratégia validálása folyamatban...'):
        
        # Adatok feldolgozása
        df = load_and_prepare_data(ticker)
        
        # Stratégia és Portfólió futtatása
        strategy = ICTStrategy(df)
        df_signals = strategy.generate_signals()
        
        portfolio = Portfolio(df_signals, initial_capital=kezdoto, contract_multiplier=multiplier, commission=commission)
        df_portfolio = portfolio.simulate_trades()
        
        # Statisztikák számolása
        df_portfolio['Trade_ID'] = df_portfolio['Trade_Executed'].cumsum()
        active_trades = df_portfolio[df_portfolio['Position'] != 0]
        trade_results = active_trades.groupby('Trade_ID')['PnL'].sum()
        
        total_trades = len(trade_results)
        winning_trades = (trade_results > 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        final_equity = df_portfolio['Equity'].iloc[-1]
        net_profit = final_equity - kezdoto
        roi = (net_profit / kezdoto) * 100
        
        rolling_max = df_portfolio['Equity'].cummax()
        drawdown = (df_portfolio['Equity'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        daily_equity = df_portfolio['Equity'].resample('D').last().dropna()
        daily_returns = daily_equity.pct_change().dropna()
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 1 else 0

        # --- EREDMÉNYEK MEGJELENÍTÉSE (KPI Kártyák) ---
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Záró Egyenleg", f"${final_equity:,.2f}", f"{roi:.2f}% Hozam")
        col2.metric("Nettó Profit", f"${net_profit:,.2f}")
        col3.metric("Nyerési Arány (Win Rate)", f"{win_rate:.2f}%")
        col4.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        col5.metric("Sharpe-ráta", f"{sharpe_ratio:.2f}")
        
        st.divider()

        # --- FÜLES ELRENDEZÉS (TABS) ---
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Grafikonok és Árfolyam", "📉 Visszaesés (Drawdown)", "📋 Kötésnapló", "📖 Módszertan"])
        
        with tab1:
            st.subheader("Interaktív Árfolyam és Egyenleggörbe")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            fig.add_trace(go.Candlestick(x=df_portfolio.index, open=df_portfolio['Open'], high=df_portfolio['High'], low=df_portfolio['Low'], close=df_portfolio['Close'], name='Árfolyam'), row=1, col=1)
            
            buy_signals = df_portfolio[df_portfolio['Signal'] == 1]
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low'] - 20, mode='markers', marker=dict(symbol='triangle-up', color='lightgreen', size=12, line=dict(color='black', width=1)), name='Vétel (Long)'), row=1, col=1)
            
            sell_signals = df_portfolio[df_portfolio['Signal'] == -1]
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High'] + 20, mode='markers', marker=dict(symbol='triangle-down', color='red', size=12, line=dict(color='black', width=1)), name='Eladás (Short)'), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df_portfolio.index, y=df_portfolio['Equity'], mode='lines', line=dict(color='#00BFFF', width=2), fill='tozeroy', name='Tőke (Equity)'), row=2, col=1)
            
            fig.update_layout(height=700, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Tőkevisszaesés (Underwater Curve)")
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown * 100, mode='lines', fill='tozeroy', line=dict(color='red'), name='Drawdown %'))
            fig_dd.update_layout(height=400, template="plotly_dark", yaxis_title="Visszaesés (%)")
            st.plotly_chart(fig_dd, use_container_width=True)
            st.info("A grafikon azt mutatja meg, hogy az eddigi legmagasabb tőkeértékhez (csúcshoz) képest hány százalékos lebegő vagy realizált mínuszban volt a portfólió.")

        with tab3:
            st.subheader(f"Végrehajtott Tranzakciók (Összesen: {total_trades} db)")
            # Csak azokat a sorokat mutatjuk, ahol tranzakció (belépés vagy kilépés) történt
            trades_df = df_portfolio[df_portfolio['Trade_Executed'] == True].copy()
            trades_df = trades_df[['Close', 'Signal', 'Position', 'Price_Change', 'PnL', 'Equity']]
            trades_df.index.name = "Dátum"
            st.dataframe(trades_df.style.format({'Close': '{:.2f}', 'PnL': '${:.2f}', 'Equity': '${:.2f}'}), use_container_width=True)

        with tab4:
            st.subheader("Az alkalmazott ICT Stratégia Logikája")
            st.markdown("""
            Ez a keretrendszer az **Inner Circle Trader (ICT)** koncepciók algoritmikus validálására épült.
            
            **A belépési feltételrendszer (Long pozíció esetén):**
            1. **Liquidity Sweep (Likviditás vadászat):** Az algoritmus figyeli az ázsiai kereskedési szakasz (00:00 - 08:00) mélypontját. Ha az árfolyam alá szúr, de felette zár be, a piac likviditást gyűjtött.
            2. **CHoCH / BOS (Piaci struktúra törése):** A gyertya egy korábbi 5-ös ablakú lokális csúcs felett kell, hogy zárjon, jelezve a trendfordulót.
            3. **VWAP (Intézményi átlagár):** A pozíció felvételének pillanatában az árfolyamnak a napi Volumen-súlyozott Átlagár (VWAP) felett kell lennie, megerősítve a bika trendet.
            
            *Az eladási (Short) pozíciók logikája ennek pontosan az inverze az ázsiai csúcsok felett.*
            """)
else:
    st.info("👈 Kérlek, indítsd el a szimulációt az oldalsávon található gombbal!")