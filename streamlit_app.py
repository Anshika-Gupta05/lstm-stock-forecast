import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import plotly.graph_objs as go

st.set_page_config(page_title="Stock Price Prediction with LSTM",page_icon="📈", layout="wide")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=EB+Garamond&display=swap');

        html, body, .main, .block-container, .stButton>button, .css-1d391kg, .css-1v3fvcr, [class*="css"] {
            font-family: 'EB Garamond', serif !important;
        }

        /* Also apply font to headers, labels, inputs, and other common elements */
        h1, h2, h3, h4, h5, h6, p, label, span, div, button, input, select, textarea {
            font-family: 'EB Garamond', serif !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("Options")

companies = {
    'HDFC Bank': 'HDB',
    'Intra-Cellular Therapies': 'ITCI',
    'ONGC': 'ONGC.NS',
    'Tata Steel': 'TATASTEEL.NS',
    'Britannia': 'BRITANNIA.NS',
    'Reliance': 'RELIANCE.NS',
    'Divis Lab': 'DIVISLAB.NS',
    'Bajaj Finance': 'BAJFINANCE.NS',
    'SBI': 'SBIN.NS',
    'Asian Paints': 'ASIANPAINT.NS'
}

selected_company_name = st.sidebar.selectbox("Select a Company", list(companies.keys()))
ticker = companies[selected_company_name]

# --- Load Data ---
end = datetime.now()
start = datetime(end.year - 4, end.month, end.day)

@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, start=start, end=end)

    # Flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Use 'Adj Close' if 'Close' missing
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']

    return df[['Close']].dropna()

df = load_data(ticker)

st.title("📈 Stock Price Predictor with LSTM")

# --- Plot Price Trend with Plotly ---
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
fig_price.update_layout(title=f"Price Trend for {selected_company_name}",
                        xaxis_title="Date", yaxis_title="Price",
                        hovermode="x unified")
st.plotly_chart(fig_price, use_container_width=True)

# --- Preprocessing ---
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

seq_len = 60
X, y = [], []
for i in range(seq_len, len(scaled_data)):
    X.append(scaled_data[i - seq_len:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# --- Split Data ---
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- LSTM Model ---
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

# --- Predictions ---
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# --- Plotting Predictions with Plotly ---
train_idx = range(seq_len, seq_len + len(train_predict))
test_idx = range(seq_len + len(train_predict), seq_len + len(train_predict) + len(test_predict))

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=df.index[train_idx], y=y_train_actual.flatten(), mode='lines', name='Train Actual'))
fig_pred.add_trace(go.Scatter(x=df.index[train_idx], y=train_predict.flatten(), mode='lines', name='Train Predict'))
fig_pred.add_trace(go.Scatter(x=df.index[test_idx], y=y_test_actual.flatten(), mode='lines', name='Test Actual'))
fig_pred.add_trace(go.Scatter(x=df.index[test_idx], y=test_predict.flatten(), mode='lines', name='Test Predict'))
fig_pred.update_layout(title=f"LSTM Prediction for {selected_company_name}",
                       xaxis_title="Date", yaxis_title="Price",
                       hovermode="x unified")
st.plotly_chart(fig_pred, use_container_width=True)

# --- Evaluation ---
st.subheader("📊 Model Evaluation")
st.write(f"**Train MAPE:** {mean_absolute_percentage_error(y_train_actual, train_predict):.4f}")
st.write(f"**Test MAPE:** {mean_absolute_percentage_error(y_test_actual, test_predict):.4f}")
st.write(f"**Train R²:** {r2_score(y_train_actual, train_predict):.4f}")
st.write(f"**Test R²:** {r2_score(y_test_actual, test_predict):.4f}")

# --- Year-wise Best Performing Companies ---
st.subheader("📅 Year-wise Best Performing Companies")

@st.cache_data
def get_all_data(companies_dict, start, end):
    data = {}
    for name, symbol in companies_dict.items():
        dfc = yf.download(symbol, start=start, end=end)
        if dfc.empty:
            continue

        # Flatten MultiIndex columns if any
        if isinstance(dfc.columns, pd.MultiIndex):
            dfc.columns = dfc.columns.get_level_values(0)

        # Use 'Adj Close' if 'Close' missing
        if 'Close' not in dfc.columns and 'Adj Close' in dfc.columns:
            dfc['Close'] = dfc['Adj Close']

        if 'Close' not in dfc.columns:
            print(f"No 'Close' data for {name} ({symbol}), skipping.")
            continue

        dfc["Company"] = name
        data[name] = dfc
    return data

data_all = get_all_data(companies, start, end)

annual_returns = pd.DataFrame()
for name, dfc in data_all.items():
    dfc = dfc.copy()
    if "Close" not in dfc.columns:
        continue
    dfc["Year"] = dfc.index.year
    dfc = dfc.dropna(subset=["Close"])
    if dfc.empty:
        continue
    returns = dfc.groupby("Year")["Close"].apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100)
    returns = returns.to_frame(name=name)
    annual_returns = annual_returns.join(returns, how="outer")


best_performers = annual_returns.idxmax(axis=1)
best_performance_values = annual_returns.max(axis=1)

best_df = pd.DataFrame({
    "Year": best_performers.index,
    "Best Company": best_performers.values,
    "Annual Return (%)": best_performance_values.values
})
st.dataframe(best_df)

# --- Plot Annual Returns with Plotly ---
for year in best_df["Year"]:
    fig_year = go.Figure()
    values = annual_returns.loc[year]
    fig_year.add_trace(go.Bar(x=values.index, y=values.values, name="All Companies"))
    
    best = best_df.loc[best_df["Year"] == year]
    fig_year.add_trace(go.Bar(x=[best["Best Company"].values[0]], 
                              y=[best["Annual Return (%)"].values[0]],
                              marker_color='red', name='Best Company'))
    
    fig_year.update_layout(title=f"Annual Returns in {year}", xaxis_title="Company", yaxis_title="Return (%)")
    st.plotly_chart(fig_year, use_container_width=True)

# --- Overall Cumulative Returns ---
st.subheader("🏆 Overall Best Performing Company")

cumulative_returns = {}
for name, dfc in data_all.items():
    if len(dfc) > 1 and 'Close' in dfc.columns:
        start_price = dfc["Close"].iloc[0]
        end_price = dfc["Close"].iloc[-1]
        cumulative_returns[name] = (end_price / start_price - 1) * 100

if cumulative_returns:
    best_company = max(cumulative_returns, key=cumulative_returns.get)
    best_return = cumulative_returns[best_company]

    fig_cum = go.Figure()
    colors = ['green' if name == best_company else 'skyblue' for name in cumulative_returns.keys()]
    fig_cum.add_trace(go.Bar(x=list(cumulative_returns.keys()),
                             y=list(cumulative_returns.values()),
                             marker_color=colors))

    fig_cum.update_layout(title="Cumulative Returns Over 4 Years",
                          xaxis_title="Company", yaxis_title="Return (%)",
                          xaxis_tickangle=45)
    st.plotly_chart(fig_cum, use_container_width=True)

    st.success(f"🏅 **Best Overall Performer**: {best_company} with a return of {best_return:.2f}%")
else:
    st.warning("No cumulative returns data available.")

