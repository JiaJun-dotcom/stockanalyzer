import yfinance as yf
import pandas as pd
import ta

def get_data_with_indicators(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    print(data.head())

    def add_technical_indicators(df):
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()  # Force to Series if accidentally DataFrame

        df['EMA_200'] = ta.trend.EMAIndicator(close=close, window=200)
        df['RSI'] = ta.momentum.RSIIndicator(close=close).rsi()
        macd = ta.trend.MACD(close=close)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        bb = ta.volatility.BollingerBands(close=close)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        return df

    data_with_indicators = add_technical_indicators(data)
    return data_with_indicators
