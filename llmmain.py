# Currently using OpenAI/DeepSeek APIs for LLM-based trading insights.
# Experimenting with RAG Agents in collaboration with team.
# Future plans include incorporating sentiment analysis, pattern recognition from pattern dataset,
# and integration with financial news or social signals for enhanced decision-making.

import openai
import os
from dotenv import load_dotenv
from technical_indicators import get_data_with_indicators

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Fetch Historical Stock Data, specify ticker
ticker = "GOOG" 

# Define the date range
start_date = '2015-01-01'
end_date = '2025-04-23'

data = get_data_with_indicators(ticker=ticker, start_date=start_date, end_date=end_date)

# Flatten the columns after adding indicators
data.columns = data.columns.get_level_values(0)  # Flatten MultiIndex columns

# Access the most recent row of data
latest = data.dropna().iloc[-1]

# Ensure you are working with scalar values
close_price = latest['Close']
ema_200 = latest['EMA_200']
rsi = latest['RSI']
macd = latest['MACD']
macd_signal = latest['MACD_Signal']
bb_high = latest['BB_High']
bb_low = latest['BB_Low']

# Create the prompt for OpenAI
prompt = f"""
You are a stock trading assistant. Analyze the following market conditions for {ticker}:

- Close Price: {close_price:.2f}
- EMA 200: {ema_200:.2f}
- RSI: {rsi:.2f}
- MACD: {macd:.2f}
- MACD Signal: {macd_signal:.2f}
- Bollinger High: {bb_high:.2f}
- Bollinger Low: {bb_low:.2f}

Given this data, suggest whether to Buy, Hold, or Sell. Justify the recommendation based on technical analysis principles.
"""

# Send the prompt to OpenAI and print the response
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a financial market assistant."},
        {"role": "user", "content": prompt}
    ]
)

print(response['choices'][0]['message']['content'])
