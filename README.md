📈 Stock Price Prediction with LSTM Neural Networks
This project explores time series forecasting using Long Short-Term Memory (LSTM), a powerful variant of Recurrent Neural Networks (RNNs), to predict daily stock prices based on historical price data. 
This implementation demonstrates the use of deep learning for financial forecasting, serves as my foundational exploration into time-series forecasting to work with sequential data.

🚀 Key Features
✅ Fetches real-time historical stock data using yfinance

✅ Preprocesses data using MinMaxScaler for normalization

✅ Constructs an LSTM-based neural network for time series prediction

✅ Implements a train-validation split to evaluate generalization

✅ Uses early stopping to avoid overfitting

✅ Visualizes predicted vs. actual stock prices

Model Architecture: 3 LSTM layers, dropout, dense + output layers

Output: Graphs showing actual vs predicted prices + loss trends

📊 Usage
Run the prediction script by specifying a stock ticker eg GOOG/TSLA

🧠 Model Architecture
🔁 Three LSTM layers with 100 units each and 20% dropout

🔸 Dense layer with 50 neurons (ReLU activation)

🔚 Output layer with 1 neuron to predict the next closing price

🧮 Loss Function: Mean Squared Error (MSE)

⚙️ Optimizer: Adam

📈 Results:
The model forecasts the next day's stock price based on prior days. It plots the prediction curve against the actual closing prices, providing a visual insight into the model’s performance.

📊 Visualizations:
📉 Loss Curve – Training vs. Validation loss

📈 Prediction Graph – Actual vs. Predicted stock prices


LOOKING FORWARD:
Multi-Modal AI System(In Progress):
* Pattern recognition through openCV for "Double Top" and "Double Bottom"
User Input: Uploads chart image + types eg: "TSLA — what trade can I take here?"

1. 🖼️ Image → Pattern
   - Use a CNN or CLIP to classify the chart as "Double Top"(Pretrain a CNN model here with the dataset we have of "Double Top" and "Double Bottom" price patterns, using just these 2 for now and extend to more patterns in future.)

2. 📊 Pull particular stock ticker from market context
   - Fetch eg TSLA, recent OHLCV (open-high-low-close-volume, with specific timeframe, using yfinance api)
   - Recent news headline embeddings (ChromaDB + RAG, prompt with LangChain)

3. ⚙️ Strategy Generator
   Provide entry price, stop loss, target price, based on factors like pattern recognition(if the chart matches any of the patterns model was trained on, scan chart(through screen capture of user screen) for pattern first).
   Also include technical indicators like MACD, RSI, Bollinger Bands and news sentiment analysis.

   Sample pattern recognition response:
   If "Double Top":
   - Entry: Below neckline support level
   - Stop Loss: Above recent peak
   - Target: Height of pattern subtracted from neckline

4. 🧠 LLM Response Generator (with llama.cpp)
   - Inputs:
     - Pattern = Double Top
     - TSLA close price = $192
     - Suggested trade plan
   - Output:
     > “TSLA formed a double top. Entry: below $190. Stop loss: $200. Target: $170. This setup historically signals a bearish reversal with 60–70% probability.”

5. 📤 Return answer to user(Price chart + ticker → Receives full AI-generated trade analysis)
