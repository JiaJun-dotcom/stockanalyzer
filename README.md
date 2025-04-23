User Input: Uploads chart image + types eg: "TSLA — what trade can I take here?"

1. 🖼️ Image → Pattern
   - Use a CNN or CLIP to classify the chart as "Double Top"(might need pretrain a CNN model here with the dataset we have of "Double Top" and "Double Bottom" price patterns, use just these 2 for now and extend to more patterns in future.)

2. 📊 Pull TSLA market context
   - Fetch TSLA recent OHLCV (open-high-low-close-volume, with specific timeframe, using yfinance api)
   - Recent news headline embeddings (vector DB + RAG)

3. ⚙️ Strategy Generator
   Provide entry price, stop loss, target price, based on factors like pattern recognition(if the chart matches any of the patterns model was trained on, scan chart for pattern first), Technical indicators like MACD, RSI, news headline embeddings

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

5. 📤 Return answer to user
