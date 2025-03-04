# Machine Learning and Deep Learning Investment Strategies

## 📌 Project Overview
This project evaluates machine learning and deep learning models for stock market prediction. It integrates technical indicators, historical price data, and advanced ML/DL techniques to develop and backtest trading strategies across developed, emerging, and frontier markets.

## 📊 Features
- **Data Collection:**
  - Uses Yahoo Finance and other APIs to fetch stock market data.
  - Covers global markets: S&P 500, DAX, FTSE 100, VNINDEX, BET, etc.
  
- **Feature Engineering:**
  - Implements RSI, MACD, Bollinger Bands, VWAP, Ichimoku Cloud, and over 100+ technical indicators.
  - Uses Fourier Transform and statistical features to enhance signal extraction.
  
- **Machine Learning Models:**
  - Traditional ML: Logistic Regression, SVM, Decision Trees, Random Forest, XGBoost, LightGBM.
  - Deep Learning: LSTMs, GRUs, CNNs, Transformers, GANs, Hybrid Models (CNN-LSTM, CNN-Transformer).
  
- **Backtesting & Validation:**
  - Combinatorial Purged Cross-Validation (CPCV) ensures robustness.
  - Sharpe Ratio, Sortino Ratio, Maximum Drawdown used for strategy evaluation.

- **Trading Signal Generation & Strategy Development:**
  - Translates predictions into buy/sell signals.
  - Compares different ML/DL-based strategies with buy-and-hold benchmarks.

## 🛠 Installation & Dependencies
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Yahoo Finance API (if needed):**
   - Some data fetching functions may require authentication or API keys.
   - Ensure you have the required API access for live data fetching.

## 📂 Project Structure
```
├── data/                     # Raw & Processed Data
├── models/                   # Trained ML & DL models
├── notebooks/                # Jupyter Notebooks for Exploratory Analysis
├── src/                      # Core scripts for data processing and model training
│   ├── data_collection.py    # Fetch stock data from Yahoo Finance
│   ├── feature_engineering.py# Compute technical indicators
│   ├── model_training.py     # Train ML/DL models
│   ├── backtesting.py        # Implement backtesting framework
│   ├── evaluation.py         # Compute performance metrics
├── results/                  # Model evaluation and backtest results
├── README.md                 # Project documentation
├── requirements.txt          # Required dependencies
├── LICENSE                   # License information
```

## 📈 Usage
1. **Fetch Data:**
   ```python
   from src.data_collection import fetch_data
   df = fetch_data(tickers=['^GSPC', '^VNINDEX'], start_date="2000-01-01", end_date="2025-01-01")
   ```

2. **Generate Features:**
   ```python
   from src.feature_engineering import compute_indicators
   df = compute_indicators(df)
   ```

3. **Train Model:**
   ```python
   from src.model_training import train_model
   model = train_model(df, model_type="LSTM")
   ```

4. **Backtest Strategy:**
   ```python
   from src.backtesting import backtest_strategy
   results = backtest_strategy(model, df)
   ```

5. **Evaluate Performance:**
   ```python
   from src.evaluation import evaluate_model
   metrics = evaluate_model(model, df)
   ```

## 📊 Results
- **Key Findings:**
  - Deep Learning models (LSTM, GRU, Transformers) showed superior performance over traditional ML models.
  - CPCV enhanced robustness by reducing lookahead bias.
  - Hybrid CNN-LSTM and CNN-Transformer models demonstrated the best generalization across different market conditions.
  
- **Trading Strategy Performance:**
  - Compared ML/DL-based signals with a buy-and-hold benchmark.
  - Evaluated profitability using Sharpe Ratio, Sortino Ratio, and Maximum Drawdown.

## 📜 License
This project is licensed under the **MIT License** – feel free to modify and distribute.

