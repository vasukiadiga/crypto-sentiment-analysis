# 📊 Crypto Trader Behavior vs. Market Sentiment

## 🎯 Project Overview
This project analyzes over 211,000 individual cryptocurrency trades to determine how macro-market sentiment—specifically the Crypto Fear & Greed Index—impacts individual trader behavior, trade frequency, and overall profitability. 

By combining exploratory data analysis, K-Means clustering, and a Gradient Boosting predictive model, this repository provides actionable, data-driven strategies for risk management and capital allocation.

---

## 📂 Repository Contents
* `crypto_analysis.ipynb`: The core Jupyter Notebook containing all data cleaning, feature engineering, clustering, and predictive modeling code.
* `app.py`: The interactive Streamlit dashboard summarizing the insights.
* `historical_data.zip`: Zip file containing Raw dataset with 211k+ individual trade executions.
* `fear_greed_index.csv`: Raw dataset containing daily Crypto Fear & Greed Index values.
* `requirements.txt`: List of dependencies for reproducibility.

---

## 🚀 How to Run (Reproducibility)

**1. Clone the repository:**
~~~bash
git clone https://github.com/vasukiadiga/crypto-sentiment-analysis.git
cd crypto-sentiment-analysis
~~~

**2. Install dependencies:**
It is highly recommended to use a virtual environment.
~~~bash
pip install -r requirements.txt
~~~

**3. Explore the Analysis & Dashboard:**
* **To view the core data science work:** Open `crypto_analysis.ipynb` in Jupyter Notebook or VS Code to step through the data pipeline, K-Means clustering, and Gradient Boosting model evaluation.
* **To launch the interactive dashboard:**
~~~bash
python -m streamlit run app.py
~~~

---

## 🧠 Methodology
1. **Data Engineering:** Standardized timestamps (IST to UTC daily) to accurately merge trade logs with daily sentiment scores. Condensed raw trades into 2,340 daily trader profiles calculating Net PnL (adjusted for fees), Win Rates, Trade Frequency, and Average Sizing.
2. **Behavioral Segmentation:** Applied **K-Means Clustering** to scaled lifetime metrics to identify three distinct trader archetypes: *"The Whales"*, *"The Machine-Gunners" (Bots)*, and *"The Retail Gamblers"*.
3. **Predictive Modeling:** Trained and tuned a **Gradient Boosting Classifier** via 5-Fold Stratified Cross-Validation. The model predicts next-day trader profitability with ~70% accuracy based on their current-day behavior and market sentiment.

---

## 💡 Key Findings
1. **"Panic Trading" is Quantifiable:** During 'Extreme Fear' days, the median daily trade frequency doubles (jumping from ~26 to 50.5 trades/day). Consequently, the average win rate plummets to its lowest level across all market conditions (77%).
2. **The "Whale" Skew in Bear Markets:** On standard 'Fear' days, there is a massive discrepancy between Average PnL ($5,182) and Median PnL ($97). This indicates that while everyday retail traders struggle, a small handful of highly capitalized, sophisticated traders effectively "buy the dip" to capture massive upside.
3. **Behavior Trumps Sentiment:** The Gradient Boosting model revealed that a trader's internal risk management—specifically their Average Trade Size (28% feature importance) and Trade Frequency (25% importance)—are vastly stronger predictors of their future success than broader market sentiment (Fear/Greed Index, 16% importance).

---

## 🎯 Actionable Strategy Recommendations

Based on the quantitative findings, I propose the following dynamic risk-management rules:

* **🛡️ Rule 1: The "Fear Circuit-Breaker" (For Retail Accounts)**
  * **Trigger:** Fear & Greed Index drops to 'Extreme Fear'.
  * **Target:** Accounts mathematically clustered as "Retail Gamblers".
  * **Action:** Implement a hard daily trade-limit cap.
  * **Rationale:** The data proves this segment will double their trade frequency and drastically lower their win rate during these periods. A circuit-breaker preserves capital when the market (and the trader) acts irrationally.

* **📈 Rule 2: The "Greed Multiplier" (For Algorithmic/High-Frequency Accounts)**
  * **Trigger:** Fear & Greed Index rises to 'Extreme Greed'.
  * **Target:** Accounts mathematically clustered as "Machine-Gunners / Bots".
  * **Action:** Incrementally increase position size limits.
  * **Rationale:** Extreme Greed yields the highest median PnL ($362/day) and highest win rates (87%) without causing erratic trading spikes. This allows consistent, high-win-rate strategies to scale safely in clean, trending environments.

---

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (K-Means, Gradient Boosting, GridSearchCV)
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit
