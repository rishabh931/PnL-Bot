# 📈 Indian Stock P&L Analyzer

A Streamlit web app to analyze Indian stock financials (Annual & Quarterly) using data from Yahoo Finance.

## 🔍 Features

- Analyze any NSE-listed Indian stock (e.g., RELIANCE, INFY, TCS)
- View annual financial metrics:
  - Revenue, Operating Profit, OPM%, PBT, PAT, EPS
- View quarterly financial metrics:
  - Revenue, Operating Profit, OPM%, PBT, PAT, EPS
- Download CSVs of annual and quarterly data
- Interactive visualizations with trend analysis
- Key insights and growth metrics

## 🛠️ Installation

### Clone the repository:

```bash
git clone https://github.com/yourusername/stock-pl-analysis.git
cd stock-pl-analysis
```

### Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Run the App

```bash
streamlit run app.py
```

## 📦 File Structure

- `app.py`: Main Streamlit app
- `requirements.txt`: Python dependencies
- `analysis_template.ipynb`: (Optional) Notebook version of analysis
- `stock_data.csv`: (Optional) Sample CSV for testing

## 📚 Data Source

- Yahoo Finance via `yfinance` package

## 🔗 Links

- 📦 GitHub: [https://github.com/yourusername/stock-pl-analysis](https://github.com/yourusername/stock-pl-analysis)

---

## 📜 License

MIT License – feel free to use and modify!
