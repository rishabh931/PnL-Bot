import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import re
import calendar
from datetime import datetime

# Configure page
st.set_page_config(page_title="Stock P&L Analyzer", layout="wide")
st.title("📈 Indian Stock P&L Statement Analysis")

# Function to format large numbers
def format_number(num):
    if abs(num) >= 1e7:  # Crores
        return f"₹{num/1e7:,.2f} Cr"
    elif abs(num) >= 1e5:  # Lakhs
        return f"₹{num/1e5:,.2f} L"
    elif abs(num) >= 1000:  # Thousands
        return f"₹{num/1000:,.2f} K"
    return f"₹{num:,.2f}"

# Function to find stock symbol
def find_stock_symbol(query):
    query = query.upper().strip()
    if not query.endswith('.NS'):
        query += '.NS'
    if not re.match(r"^[A-Z0-9.-]{1,20}\.NS$", query):
        return None
    return query

# Function to download financial data
def get_financials(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        if stock.info.get('regularMarketPrice') is None:
            return None, None, "Invalid stock symbol"
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        financial_data = {}

        # Revenue
        if 'Total Revenue' in income_stmt.index:
            financial_data['Revenue'] = income_stmt.loc['Total Revenue'].head(4).values[::-1]
        elif 'Revenue' in income_stmt.index:
            financial_data['Revenue'] = income_stmt.loc['Revenue'].head(4).values[::-1]
        else:
            return None, None, "Revenue data not available"

        # Operating Profit
        if 'Operating Income' in income_stmt.index:
            financial_data['Operating Profit'] = income_stmt.loc['Operating Income'].head(4).values[::-1]
        elif 'Operating Profit' in income_stmt.index:
            financial_data['Operating Profit'] = income_stmt.loc['Operating Profit'].head(4).values[::-1]
        else:
            return None, None, "Operating Profit data not available"

        # PBT
        if 'Pretax Income' in income_stmt.index:
            financial_data['PBT'] = income_stmt.loc['Pretax Income'].head(4).values[::-1]
        else:
            return None, None, "PBT data not available"

        # PAT
        if 'Net Income' in income_stmt.index:
            financial_data['PAT'] = income_stmt.loc['Net Income'].head(4).values[::-1]
        else:
            return None, None, "PAT data not available"

        # Shares Outstanding
        if 'Ordinary Shares Number' in balance_sheet.index:
            shares_outstanding = balance_sheet.loc['Ordinary Shares Number'].head(4).values[::-1]
        elif 'Share Issued' in balance_sheet.index:
            shares_outstanding = balance_sheet.loc['Share Issued'].head(4).values[::-1]
        else:
            shares = stock.info.get('sharesOutstanding')
            if shares:
                shares_outstanding = np.array([shares] * 4)
            else:
                return None, None, "Shares outstanding data not available"

        # Calculate EPS
        financial_data['EPS'] = financial_data['PAT'] / (shares_outstanding / 1e6)
        # Calculate metrics
        financial_data['OPM %'] = (financial_data['Operating Profit'] / financial_data['Revenue']) * 100
        # Fixed EPS Growth % calculation
        eps_growth = [0]
        for i in range(1, len(financial_data['EPS'])):
            prev = financial_data['EPS'][i-1]
            if prev != 0:
                growth = ((financial_data['EPS'][i] - prev) / abs(prev)) * 100
            else:
                growth = 0
            eps_growth.append(growth)
        financial_data['EPS Growth %'] = eps_growth
        years = income_stmt.columns[:4].strftime('%Y').values[::-1]
        return pd.DataFrame(financial_data, index=years), years, None
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# Function to get quarterly financial data
def get_quarterly_financials(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        quarterly_income = stock.quarterly_financials
        quarterly_data = {}

        # Revenue
        if 'Total Revenue' in quarterly_income.index:
            quarterly_data['Revenue'] = quarterly_income.loc['Total Revenue'].head(6).values[::-1]
        elif 'Revenue' in quarterly_income.index:
            quarterly_data['Revenue'] = quarterly_income.loc['Revenue'].head(6).values[::-1]
        else:
            return None, None, "Quarterly revenue data not available"

        # Operating Profit
        if 'Operating Income' in quarterly_income.index:
            quarterly_data['Operating Profit'] = quarterly_income.loc['Operating Income'].head(6).values[::-1]
        elif 'Operating Profit' in quarterly_income.index:
            quarterly_data['Operating Profit'] = quarterly_income.loc['Operating Profit'].head(6).values[::-1]
        else:
            return None, None, "Quarterly operating profit data not available"

        # PBT
        if 'Pretax Income' in quarterly_income.index:
            quarterly_data['PBT'] = quarterly_income.loc['Pretax Income'].head(6).values[::-1]
        else:
            return None, None, "Quarterly PBT data not available"

        # PAT
        if 'Net Income' in quarterly_income.index:
            quarterly_data['PAT'] = quarterly_income.loc['Net Income'].head(6).values[::-1]
        else:
            return None, None, "Quarterly PAT data not available"

        # Calculate OPM%
        quarterly_data['OPM %'] = (quarterly_data['Operating Profit'] / quarterly_data['Revenue']) * 100
        # Get EPS from quarterly earnings
        quarterly_earnings = stock.quarterly_earnings
        if not quarterly_earnings.empty and 'EPS' in quarterly_earnings.columns:
            quarterly_data['EPS'] = quarterly_earnings['EPS'].head(6).values[::-1]
        else:
            shares = stock.info.get('sharesOutstanding', 1)
            quarterly_data['EPS'] = quarterly_data['PAT'] / (shares / 1e6)
        # Calculate EPS Growth %
        quarterly_data['EPS Growth %'] = [0] + [
            ((quarterly_data['EPS'][i] - quarterly_data['EPS'][i-1]) / 
             abs(quarterly_data['EPS'][i-1]) * 100)
            if quarterly_data['EPS'][i-1] != 0 else 0
            for i in range(1, len(quarterly_data['EPS']))
        ]
        # Get quarter labels
        quarters = quarterly_income.columns[:6].strftime('%Y-%m').values[::-1]
        formatted_quarters = []
        for q in quarters:
            year, month = q.split('-')
            quarter = (int(month) - 1) // 3 + 1
            fiscal_year = int(year) if int(month) > 3 else int(year) - 1
            formatted_quarters.append(f"Q{quarter} FY{str(fiscal_year)[2:]}")
        return pd.DataFrame(quarterly_data, index=formatted_quarters), formatted_quarters, None
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# The rest of the code remains unchanged...
