import pandas as pd
import numpy as np
import yfinance as yf
import datetime

# Set float format for display
pd.options.display.float_format = '{:.3f}'.format       #adjust the number to change how many decimal places are displayed

# Add in stocks from the selected Portfolio
stocks = ['^SPX','AMP','SCHW','BRK-B','AAPL','STLA','YUM','ADM','AMR','PWR','XOM','CE','ELV','DGX','REGN','MRK','AMAT','GOOG','JBL','ADI']
# stocks = ['^SPX', 'INTC','AMD','NVDA']


# Define the time period
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=5*365)


# Initialize a dictionary to hold ticker data
stock_data = {}

# Fetch the monthly closing prices for each stock
for ticker in stocks:
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, interval="1mo")
    stock_data[ticker] = df['Close']

# Combine the data into a single DataFrame
combined_data = pd.DataFrame(stock_data)

# Clean up the DataFrame by dropping rows with NaN values that can occur at the end
combined_data.dropna(how='all', inplace=True)

# Calculate excess the month to month excess returns for each stock and store it in a new dataframe
excess_ret = combined_data.pct_change()


# Separate the S&P 500 returns and the stocks' returns
spx_returns = excess_ret.iloc[:, 0]
stocks_returns = excess_ret.iloc[:, 0:]

# Create an empty DataFrame to store the analysis results
perf_metrics = pd.DataFrame(index=excess_ret.columns)

# Calculate each metric
perf_metrics['Mean Annualized Return'] = stocks_returns.mean() * 12
perf_metrics['Annualized Std Dev'] = stocks_returns.std() * np.sqrt(12)


# Market variance for Beta calculations
market_variance = spx_returns.var() * 12

# Calculate metrics that require row-wise operations
for stock in stocks_returns:
    stock_returns = stocks_returns[stock]
    cov_with_market = stock_returns.cov(spx_returns) * 12  # Annualize the covariance
    beta = cov_with_market / market_variance                                                                   
    
    perf_metrics.loc[stock, 'SPX Correlation'] = stock_returns.corr(spx_returns)
    perf_metrics.loc[stock, 'Beta'] = beta
    perf_metrics.loc[stock, 'Total Variance'] = stock_returns.var() * 12  # Annualize the variance         
    perf_metrics.loc[stock, 'Systematic Variance'] = beta ** 2 * market_variance                           
    perf_metrics.loc[stock, 'Unique Variance'] = perf_metrics.loc[stock, 'Total Variance'] - perf_metrics.loc[stock, 'Systematic Variance']
    perf_metrics.loc[stock, 'R-squared'] = perf_metrics.loc[stock, 'SPX Correlation'] ** 2 


# Transpose the df
stks_anl_results_T = perf_metrics.transpose()

# Print the results
print(stks_anl_results_T)