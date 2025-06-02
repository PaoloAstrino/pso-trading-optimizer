import yfinance as yf
import pandas as pd
import time
import numpy as np

# Function to get NASDAQ-100 components
def get_nasdaq100_symbols():
    try:
        # Method 1: Using the NASDAQ-100 ETF (QQQ) holdings
        qqq = yf.Ticker("QQQ")
        # Get holdings data - this might not always work depending on API limitations
        holdings = qqq.get_holdings()
        if holdings is not None and not holdings.empty:
            # Extract ticker symbols from holdings
            symbols = holdings.index.tolist()
            return symbols[:100]  # Return top 100 holdings
    except Exception as e:
        print(f"Error getting NASDAQ-100 symbols from QQQ: {e}")
    
    # Fallback method: Hardcoded list of NASDAQ-100 components
    # This is a snapshot and may not be current, but serves as a fallback
    nasdaq100 = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'COST',
        'PEP', 'ADBE', 'CSCO', 'NFLX', 'TMUS', 'CMCSA', 'AMD', 'TXN', 'INTC', 'INTU',
        'QCOM', 'AMGN', 'HON', 'AMAT', 'SBUX', 'ISRG', 'MDLZ', 'ADI', 'PYPL', 'REGN',
        'VRTX', 'GILD', 'PANW', 'KLAC', 'SNPS', 'CDNS', 'LRCX', 'ASML', 'ADP', 'BKNG',
        'CHTR', 'MAR', 'MNST', 'ORLY', 'CTAS', 'MRVL', 'ADSK', 'FTNT', 'ABNB', 'MELI',
        'PCAR', 'KDP', 'CRWD', 'DXCM', 'PAYX', 'KHC', 'NXPI', 'LULU', 'IDXX', 'WDAY',
        'FAST', 'ODFL', 'BIIB', 'EA', 'VRSK', 'ILMN', 'CPRT', 'CTSH', 'XEL', 'ROST',
        'CSGP', 'DLTR', 'ANSS', 'MCHP', 'TEAM', 'DDOG', 'SIRI', 'MTCH', 'ALGN', 'WBD',
        'FANG', 'EBAY', 'ZS', 'VRSN', 'SPLK', 'LCID', 'RIVN', 'SGEN', 'SWKS', 'ENPH',
        'DASH', 'OKTA', 'ZM', 'MRNA', 'GEHC', 'TTWO', 'DOCU', 'CTLT', 'NTAP', 'TOST'
    ]
    return nasdaq100

# Get NASDAQ-100 symbols
stocks = get_nasdaq100_symbols()
print(f"Found {len(stocks)} NASDAQ-100 stocks")

# Set the time period for data download (2012-2020)
start_date = "2012-01-01"
end_date = "2022-12-31"

# Function to download data with error handling and rate limiting
def download_stock_data(symbols, start_date, end_date, batch_size=20):
    all_data = pd.DataFrame()
    
    # Process in batches to avoid rate limits
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        print(f"Downloading batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1} ({len(batch)} stocks)")
        
        try:
            # Download data for this batch
            batch_data = yf.download(
                batch, 
                start=start_date,
                end=end_date,
                interval="1d",
                group_by='ticker',
                auto_adjust=True,
                progress=True
            )
            
            # If we have multiple stocks, the data will be multi-level
            if len(batch) > 1:
                # Extract just the Close prices
                close_prices = pd.DataFrame()
                for ticker in batch:
                    if (ticker, 'Close') in batch_data.columns:
                        close_prices[ticker] = batch_data[(ticker, 'Close')]
                
                # Merge with existing data
                if all_data.empty:
                    all_data = close_prices
                else:
                    all_data = pd.concat([all_data, close_prices], axis=1)
            else:
                # Single stock case
                ticker = batch[0]
                if 'Close' in batch_data.columns:
                    if all_data.empty:
                        all_data = pd.DataFrame()
                    all_data[ticker] = batch_data['Close']
            
            # Sleep to avoid hitting rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error downloading batch {i//batch_size + 1}: {e}")
            time.sleep(5)  # Wait longer after an error
    
    return all_data

# Download the data
print(f"Downloading daily closing prices for NASDAQ-100 stocks from {start_date} to {end_date}...")
close_prices = download_stock_data(stocks, start_date, end_date)

# Check for missing data
missing_pct = close_prices.isna().mean() * 100
print("\nMissing data percentage by stock:")
print(missing_pct.sort_values(ascending=False).head(10))

# Fill missing values with forward fill then backward fill
close_prices_filled = close_prices.fillna(method='ffill').fillna(method='bfill')

# Save to CSV
output_file = 'nasdaq100_2012_2020.csv'
close_prices_filled.to_csv(output_file)
print(f"\nData saved to {output_file}")
print(f"Downloaded data for {close_prices_filled.shape[1]} stocks over {close_prices_filled.shape[0]} trading days")

# Show the first few rows
print("\nSample of the data:")
print(close_prices_filled.head())