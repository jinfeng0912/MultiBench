"""Implements dataloaders for the robotics datasets."""

from robustness.timeseries_robust import add_timeseries_noise
import copy
import datetime
from posixpath import split
import io
import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import DataLoader
from torch import nn
import yfinance as yf
import time
from pandas_datareader import data as pdr

def get_dataloader(stocks, input_stocks, output_stocks, batch_size=16, train_shuffle=True, start_date=datetime.datetime(2000, 6, 1), end_date=datetime.datetime(2021, 2, 28), window_size=500, val_split=3200, test_split=3700, modality_first=True, cuda=True, cache_dir='datasets/stocks/cache'):
    """Generate dataloader for stock data.

    Args:
        stocks (list): List of strings of stocks to grab data for.
        input_stocks (list): List of strings of input stocks
        output_stocks (list): List of strings of output stocks
        batch_size (int, optional): Batchsize. Defaults to 16.
        train_shuffle (bool, optional): Whether to shuffle training dataloader or not. Defaults to True.
        start_date (datetime, optional): Start-date to grab data from. Defaults to datetime.datetime(2000, 6, 1).
        end_date (datetime, optional): End-date to grab data from. Defaults to datetime.datetime(2021, 2, 28).
        window_size (int, optional): Window size. Defaults to 500.
        val_split (int, optional): Number of samples in validation split. Defaults to 3200.
        test_split (int, optional): Number of samples in test split. Defaults to 3700.
        modality_first (bool, optional): Whether to make modality the first index or not. Defaults to True.
        cuda (bool, optional): Whether to load data to cuda objects or not. Defaults to True.
        cache_dir (str, optional): Directory to cache downloaded stock data. Defaults to 'datasets/stocks/cache'.

    Returns:
        tuple: Tuple of training data-loader, test data-loader, and validation data-loader.
    """
    import os
    from pathlib import Path
    
    stocks = np.array(stocks)
    
    # Create cache directory if it doesn't exist
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    def _fetch_finance_data(symbol, start, end):
        # The old function using requests is unreliable.
        # url = f'https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={start.strftime("%s")}&period2={end.strftime("%s")}&interval=1d&events=history&includeAdjustedClose=true'
        # user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        # text = requests.get(url, headers={'User-Agent': user_agent}).text
        # return pd.read_csv(io.StringIO(text), encoding='utf8', parse_dates=True, index_col=0)
        
        # Use the much more stable yfinance library instead.
        return yf.download(symbol, start=start, end=end)

    # In get_dataloader function in get_data.py

    # Download all stocks (with caching)
    all_data = []
    print("Loading stock data (cached if available)...")
    for i, stock in enumerate(stocks):
        cache_file = cache_path / f"{stock}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        
        # Try loading from cache first
        if cache_file.exists():
            try:
                fetch = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                # Standardize column names from cache (same as download path)
                col_map = {c: c.capitalize() if c.lower() in ['open', 'high', 'low', 'close', 'volume', 'date', 'adj close'] else c for c in fetch.columns}
                fetch = fetch.rename(columns=col_map)
                print(f"  ✓ Loaded {stock} from cache ({i+1}/{len(stocks)})")
            except Exception as e:
                print(f"  Cache corrupted for {stock}, re-downloading... ({e})")
                fetch = None
        else:
            fetch = None
        
        # Download if not in cache
        if fetch is None or fetch.empty:
            print(f"Fetching {stock} ({i+1}/{len(stocks)})...")
            
            # Try yfinance first
            try:
                fetch = yf.download(stock, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if not fetch.empty and isinstance(fetch, pd.DataFrame):
                    print(f"  ✓ Downloaded {stock} via yfinance")
                else:
                    fetch = None
            except Exception as e:
                print(f"  yfinance failed for {stock}: {e}")
                fetch = None
            
            # Fallback to Stooq if yfinance failed
            if fetch is None or (isinstance(fetch, pd.DataFrame) and fetch.empty):
                try:
                    print(f"  Trying Stooq as fallback for {stock}...")
                    fetch = pdr.DataReader(stock, 'stooq', start=start_date, end=end_date)
                    if isinstance(fetch, pd.DataFrame) and not fetch.empty:
                        fetch = fetch.sort_index()
                        # Stooq column names are lowercase, standardize them
                        col_map = {c.lower(): c.capitalize() for c in fetch.columns}
                        fetch = fetch.rename(columns=col_map)
                        print(f"  ✓ Downloaded {stock} via Stooq")
                    else:
                        fetch = None
                except Exception as e:
                    print(f"  Stooq also failed for {stock}: {e}")
                    fetch = None
            
            # Save to cache if download succeeded
            if fetch is not None and not fetch.empty:
                try:
                    # Ensure consistent format before saving
                    fetch_to_save = fetch.copy()
                    # Make sure Date is the index
                    if 'Date' in fetch_to_save.columns:
                        fetch_to_save = fetch_to_save.set_index('Date')
                    elif fetch_to_save.index.name != 'Date':
                        fetch_to_save.index.name = 'Date'
                    # Save with Date as index
                    fetch_to_save.to_csv(cache_file)
                    print(f"  💾 Cached {stock} to {cache_file.name}")
                except Exception as e:
                    print(f"  Warning: Failed to cache {stock}: {e}")
            
            time.sleep(0.5)  # Rate limit
        
        if fetch is not None and not fetch.empty:
            # Reset index to make Date a column (for long format)
            if fetch.index.name or 'Date' not in fetch.columns:
                fetch = fetch.reset_index()
            
            # Standardize the date column name (could be 'index', 'Date', 'Price', 'Unnamed: 0', etc.)
            date_candidates = ['index', 'Price', 'Unnamed: 0', 'level_0']
            for candidate in date_candidates:
                if candidate in fetch.columns and 'Date' not in fetch.columns:
                    fetch = fetch.rename(columns={candidate: 'Date'})
                    break
            
            # Ensure we have a Date column
            if 'Date' not in fetch.columns:
                print(f"  WARNING: No Date column found for {stock}, columns: {list(fetch.columns)}")
                continue
            
            # Normalize to long format: only keep Date, Open, Symbol
            # Handle different column layouts (single-level, MultiIndex, suffixed)
            open_series = None
            if isinstance(fetch.columns, pd.MultiIndex):
                if ('Open', stock) in fetch.columns:
                    open_series = fetch[('Open', stock)]
                elif ('Open', '') in fetch.columns:
                    open_series = fetch[('Open', '')]
            else:
                col_candidates = [f'Open_{stock}', 'Open']
                for c in col_candidates:
                    if c in fetch.columns:
                        open_series = fetch[c]
                        break
            if open_series is None and 'Open' in fetch.columns:
                open_series = fetch['Open']

            if open_series is None:
                print(f"  WARNING: No 'Open' column found for {stock}, columns: {list(fetch.columns)[:10]}")
                continue

            normalized = pd.DataFrame({
                'Date': fetch['Date'],
                'Open': open_series,
                'Symbol': stock
            })
            all_data.append(normalized)
        else:
            print(f"  ✗ Skipping {stock} (all sources failed)")
            
    if not all_data:
        raise ValueError("Failed to download any stock data. Please check your network connection or try again later.")

    # Concatenate all downloaded data into a single DataFrame (long format: each row = one date for one stock)
    print(f"DEBUG: Concatenating {len(all_data)} dataframes...")
    for i, df in enumerate(all_data[:3]):  # Check first 3
        print(f"  DataFrame {i}: shape={df.shape}, columns={list(df.columns)[:5]}")
    
    data = pd.concat(all_data, ignore_index=True, sort=False)
    
    print(f"DEBUG: After concat - shape={data.shape}, columns={list(data.columns)[:10]}")
    print(f"DEBUG: Column types: {[type(c) for c in data.columns[:5]]}")
    
    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        # Extract only the first level (column name), discard second level (stock symbol from concat)
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        print(f"DEBUG: After flatten MultiIndex - columns={list(data.columns)[:10]}")
    
    # Standardize column names (Stooq uses lowercase, yfinance uses title case)
    data.columns = [c.capitalize() if isinstance(c, str) and c.lower() in ['open', 'high', 'low', 'close', 'volume', 'date', 'adj close'] else c for c in data.columns]
    print(f"DEBUG: After standardize - columns={list(data.columns)[:10]}")
    
    # Ensure we have the required columns
    required_cols = ['Open', 'Symbol', 'Date']
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        print(f"DEBUG: Available columns: {list(data.columns)}")
        print(f"DEBUG: First few rows:\n{data.head()}")
        raise ValueError(f"Missing required columns after download: {missing}. Available: {list(data.columns)}")
    
    # Convert to pivot table: rows=dates, columns=stocks
    # This handles mixed sources (yfinance + Stooq) with different date ranges
    
    # Filter out any non-date rows (e.g., 'Ticker' header that may have been cached)
    data = data[data['Date'] != 'Date']  # Remove header row if present
    data = data[data['Date'] != 'Ticker']  # Remove 'Ticker' if present
    data = data.dropna(subset=['Date'])  # Remove NaN dates
    
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date'])  # Remove rows where Date conversion failed
    
    pivot = data.pivot_table(index='Date', columns='Symbol', values='Open', aggfunc='first')
    
    print(f"DEBUG: Pivot columns (stocks): {list(pivot.columns)}")
    print(f"DEBUG: Requested stocks: {list(stocks)}")
    
    # Ensure all requested stocks are present
    missing_stocks = [s for s in stocks if s not in pivot.columns]
    if missing_stocks:
        raise ValueError(f"Failed to download data for stocks: {missing_stocks}")
    
    # Keep only requested stocks and forward-fill missing dates
    pivot = pivot[stocks].ffill().bfill()
    
    # Drop any rows that still have NaN (beginning/end of series)
    pivot = pivot.dropna()
    
    if len(pivot) < window_size + test_split + 100:
        raise ValueError(f"Insufficient data: only {len(pivot)} days available, need at least {window_size + test_split + 100}")
    
    # Map stock symbols to column indices
    stock_to_idx = {symbol: i for i, symbol in enumerate(stocks)}
    input_stocks = np.array([stock_to_idx[x] for x in input_stocks])
    output_stocks = np.array([stock_to_idx[x] for x in output_stocks])

    # Convert to tensor: shape (n_days, n_stocks)
    X = torch.tensor(pivot.values.astype(float), dtype=torch.float32)
    RX = torch.log(X[1:] / X[:-1])
    Y = RX[window_size:, output_stocks]
    Y = Y * Y

    RX = RX / torch.std(RX[:window_size + val_split])
    Y = Y / torch.std(Y[:val_split])

    X = [RX[i:i + window_size, input_stocks].reshape(
        1, window_size, -1) for i in range(len(RX) - window_size)]
    X = torch.cat(X)

    if cuda:
        X = X.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        Y = Y.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    class _MyDataset(torch.utils.data.Dataset):
        """"""
        def __init__(self, X, Y, modality_first):
            """Initialize Dataset Class"""
            self.X, self.Y = X, Y
            self.modality_first = modality_first

        def __len__(self):
            """Get length of dataset."""
            return len(self.X)

        def __getitem__(self, index):
            """Get item from dataset."""
            # Data augmentation
            def _quantize(x, y):
                hi = torch.max(x)
                lo = torch.min(x)
                x = (x - lo) * 25 / (hi - lo)
                x = torch.round(x)
                x = x * (hi - lo) / 25 + lo
                return x, y

            x, y = _quantize(self.X[index], self.Y[index])

            if not modality_first:
                return x, y
            else:
                if len(x.shape) == 2:
                    x = x.permute([1, 0])
                    x = list(x)
                    x.append(y)
                    return x
                else:
                    x = x.permute([0, 2, 1])
                    res = []
                    for data, label in zip(x, y):
                        data = list(data)
                        data.append(label)
                        res.append(data)
                    return res

    train_ds = _MyDataset(X[:val_split], Y[:val_split], modality_first)
    train_loader = torch.utils.data.DataLoader(
        train_ds, shuffle=train_shuffle, batch_size=batch_size)
    val_ds = _MyDataset(X[val_split:test_split],
                       Y[val_split:test_split], modality_first)
    val_loader = torch.utils.data.DataLoader(
        val_ds, shuffle=False, batch_size=batch_size, drop_last=False)
    test_loader = dict()
    test_loader['timeseries'] = []
    for noise_level in range(9):
        X_robust = copy.deepcopy(X[test_split:].cpu().numpy())
        X_robust = torch.tensor(add_timeseries_noise(
            X_robust, noise_level=noise_level/10), dtype=torch.float32)
        if cuda:
            X_robust = X_robust.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        test_ds = _MyDataset(X_robust, Y[test_split:], modality_first)
        test_loader['timeseries'].append(torch.utils.data.DataLoader(
            test_ds, shuffle=False, batch_size=batch_size, drop_last=False))
    print(len(test_loader))
    return train_loader, val_loader, test_loader


class Grouping(nn.Module):
    """Module to collate stock data."""
    
    def __init__(self, n_groups):
        """Instantiate grouper. n_groups determines the number of groups."""
        super().__init__()
        self.n_groups = n_groups

    def forward(self, x):
        """Apply grouper to input data.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            list: List of outputs
        """
        x = x.permute(2, 0, 1)

        n_modalities = len(x)
        out = []
        for i in range(self.n_groups):
            start_modality = n_modalities * i // self.n_groups
            end_modality = n_modalities * (i + 1) // self.n_groups
            sel = list(x[start_modality:end_modality])
            sel = torch.stack(sel, dim=len(sel[0].size()))
            out.append(sel)

        return out