import os
import pandas as pd #type: ignore
import json
import matplotlib.pyplot as plt #type: ignore
import numpy as np #type: ignore
from .helpersXGB import * #type: ignore
from .helpersLR import * #type: ignore
from .helpersLSTM import * #type: ignore

metaDataPath = '../metaData/modelMetaData.json'

# ======================================================== The Helpers for data perperation ========================================================
def append_to_json_file(data, file):
    if os.path.exists(file):
        return 
    
    with open(file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    print(f"Data saved to {file}")


def load_json_data(file_path):
    """
    Load JSON data from a given file path.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def preprocess_for_forecasting(df):
    """
    Preprocesses the DataFrame for time series forecasting.
    """

    df_processed = df.copy()

    df_processed['date'] = pd.to_datetime(df_processed['date'])

    df_processed.set_index('date', inplace=True)

    df_processed.sort_index(inplace=True)

    numeric_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].interpolate(method='time')

    df_processed['dayofweek'] = df_processed.index.dayofweek
    df_processed['month'] = df_processed.index.month
    df_processed['year'] = df_processed.index.year
    df_processed['dayofyear'] = df_processed.index.dayofyear
    df_processed['weekofyear'] = df_processed.index.isocalendar().week.astype(int)

    for col in ['total_price_167', 'taux_occupation_167', 'total_price_166', 'taux_occupation_166']:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce') 

    if df_processed.isnull().sum().sum() > 0:
        print("Warning: NaNs still present after interpolation. Consider further handling (e.g., ffill/bfill).")
        df_processed.ffill(inplace=True)

    return df_processed


def select_date_range(df, start_date, end_date): 
    """
    Selects a date range from the DataFrame.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    df_range = df.loc[start:end]
    
    return df_range


def split_df(df, units=['167', '166']):
    """
    Splits the DataFrame based on the units.
    """
    common_cols = ['dayofweek', 'month', 'year', 'dayofyear', 'weekofyear']
    
    if 'date' not in df.columns and df.index.name == 'date':
        df = df.reset_index()
    
    if 'date' in df.columns:
        common_cols = ['date'] + common_cols
    
    df_167 = df[common_cols + [f'total_price_{units[0]}', f'taux_occupation_{units[0]}']]
    
    df_166 = df[common_cols + [f'total_price_{units[1]}', f'taux_occupation_{units[1]}']]

    df_166.set_index('date', inplace=True)
    df_167.set_index('date', inplace=True)
    
    return df_167, df_166

def prepare_and_plot_data(csv_path, start_date, end_date, units=['167', '166']):
    """
    Prepares data from a CSV file, performs preprocessing, selects a date range,
    splits the data by units, and plots the specified metrics.
    """
    df = pd.read_csv(csv_path)
    df = preprocess_for_forecasting(df)
    
    df_range = select_date_range(df, start_date, end_date)
    
    plt.figure(figsize=(20, 15))
    
    plt.subplot(2, 2, 1)
    df_range.plot(figsize=(15, 5), title=f'Total Price Over Time - Unit {units[0]}', y=f'total_price_{units[0]}', ax=plt.gca())
    
    plt.subplot(2, 2, 2)
    df_range.plot(figsize=(15, 5), title=f'Total Price Over Time - Unit {units[1]}', y=f'total_price_{units[1]}', ax=plt.gca())
    
    plt.subplot(2, 2, 3)
    df_range.plot(figsize=(15, 5), title=f'TAUX Over Time - Unit {units[0]}', y=f'taux_occupation_{units[0]}', ax=plt.gca())
    
    plt.subplot(2, 2, 4)
    df_range.plot(figsize=(15, 5), title=f'TAUX Over Time - Unit {units[1]}', y=f'taux_occupation_{units[1]}', ax=plt.gca())
    
    plt.tight_layout()
    plt.show()
    
    df_unit1, df_unit2 = split_df(df_range, units)
    
    return df_unit1, df_unit2

def prepare_the_data(csv_path, start_date, end_date, units=['167', '166']):
    """
    Prepares data from a CSV file, performs preprocessing, selects a date range,
    splits the data by units, and plots the specified metrics.
    """
    df = pd.read_csv(csv_path)
    df = preprocess_for_forecasting(df)
    
    df_range = select_date_range(df, start_date, end_date)
    
    df_unit1, df_unit2 = split_df(df_range, units)
    
    return df_unit1, df_unit2


def concatenate_on_index(df1, df2):
    """
    Concatenate two DataFrames along the columns (axis=1), aligning them by index.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.
    """
    return pd.concat([df1, df2], axis=1)
