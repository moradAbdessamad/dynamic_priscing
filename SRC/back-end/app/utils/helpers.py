import os
import requests # type: ignore
from openai import OpenAI # type: ignore
from dotenv import load_dotenv # type: ignore
from flask import Blueprint, request, jsonify # type: ignore
from pathlib import Path
from datetime import datetime
import seaborn as sns #type: ignore
import os
import json

from sklearn.linear_model import LinearRegression #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score #type: ignore

import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.optim as optim #type: ignore
from torch.utils.data import DataLoader, TensorDataset #type: ignore
from sklearn.preprocessing import MinMaxScaler #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.metrics import mean_squared_error, r2_score #type: ignore
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
from tqdm import tqdm #type: ignore
import pandas as pd #type: ignore
import pickle
from datetime import datetime
import warnings
import traceback

import xgboost as xgb #type: ignore
from statsmodels.tsa.statespace.sarimax import SARIMAX #type: ignore
from statsmodels.tsa.stattools import adfuller #type: ignore
from datetime import timedelta
import re #type: ignore
import dateparser #type: ignore

warnings.filterwarnings('ignore')

load_dotenv()
access_token = os.getenv("META_ACCESS_TOKEN")
Openai_api_key = os.getenv("OPENAI_API_KEY")

phone_number_id = os.getenv("RECIPIENT_PHONE")
recipient_phone = os.getenv("PHONE_NUMBER_ID")


def send_whatsapp_session_message(phone_number_id=phone_number_id, recipient_phone=recipient_phone, 
                                 access_token=access_token, message_text="Hello! This is a direct message."):
    """
    Send a direct message to a WhatsApp user using the WhatsApp Business API.
    This function sends a text message to a specified recipient using the WhatsApp Business API.
    """
    url = f"https://graph.facebook.com/v17.0/{phone_number_id}/messages"
    
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_phone,
        "type": "text",
        "text": {
            "body": message_text
        }
    }

    params = {
        "access_token": access_token
    }

    response = requests.post(url, headers=headers, params=params, json=payload)
    print("Status:", response.status_code)
    print("Response:", response.text)

    return response

def load_and_format_prompt(file_path, **kwargs):
    """
    Loads a prompt template from a file and replaces placeholders with provided values.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            template = file.read()
            
        formatted_prompt = template
        for key, value in kwargs.items():
            placeholder = f"{{{key}}}"
            formatted_prompt = formatted_prompt.replace(placeholder, str(value))
            
        return formatted_prompt
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt template file not found at: {file_path}")
    
    except Exception as e:
        raise Exception(f"Error processing prompt template: {str(e)}")


def send_openai_request(user_prompt,
                        Openai_api_key=Openai_api_key, 
                        system_prompt=None, 
                        temperature=0.3, 
                        model="gpt-4.1", 
                        stream=False):
    """
    Sends a request to the OpenAI API with the given prompts and parameters.
    """
    client = OpenAI(api_key=Openai_api_key)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})
    
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=stream
    )
    
    response_content = ""
    if stream:
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                content_piece = chunk.choices[0].delta.content
                print(content_piece, end="")
                response_content += content_piece
    else:
        response_content = completion.choices[0].message.content
        
    return response_content  


def train_evaluate_xgboost_model(X_train, X_test, y_train, y_test, target_col,
                                 n_estimators=1000, learning_rate=0.01, max_depth=5,
                                 subsample=0.8, colsample_bytree=0.8):
    """
    Trains an XGBoost Regressor model, evaluates it, and returns results.
    (Simplified version with early stopping and additional params removed)
    """
    print(f"\nTraining XGBoost model for target: {target_col}")
    print(f"Hyperparameters: n_estimators={n_estimators}, lr={learning_rate}, max_depth={max_depth}")
    print(f"Features used (X): {X_train.columns.tolist()}")
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        n_jobs=-1
    )

    print("\nStarting Training...")
    model.fit(X_train, y_train)
    print("Training Finished.")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\nModel Evaluation Metrics:")
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Train MAE: {train_mae:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")

    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'metrics': {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        },
        'feature_importance': feature_importance
    }

def split_data_train_test(df, unit_id, test_size=0.2):
    """
    Splits the DataFrame chronologically into training and testing sets.
    """
    df_copy = df.copy()

    if df_copy.index.name == 'date':
        df_copy = df_copy.sort_index()
    
    elif 'date' in df_copy.columns:
        df_copy = df_copy.sort_values('date').reset_index(drop=True)
    
    else:
        print("Warning: No 'date' index or column found. Assuming index is chronological.")
        df_copy = df_copy.sort_index()
        if df_copy.index.name == 'date':
             df_copy = df_copy.reset_index()


    target_col = f'total_price_{unit_id}'
    if target_col not in df_copy.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    X = df_copy.drop([target_col], axis=1, errors='ignore')
    y = df_copy[target_col]

    split_idx = int(len(df_copy) * (1 - test_size))
    if split_idx == 0 or split_idx >= len(df_copy):
        raise ValueError("test_size results in an invalid split index.")

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    date_col_in_features = 'date' in X_train.columns
    if date_col_in_features:
        X_train = X_train.drop('date', axis=1, errors='ignore')
        X_test = X_test.drop('date', axis=1, errors='ignore')

    print(f"split complete. Test size: {test_size:.2f}")
    print(f"Training set range: {X_train.index.min()} to {X_train.index.max()} ({len(X_train)} samples)")
    print(f"Testing set range: {X_test.index.min()} to {X_test.index.max()} ({len(X_test)} samples)")

    return X_train, X_test, y_train, y_test, target_col


def plot_predictions_over_time(model_results, df, unit_id):
    """
    Plots actual vs. predicted values over time with dates on the x-axis,
    using the indices from the model results for chronological alignment.
    """

    y_train = model_results['y_train']
    y_test = model_results['y_test']
    y_train_pred = model_results['y_train_pred']
    y_test_pred = model_results['y_test_pred']

    train_df = pd.DataFrame({
        'date': y_train.index,  
        'actual': y_train.values,
        'predicted': y_train_pred
    }).sort_values('date')

    test_df = pd.DataFrame({
        'date': y_test.index,   
        'actual': y_test.values,
        'predicted': y_test_pred
    }).sort_values('date')

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(train_df['date'], train_df['actual'], 'b-', label='Actual Price', alpha=0.7)
    axes[0].plot(train_df['date'], train_df['predicted'], 'r--', label='Predicted Price')
    axes[0].set_title(f'Training Data: Unit {unit_id} - Actual vs Predicted Prices')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(test_df['date'], test_df['actual'], 'b-', label='Actual Price', alpha=0.7)
    axes[1].plot(test_df['date'], test_df['predicted'], 'r--', label='Predicted Price')
    axes[1].set_title(f'Test Data: Unit {unit_id} - Actual vs Predicted Prices')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Price')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.autofmt_xdate() 

    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(model, df, unit_id):
    """
    Plots the actual vs. predicted values for the entire dataset using a trained model.
    """
    df_copy = df.copy()

    if df_copy.index.name == 'date':
        df_copy = df_copy.sort_index()
    elif 'date' in df_copy.columns:
        df_copy = df_copy.sort_values('date') 
    else:
        print("Warning: DataFrame must have a 'date' index or column for plotting.")
        df_copy = df_copy.sort_index()


    target_col = f'total_price_{unit_id}'
    if target_col not in df_copy.columns:
        print(f"Error: Target column '{target_col}' not found in DataFrame.")
        return

    features = df_copy.drop([target_col], axis=1, errors='ignore')
    if 'date' in features.columns:
         features = features.drop('date', axis=1, errors='ignore') 

    print(f"Columns used for prediction: {features.columns.tolist()}")

    predictions = model.predict(features)

    plot_df = pd.DataFrame({
        'Actual': df_copy[target_col],
        'Predicted': predictions
    })
    plot_df.index = df_copy.index if df_copy.index.name == 'date' else df_copy['date']


    plt.figure(figsize=(15, 7))
    plt.plot(plot_df.index, plot_df['Actual'], 'b-', label='Actual Price', alpha=0.7)
    plt.plot(plot_df.index, plot_df['Predicted'], 'r--', label='Predicted Price')

    plt.title(f'Unit {unit_id}: Actual vs. Predicted Prices (Full Dataset)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_xgboost_model(xgb_results, folder_path, model_name):
    """
    Save only the XGBoost model using pickle to a specified folder path.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created directory: {folder_path}")
    
    model_file_path = os.path.join(folder_path, f"{model_name}.pkl")
    
    model = xgb_results.get('model')
    if model is None:
        print("Error: No model found in the xgb_results dictionary")
        return None
    
    try:
        with open(model_file_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model successfully saved to {model_file_path}")
        return model_file_path
    except Exception as e:
        print(f"Error saving model: {e}")
        return None
    

def load_xgboost_model(folder_path, model_name):
    """
    Load a pickled XGBoost model from a specified folder path.

    Args:
        folder_path (str): The directory where the model is saved.
        model_name (str): The name of the model file (without .pkl extension).
    """
    file_path = os.path.join(folder_path, f"{model_name}.pkl")
    
    if not os.path.exists(file_path):
        print(f"Error: Model file not found at {file_path}")
        return None
    
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"XGBoost model successfully loaded from {file_path}")
        return model
    except Exception as e:
        print(f"Error loading XGBoost model: {e}")
        return None


def predict_single_date_xgboost_model(model, target_date_str, unit_id, taux_occupation_target):
    """
    Predicts the total_price for a single future date using a trained XGBoost model.

    Args:
        model: The trained XGBoost model (e.g., xgb.XGBRegressor).
        target_date_str (str): The target date for prediction in 'YYYY-MM-DD' format.
        unit_id (str): The identifier for the unit (e.g., '166', '167').
        taux_occupation_target (float): The expected taux_occupation for the target date.
    """
    if model is None:
        print("Error: The XGBoost model provided is None.")
        return None

    try:
        target_date = pd.to_datetime(target_date_str)
    except ValueError:
        print(f"Error: Invalid target_date_str format: {target_date_str}. Please use 'YYYY-MM-DD'.")
        return None

    features = {
        'dayofweek': target_date.dayofweek,
        'month': target_date.month,
        'year': target_date.year,
        'dayofyear': target_date.dayofyear,
        'weekofyear': int(target_date.isocalendar()[1]),  
        f'taux_occupation_{unit_id}': taux_occupation_target,
    }

    if hasattr(model, 'feature_names_in_'):
        feature_order = model.feature_names_in_
    else:
        feature_order = ['dayofweek', 'month', 'year', 'dayofyear', 'weekofyear', f'taux_occupation_{unit_id}']
        print(f"Warning: Model does not have 'feature_names_in_'. Using default feature order: {feature_order}")


    try:
        input_df = pd.DataFrame([features], columns=feature_order)
    except Exception as e:
        print(f"Error creating DataFrame for prediction: {e}")
        return None
        
    for col in feature_order:
        if col not in input_df.columns:
            print(f"Error: Missing expected feature '{col}' in the generated input for XGBoost.")
            print(f"Available columns: {list(input_df.columns)}")
            print(f"Input data: {features}")
            return None
            
    print(f"\nPredicting with XGBoost for Unit {unit_id} on {target_date_str} with features:")
    print(input_df)

    try:
        prediction = model.predict(input_df)
        return float(prediction[0])  
    except Exception as e:
        print(f"Error during XGBoost prediction: {e}")
        if hasattr(model, 'feature_names_in_'):
            print(f"Model was trained/expects features: {model.feature_names_in_}")
        print(f"Input features provided: {input_df.columns.tolist()}")
        return None


def find_max_price_for_date_xgboost(model, target_date_str, unit_id, min_taux=0.0, max_taux=95.0, step=0.01):
    """
    Finds the taux_occupation that maximizes the predicted price for a single date
    using a trained XGBoost model by iterating through a range of taux_occupation values.

    Args:
        model: The trained XGBoost model (e.g., xgb.XGBRegressor).
        target_date_str (str): The target date for prediction in 'YYYY-MM-DD' format.
        unit_id (str): The identifier for the unit (e.g., '166', '167').
        min_taux (float): Minimum taux_occupation to test.
        max_taux (float): Maximum taux_occupation to test.
        step (float): Step for iterating taux_occupation.
    """
    best_taux_occupation = None
    max_predicted_price = -float('inf') 

    if model is None:
        print("Error: The XGBoost model provided is None for finding max price.")
        return None, None

    print(f"\nSearching for optimal taux_occupation for XGBoost model (unit {unit_id}) on {target_date_str}...")
    print(f"Range: {min_taux}-{max_taux}, step: {step}")

    num_steps = int((max_taux - min_taux) / step) + 1
    taux_values = np.linspace(min_taux, max_taux, num_steps)

    for taux in tqdm(taux_values, desc="Testing XGBoost taux_occupation values"):
        predicted_price = predict_single_date_xgboost_model(
            model=model,
            target_date_str=target_date_str,
            unit_id=unit_id,
            taux_occupation_target=taux
        )
        
        if predicted_price is not None:
            if predicted_price > max_predicted_price:
                max_predicted_price = predicted_price
                best_taux_occupation = taux
    
    if best_taux_occupation is not None:
        print(f"\nXGBoost Search complete. Optimal taux_occupation: {best_taux_occupation:.2f} -> Max Price: {max_predicted_price:.2f}")
        return best_taux_occupation, max_predicted_price
    else:
        print("XGBoost Search complete. No valid predictions found.")
        return None, None


def run_xgboost_pipeline(df_unit, unit_id, test_size=0.2,
                        n_estimators=1000, learning_rate=0.01, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8):
    """
    Runs the complete XGBoost training, evaluation, and plotting pipeline
    for a specific unit (simplified without additional parameters).
    """
    print(f"--- Starting XGBoost Pipeline for Unit {unit_id} ---")

    print(f"\n[Step 1/3] Splitting data with test_size={test_size}...")
    try:
        X_train, X_test, y_train, y_test, target_col = split_data_train_test(
            df=df_unit, unit_id=unit_id, test_size=test_size
        )

    except ValueError as e:
        print(f"Error during data splitting: {e}")
        return None

    except Exception as e:
         print(f"An unexpected error occurred during data splitting: {e}")
         return None

    print("\n[Step 2/3] Training and evaluating XGBoost model...")

    try:
        model_results = train_evaluate_xgboost_model(
            X_train, X_test, y_train, y_test, target_col
        )

    except Exception as e:
         print(f"An unexpected error occurred during training/evaluation: {e}")
         return None

    print("\n[Step 3/3] Plotting results...")
    
    try:
        print("Plotting predictions on train/test split...")
        plot_predictions_over_time(model_results, df_unit, unit_id)

        print("Plotting predictions on the full unit DataFrame...")
        plot_actual_vs_predicted(model_results['model'], df_unit, unit_id)

    except KeyError as e:
        print(f"Error during plotting: Missing key in model_results - {e}")
    
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")

    print(f"\nPipeline finished for Unit {unit_id}.")
    print("--- End XGBoost Pipeline ---")

    return model_results


def extract_date_unit_from_text(user_input):
    """
    Extracts the date from the user input sting using regex and dateparser.
    """
    date_match = re.search(r'["\']?(\d{4}-\d{2}-\d{2})["\']?', user_input)
    date = None
    if date_match:
        parsed_date = dateparser.parse(date_match.group(1))
        if parsed_date:
            date = parsed_date.date()

    unit_match = re.search(r'unit\s*=\s*(\d+)', user_input, re.IGNORECASE)
    unit_id = int(unit_match.group(1)) if unit_match else None

    return date, unit_id