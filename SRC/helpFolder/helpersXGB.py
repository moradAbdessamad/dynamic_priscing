import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns #type: ignore
import os #type: ignore
import json #type: ignore

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
import pickle #type: ignore
from datetime import datetime #type: ignore
import warnings #type: ignore
import traceback #type: ignore

import xgboost as xgb #type: ignore
from statsmodels.tsa.statespace.sarimax import SARIMAX #type: ignore
from statsmodels.tsa.stattools import adfuller #type: ignore
from datetime import timedelta #type: ignore
from SRC.helpFolder.helpersWeather import get_weather_dataframe_for_city_single_date_fast, get_weather_dataframe_for_city_single_date
import SRC

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_type = 'price'  
target_type_taux = 'taux'

metaDataPath = '../metaData/modelMetaData.json'
metaData =SRC.load_model_metadata() 
# ==================================================== XGBoost Helper ====================================================

def split_data_train_test(df, unit_id, test_size=0.2, target_type='price'):
    """
    Splits the DataFrame chronologically into training and testing sets.
    
    Args:
        df: DataFrame containing the data
        unit_id: Unit identifier (e.g., '166', '167')
        test_size: Proportion of data to use for testing
        target_type: Type of target ('price' or 'taux')
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

    price_col = f'total_price_{unit_id}'
    taux_col = f'taux_occupation_{unit_id}'
    
    if target_type.lower() == 'price':
        target_col = price_col
        feature_col = taux_col
    else:  
        target_col = taux_col
        feature_col = price_col
        
    if target_col not in df_copy.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    
    if feature_col not in df_copy.columns:
        print(f"Warning: Feature column '{feature_col}' not found in DataFrame.")

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

    print(f"Split complete. Test size: {test_size:.2f}")
    print(f"Target column: {target_col}")
    print(f"Training set range: {X_train.index.min()} to {X_train.index.max()} ({len(X_train)} samples)")
    print(f"Testing set range: {X_test.index.min()} to {X_test.index.max()} ({len(X_test)} samples)")

    return X_train, X_test, y_train, y_test, target_col, feature_col


def plot_predictions_over_time(model_results, df, unit_id, target_type=None):
    """
    Plots actual vs. predicted values over time with dates on the x-axis,
    using the indices from the model results for chronological alignment.
    
    Args:
        model_results: Dictionary containing model results
        df: Original DataFrame with data
        unit_id: Unit identifier (e.g., '166', '167')
        target_type: Optional override for target type ('price' or 'taux'). If None, will auto-detect.
    """
    y_train = model_results['y_train']
    y_test = model_results['y_test']
    y_train_pred = model_results['y_train_pred']
    y_test_pred = model_results['y_test_pred']
    
    price_col = f'total_price_{unit_id}'
    taux_col = f'taux_occupation_{unit_id}'
    
    if target_type == 'price':
        y_label = 'Price'
        data_type = 'Prices'
    elif target_type == 'taux':
        y_label = 'Occupancy Rate'
        data_type = 'Occupancy Rates'
    else:
        target_col_name = model_results.get('target_col', None)
        if not target_col_name:

            if hasattr(y_train, 'name') and y_train.name:
                target_col_name = y_train.name

            else:
                if price_col in df.columns:
                    target_col_name = price_col
                elif taux_col in df.columns:
                    target_col_name = taux_col
        
        if target_col_name and 'taux' in target_col_name:
            y_label = 'Occupancy Rate'
            data_type = 'Occupancy Rates'
        else:
            y_label = 'Price'
            data_type = 'Prices'
    
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

    axes[0].plot(train_df['date'], train_df['actual'], 'b-', label=f'Actual {y_label}', alpha=0.7)
    axes[0].plot(train_df['date'], train_df['predicted'], 'r--', label=f'Predicted {y_label}')
    axes[0].set_title(f'Training Data: Unit {unit_id} - Actual vs Predicted {data_type}')
    axes[0].set_ylabel(y_label)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(test_df['date'], test_df['actual'], 'b-', label=f'Actual {y_label}', alpha=0.7)
    axes[1].plot(test_df['date'], test_df['predicted'], 'r--', label=f'Predicted {y_label}')
    axes[1].set_title(f'Test Data: Unit {unit_id} - Actual vs Predicted {data_type}')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel(y_label)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.autofmt_xdate() 

    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(model, df, unit_id, target_type=None):
    """
    Plots the actual vs. predicted values for the entire dataset using a trained model.
    Can handle both price and occupancy rate (taux) predictions.
    """
    df_copy = df.copy()

    if df_copy.index.name == 'date':
        df_copy = df_copy.sort_index()
    elif 'date' in df_copy.columns:
        df_copy = df_copy.sort_values('date') 
    else:
        print("Warning: DataFrame must have a 'date' index or column for plotting.")
        df_copy = df_copy.sort_index()

    price_col = f'total_price_{unit_id}'
    taux_col = f'taux_occupation_{unit_id}'
    
    if target_type == 'price':
        target_col = price_col
        non_target_col = taux_col
        y_label = 'Price'
        title_prefix = 'Prices'
    elif target_type == 'taux':
        target_col = taux_col
        non_target_col = price_col
        y_label = 'Occupancy Rate'
        title_prefix = 'Occupancy Rates'

    else:
        if price_col not in df_copy.columns:
            print(f"Error: Neither price nor taux column found for unit {unit_id}.")
            return
            
        test_row = df_copy.iloc[[0]]
        test_features = test_row.drop([price_col, taux_col], errors='ignore')
        if 'date' in test_features.columns:
            test_features = test_features.drop('date', axis=1)
            
        try:
            model.predict(test_features.drop([price_col], errors='ignore'))
            target_col = price_col
            non_target_col = taux_col
            y_label = 'Price'
            title_prefix = 'Prices'
        except:
            target_col = taux_col
            non_target_col = price_col
            y_label = 'Occupancy Rate'
            title_prefix = 'Occupancy Rates'
    
    if target_col not in df_copy.columns:
        print(f"Error: Target column '{target_col}' not found in DataFrame.")
        return

    features = df_copy.drop([target_col], axis=1, errors='ignore')
    if 'date' in features.columns:
        features = features.drop('date', axis=1, errors='ignore') 

    print(f"Columns used for prediction: {features.columns.tolist()}")
    print(f"Target column: {target_col}")

    predictions = model.predict(features)

    plot_df = pd.DataFrame({
        'Actual': df_copy[target_col],
        'Predicted': predictions
    })
    plot_df.index = df_copy.index if df_copy.index.name == 'date' else df_copy['date']

    plt.figure(figsize=(15, 7))
    plt.plot(plot_df.index, plot_df['Actual'], 'b-', label=f'Actual {y_label}', alpha=0.7)
    plt.plot(plot_df.index, plot_df['Predicted'], 'r--', label=f'Predicted {y_label}')

    plt.title(f'Unit {unit_id}: Actual vs. Predicted {title_prefix} (Full Dataset)')
    plt.xlabel('Date')
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


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


def run_xgboost_pipeline(df_unit, unit_id, test_size=0.2,
                        n_estimators=1000, learning_rate=0.01, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8, target_type='price'):
    """
    Runs the complete XGBoost training, evaluation, and plotting pipeline
    for a specific unit (simplified without additional parameters).
    
    Args:
        df_unit: DataFrame containing the unit data
        unit_id: Unit identifier (e.g., '166', '167')
        test_size: Proportion of data to use for testing
        n_estimators: Number of gradient boosted trees
        learning_rate: Boosting learning rate
        max_depth: Maximum depth of trees
        subsample: Subsample ratio of training instances
        colsample_bytree: Subsample ratio of columns when constructing each tree
        target_type: Type of target ('price' or 'taux')
    
    Returns:
        Dictionary containing model results or None if an error occurs
    """
    print(f"--- Starting XGBoost Pipeline for Unit {unit_id} ---")
    print(f"Target type: {target_type}")

    print(f"\n[Step 1/3] Splitting data with test_size={test_size}...")
    try:
        X_train, X_test, y_train, y_test, target_col, feature_col = split_data_train_test(
            df=df_unit, unit_id=unit_id, test_size=test_size, target_type=target_type
        )

    except ValueError as e:
        print(f"Error during data splitting: {e}")
        return None

    except Exception as e:
         print(f"An unexpected error occurred during data splitting: {e}")
         traceback.print_exc()
         return None

    print("\n[Step 2/3] Training and evaluating XGBoost model...")

    try:
        model_results = train_evaluate_xgboost_model(
            X_train, X_test, y_train, y_test, target_col, n_estimators, learning_rate,
            max_depth, subsample, colsample_bytree
        )
        
        model_results['target_type'] = target_type
        model_results['target_col'] = target_col
        model_results['feature_col'] = feature_col

    except Exception as e:
         print(f"An unexpected error occurred during training/evaluation: {e}")
         traceback.print_exc()
         return None

    print("\n[Step 3/3] Plotting results...")
    
    try:
        print("Plotting predictions on train/test split...")
        plot_predictions_over_time(model_results, df_unit, unit_id, target_type=target_type)

        print("Plotting predictions on the full unit DataFrame...")
        plot_actual_vs_predicted(model_results['model'], df_unit, unit_id, target_type=target_type)

    except KeyError as e:
        print(f"Error during plotting: Missing key in model_results - {e}")
        traceback.print_exc()
    
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")
        traceback.print_exc()

    print(f"\nPipeline finished for Unit {unit_id} (target: {target_type}).")
    print("--- End XGBoost Pipeline ---")

    return model_results




def predict_single_date_xgboost_model(model, target_date_str, unit_id, value_target, 
                                      target_type='price', weather_data_dict=None, city='Paris'):
    """
    Predicts either price or occupancy rate for a single future date using a trained XGBoost model,
    including weather features for the specified date. Uses pre-loaded weather data for fast performance.

    Args:
        model: The trained XGBoost model (e.g., xgb.XGBRegressor).
        target_date_str (str): The target date for prediction in 'YYYY-MM-DD' format.
        unit_id (str): The identifier for the unit (e.g., '166', '167').
        value_target (float): The value for the non-target feature (price or taux_occupation).
        target_type (str): Type of prediction ('price' or 'taux'). Defaults to 'price'.
        weather_data_dict (dict): Pre-loaded weather data dictionary with dates as keys. 
                                 If None, will fall back to API call.
        city (str): City name for weather data (only used if weather_data_dict is None).
    
    Returns:
        float or None: The predicted value, or None if prediction fails.
    """
    if model is None:
        print("Error: The XGBoost model provided is None.")
        return None

    try:
        target_date = pd.to_datetime(target_date_str)
    except ValueError:
        print(f"Error: Invalid target_date_str format: {target_date_str}. Please use 'YYYY-MM-DD'.")
        return None

    if target_type.lower() == 'price':
        feature_name = f'taux_occupation_{unit_id}'
        target_name = f'total_price_{unit_id}'
    else:  
        feature_name = f'total_price_{unit_id}'
        target_name = f'taux_occupation_{unit_id}'

    features = {
        'dayofweek': target_date.dayofweek,
        'month': target_date.month,
        'year': target_date.year,
        'dayofyear': target_date.dayofyear,
        'weekofyear': int(target_date.isocalendar()[1]),
        feature_name: value_target
    }

    try:
        if weather_data_dict is not None:
            weather_df = get_weather_dataframe_for_city_single_date_fast(target_date_str, weather_data_dict)

        else:
            weather_df = get_weather_dataframe_for_city_single_date(target_date_str, city)
        
        if not weather_df.empty:
            weather_row = weather_df.iloc[0]  
            
            features.update({
                'temperature_max': weather_row['temperature_max'],
                'temperature_min': weather_row['temperature_min'],
                'temperature_mean': weather_row['temperature_mean'],
                'precipitation': weather_row['precipitation'],
                'windspeed_max': weather_row['windspeed_max'],
                'temperature_range': weather_row['temperature_range']
            })
            
            weather_source = "pre-loaded data" if weather_data_dict is not None else f"API for {city}"
            print(f"Weather data successfully retrieved from {weather_source} for {target_date_str}")
        else:
            print(f"Warning: No weather data available for {target_date_str}. Using default values.")

            features.update({
                'temperature_max': 0.0,
                'temperature_min': 0.0,
                'temperature_mean': 0.0,
                'precipitation': 0.0,
                'windspeed_max': 0.0,
                'temperature_range': 0.0
            })
            
    except Exception as e:
        print(f"Error retrieving weather data: {e}")
        print("Using default weather values.")

        features.update({
            'temperature_max': 0.0,
            'temperature_min': 0.0,
            'temperature_mean': 0.0,
            'precipitation': 0.0,
            'windspeed_max': 0.0,
            'temperature_range': 0.0
        })

    if hasattr(model, 'feature_names_in_'):
        feature_order = list(model.feature_names_in_)
    else:

        feature_order = [
            'dayofweek', 'month', 'year', 'dayofyear', 'weekofyear', 
            feature_name,
            'temperature_max', 'temperature_min', 'temperature_mean', 
            'precipitation', 'windspeed_max', 'temperature_range'
        ]
        print(f"Warning: Model does not have 'feature_names_in_'. Using default feature order: {feature_order}")

    try:
        model_features = {col: features.get(col, 0.0) for col in feature_order}
        input_df = pd.DataFrame([model_features], columns=feature_order)
    
    except Exception as e:
        print(f"Error creating DataFrame for prediction: {e}")
        return None
        
    for col in feature_order:
        if col not in input_df.columns:
            print(f"Error: Missing expected feature '{col}' in the generated input for XGBoost.")
            print(f"Available columns: {list(input_df.columns)}")
            print(f"Input data: {features}")
            return None
            
    print(f"\nPredicting {target_name} with XGBoost for Unit {unit_id} on {target_date_str}")
    print(f"Features used:")
    print(input_df)

    try:
        prediction = model.predict(input_df)
        predicted_value = float(prediction[0])
        print(f"Predicted {target_name}: {predicted_value:.2f}")
        return predicted_value
    except Exception as e:
        print(f"Error during XGBoost prediction: {e}")
        
        if hasattr(model, 'feature_names_in_'):
            print(f"Model was trained/expects features: {model.feature_names_in_}")
        print(f"Input features provided: {input_df.columns.tolist()}")
        
        return None



    
def find_max_price_for_date_xgboost(
    target_date_str, 
    unit_id, 
    min_value=0.0, 
    max_value=95.0, 
    step=0.01,
    target_type='price',
    config_file_path=metaDataPath,
    modelsName="XGBoostModels",
    weather_data_dict=None
):
    """
    Finds the optimal input value that maximizes the predicted output for a single date
    using a trained XGBoost model. Automatically loads the appropriate model using choose_XGBoost_model.

    Args:
        target_date_str (str): The target date for prediction in 'YYYY-MM-DD' format.
        unit_id (str): The identifier for the unit (e.g., '166', '167').
        min_value (float): Minimum value to test.
        max_value (float): Maximum value to test.
        step (float): Step size for iterating through test values.
        target_type (str): Type of prediction ('price' or 'taux').
        config_file_path (str): Path to the model configuration file.
        modelsName (str): Name of the models section in the config file.
    """
    # Load the model using choose_XGBoost_model
    model = choose_XGBoost_model(
        target=target_type, 
        unit=unit_id, 
        modelsName=modelsName, 
        config_file_path=config_file_path
    )
    
    if model is None:
        print(f"Error: Could not load XGBoost model for target='{target_type}', unit='{unit_id}'")
        return None, None

    predicting_price = target_type.lower() == 'price'
    prediction_type = "price" if predicting_price else "taux"
    feature_type = "taux" if predicting_price else "price"
    
    best_input_value = None
    max_predicted_output = -float('inf')

    print(f"\nSearching for optimal {feature_type} for XGBoost model (unit {unit_id}) on {target_date_str}...")
    print(f"Target: Maximize {prediction_type}")
    print(f"Range: {min_value}-{max_value}, step: {step}")

    num_steps = int((max_value - min_value) / step) + 1
    test_values = np.linspace(min_value, max_value, num_steps)

    for test_value in tqdm(test_values, desc=f"Testing XGBoost {feature_type} values"):
        predicted_output = predict_single_date_xgboost_model(
            model=model,
            target_date_str=target_date_str,
            unit_id=unit_id,
            value_target=test_value,
            target_type=target_type,
            weather_data_dict=weather_data_dict
        )
        
        if predicted_output is not None:
            if predicted_output > max_predicted_output:
                max_predicted_output = predicted_output
                best_input_value = test_value
    
    if best_input_value is not None:
        if predicting_price:
            print(f"\nXGBoost Search complete. Optimal taux_occupation: {best_input_value:.2f} -> ")
            print(f"Max Price: {max_predicted_output:.2f}")
        else:
            print(f"\nXGBoost Search complete. Optimal price: {best_input_value:.2f} -> ")
            print(f"Max Occupancy Rate: {max_predicted_output:.2f}%")
        return best_input_value, max_predicted_output
    else:
        print(f"XGBoost Search complete. No valid predictions found for {prediction_type}.")
        return None, None

def get_dates(from_date, to_date):
    from datetime import datetime, timedelta
    
    start_date = datetime.strptime(from_date, "%Y-%m-%d")
    end_date = datetime.strptime(to_date, "%Y-%m-%d")
    
    dates = []
    current_date = start_date
    
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    
    return dates

def choose_XGBoost_model(target, unit, modelsName="XGBoostModels", config_file_path=metaDataPath):
    """
    Loads an XGBoost model based on the target and unit using a configuration file.

    Args:
        target (str): The target type ('taux' or 'price').
        unit (str): The unit ID ('166' or '167').
        config_file_path (str): Path to the JSON configuration file.
    """
    try:
        with open(config_file_path, 'r') as f:
            config = json.load(f)

        config = config[modelsName]    
        
        model_key = f"{target.lower()}_{unit}"
        
        if model_key in config:
            model_info = config[model_key]
            print(f"Loading model for target='{target}', unit='{unit}'...")
            print(f"Model name: {model_info['model_name']}")
            
            model = load_xgboost_model(
                folder_path=model_info['folder_path'],
                model_name=model_info['model_name']
            )
            print("Model loaded successfully.")
            return model
    
        else:
            print(f"Error: Model configuration for target='{target}', unit='{unit}' not found in {config_file_path}.")
            return None
    
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_file_path}.")
        return None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def find_max_price_for_date_range_xgboost(
    start_date_str,
    end_date_str,
    unit_id='166', 
    min_value=30.0,
    max_value=95.0,
    step=1,
    target_type='price',
    config_file_path=metaDataPath,
    modelsName="XGBoostModels",
    weather_data_dict=None,
):
    """
    Finds the maximum price or taux for a range of dates using the XGBoost model.

    Args:
        start_date_str (str): The start date in 'YYYY-MM-DD' format.
        end_date_str (str): The end date in 'YYYY-MM-DD' format.
        unit_id (str): The unit ID.
        min_value (float): The minimum input value to test.
        max_value (float): The maximum input value to test.
        step (float): The step for iterating through input values (passed to helper).
        target_type (str): The type of target ('price' or 'taux').
        config_file_path (str): Path to the JSON configuration file for model selection.
        modelsName (str): Key for XGBoost models in the config file.
    """
    all_dates = get_dates(start_date_str, end_date_str)
    results_for_range = {}

    for target_date_str in all_dates:
        try:
            best_input, max_output = find_max_price_for_date_xgboost(
                target_date_str=target_date_str,
                unit_id=unit_id,
                min_value=min_value,
                max_value=max_value,
                step=step,
                target_type=target_type,
                config_file_path=config_file_path,
                modelsName=modelsName,
                weather_data_dict=weather_data_dict
            )
            results_for_range[target_date_str] = {"best_input": best_input, "max_output": max_output}

        except Exception as e:
            results_for_range[target_date_str] = f"Error processing date {target_date_str}: {str(e)}"
            print(f"Error processing date {target_date_str}: {str(e)}")

    return results_for_range

def find_max_price_for_single_date_xgboost(
    target_date_str, 
    unit_id, 
    min_value=0.0, 
    max_value=95.0, 
    step=0.01,
    target_type='price',
    config_file_path=metaDataPath,
    modelsName="XGBoostModels",
    weather_data_dict=None,
):
    """
    Finds the optimal input value that maximizes the predicted output for a single date
    using a trained XGBoost model. Automatically loads the appropriate model using choose_XGBoost_model.

    Args:
        target_date_str (str): The target date for prediction in 'YYYY-MM-DD' format.
        unit_id (str): The identifier for the unit (e.g., '166', '167').
        min_value (float): Minimum value to test.
        max_value (float): Maximum value to test.
        step (float): Step size for iterating through test values.
        target_type (str): Type of prediction ('price' or 'taux').
        config_file_path (str): Path to the model configuration file.
        modelsName (str): Name of the models section in the config file.
    """

    model = choose_XGBoost_model(
        target=target_type, 
        unit=unit_id, 
        modelsName=modelsName, 
        config_file_path=config_file_path
    )
    
    if model is None:
        print(f"Error: Could not load XGBoost model for target='{target_type}', unit='{unit_id}'")
        return None, None

    predicting_price = target_type.lower() == 'price'
    prediction_type = "price" if predicting_price else "taux"
    feature_type = "taux" if predicting_price else "price"
    
    best_input_value = None
    max_predicted_output = -float('inf')

    print(f"\nSearching for optimal {feature_type} for XGBoost model (unit {unit_id}) on {target_date_str}...")
    print(f"Target: Maximize {prediction_type}")
    print(f"Range: {min_value}-{max_value}, step: {step}")

    num_steps = int((max_value - min_value) / step) + 1
    test_values = np.linspace(min_value, max_value, num_steps)

    for test_value in tqdm(test_values, desc=f"Testing XGBoost {feature_type} values"):
        predicted_output = predict_single_date_xgboost_model(
            model=model,
            target_date_str=target_date_str,
            unit_id=unit_id,
            value_target=test_value,
            target_type=target_type,
            weather_data_dict=weather_data_dict
        )
        
        if predicted_output is not None:
            if predicted_output > max_predicted_output:
                max_predicted_output = predicted_output
                best_input_value = test_value
    
    if best_input_value is not None:
        if predicting_price:
            print(f"\nXGBoost Search complete. Optimal taux_occupation: {best_input_value:.2f} -> ")
            print(f"Max Price: {max_predicted_output:.2f}")
        else:
            print(f"\nXGBoost Search complete. Optimal price: {best_input_value:.2f} -> ")
            print(f"Max Occupancy Rate: {max_predicted_output:.2f}%")
        return best_input_value, max_predicted_output
    else:
        print(f"XGBoost Search complete. No valid predictions found for {prediction_type}.")
        return None, None