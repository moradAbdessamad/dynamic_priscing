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
import SRC
from SRC.helpFolder.helpersWeather import get_weather_dataframe_for_city_single_date, get_weather_dataframe_for_city_single_date_fast

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_type = 'price'  
target_type_taux = 'taux'

metaDataPath = '../metaData/modelMetaData.json'
metaData = SRC.load_model_metadata()
## =================================================== LSTM Helper Functions ================================================== ##
def create_sequences(input_data, target_data, sequence_length):
    """
    Creates sequences for LSTM model training.
    """
    sequences = []
    targets = []
    data_len = len(input_data)
    for i in range(data_len - sequence_length):
        seq_end = i + sequence_length
        sequences.append(input_data[i:seq_end])
        targets.append(target_data[seq_end]) 
    return np.array(sequences), np.array(targets)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=80, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )

        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1) 

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x) 

        # We only need the output of the last time step
        last_time_step_out = lstm_out[:, -1, :] 

        out = self.batch_norm(last_time_step_out)
        out = self.dropout(out)
        out = self.fc(out) # Shape: (batch_size, 1)
        return out


def train_evaluate_lstm_model(X_train, X_test, y_train, y_test, target_col,
                              sequence_length=10, batch_size=32, epochs=50,
                              learning_rate=0.001, hidden_dim=80, num_layers=2, dropout=0.2,
                              device='cuda' if torch.cuda.is_available() else 'cpu',
                              target_type='price'):
    """
    Trains and evaluates an LSTM model using pre-split data.
    
    Args:
        X_train, X_test: Training and testing feature sets
        y_train, y_test: Training and testing target values
        target_col: Name of the target column
        sequence_length: Length of input sequences for LSTM
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        hidden_dim: Hidden dimension size for LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        device: Computing device ('cpu' or 'cuda')
        target_type: Type of prediction ('price' or 'taux')
    """
    
    prediction_type = "Price" if target_type.lower() == 'price' else "Occupancy Rate"
    
    print(f"\nTraining LSTM model for target: {target_col} (predicting {prediction_type})")
    print(f"Hyperparameters: seq_len={sequence_length}, batch={batch_size}, epochs={epochs}, lr={learning_rate}")
    print(f"Features used (X): {X_train.columns.tolist()}")

    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)

    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        raise ValueError(f"Not enough data to create sequences with length {sequence_length}. "
                         f"Train length: {len(X_train)}, Test length: {len(X_test)}")

    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

    input_dim = X_train.shape[1] 
    model = LSTMModel(input_dim, hidden_dim, num_layers, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []

    print("\nStarting Training...")
    for epoch in range(epochs):
        model.train() 
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_X, batch_y in progress_bar:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_epoch_loss)

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1: 
             print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.6f}")

    print("Training Finished.")

    model.eval() 
    train_predictions_scaled = []
    train_actuals_scaled = []
    test_predictions_scaled = []
    test_actuals_scaled = []

    with torch.no_grad():
        for batch_X, batch_y in DataLoader(train_dataset, batch_size=batch_size, shuffle=False):
             outputs = model(batch_X)
             train_predictions_scaled.extend(outputs.cpu().numpy())
             train_actuals_scaled.extend(batch_y.cpu().numpy())

        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            test_predictions_scaled.extend(outputs.cpu().numpy())
            test_actuals_scaled.extend(batch_y.cpu().numpy())

    train_predictions_scaled = np.array(train_predictions_scaled)
    train_actuals_scaled = np.array(train_actuals_scaled)
    test_predictions_scaled = np.array(test_predictions_scaled)
    test_actuals_scaled = np.array(test_actuals_scaled)

    y_train_pred = target_scaler.inverse_transform(train_predictions_scaled)
    y_train_actual = target_scaler.inverse_transform(train_actuals_scaled)
    y_test_pred = target_scaler.inverse_transform(test_predictions_scaled)
    y_test_actual = target_scaler.inverse_transform(test_actuals_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    train_mae = mean_absolute_error(y_train_actual, y_train_pred)
    test_mae = mean_absolute_error(y_test_actual, y_test_pred)
    train_r2 = r2_score(y_train_actual, y_train_pred)
    test_r2 = r2_score(y_test_actual, y_test_pred)

    print(f"\nModel Evaluation Metrics for {prediction_type} prediction:")
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Train MAE: {train_mae:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")

    y_train_matched = y_train[sequence_length:]
    y_test_matched = y_test[sequence_length:]

    return {
        'model': model,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'sequence_length': sequence_length,
        'X_train_orig': X_train,
        'X_test_orig': X_test,
        'y_train_orig': y_train,
        'y_test_orig': y_test,
        'y_train_actual': y_train_actual.flatten(), 
        'y_test_actual': y_test_actual.flatten(),
        'y_train_pred': y_train_pred.flatten(),
        'y_test_pred': y_test_pred.flatten(),
        'metrics': {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        },
        'y_train_indices': y_train_matched.index,
        'y_test_indices': y_test_matched.index,
        'loss_history': loss_history,
        'target_type': target_type,
        'target_col': target_col
    }

def plot_lstm_loss(lstm_results, unit_id):
    """
    Plots the training loss per epoch for the LSTM model.
    """
    if 'loss_history' not in lstm_results:
        print("Error: 'loss_history' not found in lstm_results. "
              "Ensure the training function stores and returns it.")
        return

    loss_history = lstm_results['loss_history']
    epochs = range(1, len(loss_history) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_history, 'bo-', label='Training Loss (MSE)')
    plt.title(f'Unit {unit_id}: LSTM Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


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


def plot_lstm_predictions_over_time(lstm_results, unit_id):
    """
    Plots actual vs. predicted values over time for LSTM model results.
    Supports both price and occupancy rate predictions.
    """
    required_keys = ['y_train_actual', 'y_train_pred', 'y_train_indices',
                     'y_test_actual', 'y_test_pred', 'y_test_indices']
    
    if not all(key in lstm_results for key in required_keys):
        print("Error: lstm_results dictionary is missing required keys for plotting.")
        print(f"Required keys: {required_keys}")
        return

    target_type = lstm_results.get('target_type', 'price')
    
    if target_type.lower() == 'price':
        y_label = 'Price'
        title_suffix = 'Prices'
    else:  
        y_label = 'Occupancy Rate'
        title_suffix = 'Occupancy Rates'
    
    y_train_actual = lstm_results['y_train_actual']
    y_train_pred = lstm_results['y_train_pred']
    train_indices = lstm_results['y_train_indices']

    y_test_actual = lstm_results['y_test_actual']
    y_test_pred = lstm_results['y_test_pred']
    test_indices = lstm_results['y_test_indices']

    if len(y_train_actual) != len(train_indices) or len(y_train_pred) != len(train_indices):
        print("Warning: Length mismatch between training predictions/actuals and indices.")
        min_len_train = min(len(y_train_actual), len(y_train_pred), len(train_indices))
        y_train_actual = y_train_actual[:min_len_train]
        y_train_pred = y_train_pred[:min_len_train]
        train_indices = train_indices[:min_len_train]

    if len(y_test_actual) != len(test_indices) or len(y_test_pred) != len(test_indices):
        print("Warning: Length mismatch between testing predictions/actuals and indices.")
        min_len_test = min(len(y_test_actual), len(y_test_pred), len(test_indices))
        y_test_actual = y_test_actual[:min_len_test]
        y_test_pred = y_test_pred[:min_len_test]
        test_indices = test_indices[:min_len_test]

    train_plot_df = pd.DataFrame({
        'Actual': y_train_actual,
        'Predicted': y_train_pred
    }, index=train_indices)

    test_plot_df = pd.DataFrame({
        'Actual': y_test_actual,
        'Predicted': y_test_pred
    }, index=test_indices)

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(train_plot_df.index, train_plot_df['Actual'], 'b-', label=f'Actual {y_label}', alpha=0.7)
    axes[0].plot(train_plot_df.index, train_plot_df['Predicted'], 'r--', label=f'Predicted {y_label}')
    axes[0].set_title(f'Training Data: Unit {unit_id} - LSTM Actual vs Predicted {title_suffix}')
    axes[0].set_ylabel(y_label)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(test_plot_df.index, test_plot_df['Actual'], 'b-', label=f'Actual {y_label}', alpha=0.7)
    axes[1].plot(test_plot_df.index, test_plot_df['Predicted'], 'r--', label=f'Predicted {y_label}')
    axes[1].set_title(f'Test Data: Unit {unit_id} - LSTM Actual vs Predicted {title_suffix}')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel(y_label)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def plot_lstm_predictions_on_dataframe(df, unit_id, model, feature_scaler, target_scaler, sequence_length, target_col, device):
    """
    Uses a trained LSTM model to predict on a given DataFrame and plots actual vs. predicted values.
    Supports both price and occupancy rate predictions.
    """
    df_copy = df.copy()
    
    target_type = 'price'
    if 'target_type' in dir(model):
        target_type = model.target_type
    elif target_col.startswith('taux'):
        target_type = 'taux'
    elif target_col.startswith('total_price'):
        target_type = 'price'
    
    if target_type.lower() == 'price':
        y_label = 'Price'
        title_suffix = 'Prices'
    else:  # 'taux'
        y_label = 'Occupancy Rate'
        title_suffix = 'Occupancy Rates'
        
    if target_col not in df_copy.columns:
        print(f"Error: Target column '{target_col}' not found in DataFrame.")
        return

    if isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy = df_copy.sort_index()

    if hasattr(feature_scaler, 'feature_names_in_'):
        feature_cols = feature_scaler.feature_names_in_
    else:
        feature_cols = df_copy.drop(target_col, axis=1).columns
        print(f"Warning: Inferring feature columns: {feature_cols.tolist()}")

    try:
        X = df_copy[feature_cols]
        y = df_copy[target_col]
    except KeyError as e:
        print(f"Error: Missing feature column in DataFrame: {e}")
        return

    X_scaled = feature_scaler.transform(X)
    y_scaled = target_scaler.transform(y.values.reshape(-1, 1))

    try:
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)
    except ValueError as e:
        print(f"Error creating sequences: {e}")
        return
        
    if len(X_seq) == 0:
        print(f"Error: Not enough data in the DataFrame (length {len(df_copy)}) to create sequences of length {sequence_length}.")
        return

    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)

    model.eval() 
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions_scaled = outputs.cpu().numpy()

    y_pred = target_scaler.inverse_transform(predictions_scaled).flatten()
    y_actual = y.iloc[sequence_length:].values
    actual_indices = df_copy.index[sequence_length:]

    if len(y_pred) != len(y_actual):
        print(f"Warning: Prediction length ({len(y_pred)}) differs from actual length ({len(y_actual)}). Truncating.")
        min_len = min(len(y_pred), len(y_actual))
        y_pred = y_pred[:min_len]
        y_actual = y_actual[:min_len]
        actual_indices = actual_indices[:min_len]

    plot_df = pd.DataFrame({
        'Actual': y_actual,
        'Predicted': y_pred
    }, index=actual_indices)
    
    plt.figure(figsize=(15, 7))
    plt.plot(plot_df.index, plot_df['Actual'], 'b-', label=f'Actual {y_label}', alpha=0.7)
    plt.plot(plot_df.index, plot_df['Predicted'], 'r--', label=f'Predicted {y_label}')
    plt.title(f'Unit {unit_id}: LSTM Actual vs. Predicted {title_suffix}')
    plt.xlabel('Date')
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return plot_df


def run_lstm_pipeline(df_unit, unit_id, test_size=0.2,
                      sequence_length=7, batch_size=16, epochs=100,
                      learning_rate=0.001, hidden_dim=64, num_layers=2, dropout=0.1,
                      device='cuda' if torch.cuda.is_available() else 'cpu',
                      target_type='price'):
    """
    Runs the complete LSTM training, evaluation, and plotting pipeline for a specific unit.

    Args:
        df_unit (pd.DataFrame): The DataFrame containing data for the specific unit.
        unit_id (str): The identifier for the unit (e.g., '166', '167').
        test_size (float): The proportion of the data to use for the test set.
        sequence_length (int): LSTM sequence length.
        batch_size (int): Training batch size.
        epochs (int): Number of training epochs.
        learning_rate (float): Optimizer learning rate.
        hidden_dim (int): LSTM hidden dimension size.
        num_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate.
        device (str): Device to run training on ('cpu' or 'cuda').
        target_type (str): Type of prediction ('price' or 'taux'). Defaults to 'price'.
    """
    prediction_type = "Price" if target_type.lower() == 'price' else "Occupancy Rate"
    print(f"--- Starting LSTM Pipeline for Unit {unit_id} (predicting {prediction_type}) ---")

    print(f"\n[Step 1/4] Splitting data with test_size={test_size}...")
    try:
        X_train, X_test, y_train, y_test, target_col, feature_col = split_data_train_test(
            df_unit, unit_id, test_size=test_size, target_type=target_type
        )
    except ValueError as e:
        print(f"Error during data splitting: {e}")
        return None
    except Exception as e:
         print(f"An unexpected error occurred during data splitting: {e}")
         return None

    print("\n[Step 2/4] Training and evaluating LSTM model...")
    try:
        lstm_results = train_evaluate_lstm_model(
            X_train, X_test, y_train, y_test, target_col,
            sequence_length=sequence_length,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            device=device,
            target_type=target_type
        )
    except ValueError as e:
        print(f"Error during model training/evaluation: {e}")
        return None
    except Exception as e:
         print(f"An unexpected error occurred during training/evaluation: {e}")
         return None

    print("\n[Step 3/4] Plotting results...")
    try:
        print("Plotting training loss...")
        plot_lstm_loss(lstm_results, unit_id)

        print("Plotting predictions on train/test split...")
        plot_lstm_predictions_over_time(lstm_results, unit_id)

        print("Plotting predictions on the full unit DataFrame...")
        plot_lstm_predictions_on_dataframe(
            df_unit,
            unit_id,
            lstm_results['model'],
            lstm_results['feature_scaler'],
            lstm_results['target_scaler'],
            lstm_results['sequence_length'],
            lstm_results['target_col'], 
            device
        )
    except KeyError as e:
        print(f"Error during plotting: Missing key in lstm_results - {e}")
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")

    print(f"\n[Step 4/4] Pipeline finished for Unit {unit_id}.")
    print("--- End LSTM Pipeline ---")

    return lstm_results


def save_lstm_model(lstm_results, folder_path, model_name):
    """
    Save LSTM model state dictionary, feature scaler, target scaler, and metadata.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created directory: {folder_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file_path = os.path.join(folder_path, f"{model_name}_model_state_dict.pth") 
    feature_scaler_path = os.path.join(folder_path, f"{model_name}_feature_scaler.pkl")
    target_scaler_path = os.path.join(folder_path, f"{model_name}_target_scaler.pkl")
    metadata_path = os.path.join(folder_path, f"{model_name}_metadata.json")

    model = lstm_results.get('model')
    feature_scaler = lstm_results.get('feature_scaler')
    target_scaler = lstm_results.get('target_scaler')
    sequence_length = lstm_results.get('sequence_length')
    target_col = lstm_results.get('target_col')
    target_type = lstm_results.get('target_type', 'price')  

    saved_paths = {}

    if model and isinstance(model, torch.nn.Module):
        try:
            torch.save(model.state_dict(), model_file_path)
            saved_paths['model_state_dict'] = model_file_path
            print(f"Model state dictionary successfully saved to {model_file_path}")
        except Exception as e:
            print(f"Error saving model state dictionary: {e}")
    else:
        print("Error: 'model' not found or is not a PyTorch Module in lstm_results.")

    if feature_scaler:
        try:
            with open(feature_scaler_path, 'wb') as f:
                pickle.dump(feature_scaler, f)
            saved_paths['feature_scaler'] = feature_scaler_path
            print(f"Feature scaler successfully saved to {feature_scaler_path}")
        except Exception as e:
            print(f"Error saving feature scaler: {e}")
    else:
        print("Warning: 'feature_scaler' not found in lstm_results.")

    if target_scaler:
        try:
            with open(target_scaler_path, 'wb') as f:
                pickle.dump(target_scaler, f)
            saved_paths['target_scaler'] = target_scaler_path
            print(f"Target scaler successfully saved to {target_scaler_path}")
        except Exception as e:
            print(f"Error saving target scaler: {e}")
    else:
        print("Warning: 'target_scaler' not found in lstm_results.")

    if model and sequence_length is not None:
         metadata = {
             'sequence_length': sequence_length,
             'input_dim': getattr(model, 'lstm', None).input_size if hasattr(model, 'lstm') else None,
             'hidden_dim': getattr(model, 'hidden_dim', None),
             'num_layers': getattr(model, 'num_layers', None),
             'dropout': getattr(getattr(model, 'dropout', None), 'p', None) if hasattr(model, 'dropout') else None,
             'feature_names_in': list(getattr(feature_scaler, 'feature_names_in_', [])) if feature_scaler else [],
             'target_col': target_col,  
             'target_type': target_type,  
             'saved_timestamp': timestamp
         }
         try:
             with open(metadata_path, 'w') as f:
                 json.dump(metadata, f, indent=4)
             saved_paths['metadata'] = metadata_path
             print(f"Model metadata saved to {metadata_path}")
         except Exception as e:
             print(f"Error saving metadata: {e}")
    else:
         print("Warning: Could not save metadata (model or sequence_length missing).")

    return saved_paths


def load_lstm_model(folder_path, model_name, device='cpu'):
    """
    Load LSTM model state dictionary, feature scaler, target scaler, and metadata.
    """
    model_file_path = os.path.join(folder_path, f"{model_name}_model_state_dict.pth")
    feature_scaler_path = os.path.join(folder_path, f"{model_name}_feature_scaler.pkl")
    target_scaler_path = os.path.join(folder_path, f"{model_name}_target_scaler.pkl")
    metadata_path = os.path.join(folder_path, f"{model_name}_metadata.json")

    loaded_components = {}
    model = None
    feature_scaler = None
    target_scaler = None
    metadata = None

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            loaded_components['metadata'] = metadata
            print(f"Metadata successfully loaded from {metadata_path}")
            
            if 'target_col' in metadata:
                loaded_components['target_col'] = metadata['target_col']
                print(f"Target column extracted: {metadata['target_col']}")
            if 'target_type' in metadata:
                loaded_components['target_type'] = metadata['target_type']
                print(f"Target type extracted: {metadata['target_type']}")
                
        except Exception as e:
            print(f"Error loading metadata from {metadata_path}: {e}")
            return None 
    else:
        print(f"Error: Metadata file not found at {metadata_path}")
        return None

    if metadata:
        try:
            model = LSTMModel(
                input_dim=metadata['input_dim'],
                hidden_dim=metadata.get('hidden_dim', 80), 
                num_layers=metadata.get('num_layers', 2),
                dropout=metadata.get('dropout', 0.2)
            ).to(device)
            
            if 'target_type' in metadata:
                setattr(model, 'target_type', metadata['target_type'])
                
            print(f"LSTMModel instantiated with parameters from metadata.")
        except NameError:
             print("Error: LSTMModel class definition not found. Please ensure it's defined.")
             return None
        except KeyError as e:
            print(f"Error: Missing essential key in metadata for model instantiation: {e}")
            return None
        except Exception as e:
            print(f"Error instantiating LSTMModel: {e}")
            return None

    if model and os.path.exists(model_file_path):
        try:
            model.load_state_dict(torch.load(model_file_path, map_location=device))
            model.eval() 
            loaded_components['model'] = model
            print(f"Model state dictionary successfully loaded from {model_file_path} onto {device}")
        except Exception as e:
            print(f"Error loading model state dictionary from {model_file_path}: {e}")

    elif not os.path.exists(model_file_path):
         print(f"Warning: Model state dictionary file not found at {model_file_path}")
    
    elif not model:
         print("Warning: Model could not be instantiated, skipping state dict loading.")

    if os.path.exists(feature_scaler_path):
        try:
            with open(feature_scaler_path, 'rb') as f:
                feature_scaler = pickle.load(f)
            loaded_components['feature_scaler'] = feature_scaler
            print(f"Feature scaler successfully loaded from {feature_scaler_path}")
        except Exception as e:
            print(f"Error loading feature scaler from {feature_scaler_path}: {e}")
    else:
        print(f"Warning: Feature scaler file not found at {feature_scaler_path}")

    if os.path.exists(target_scaler_path):
        try:
            with open(target_scaler_path, 'rb') as f:
                target_scaler = pickle.load(f)
            loaded_components['target_scaler'] = target_scaler
            print(f"Target scaler successfully loaded from {target_scaler_path}")
        except Exception as e:
            print(f"Error loading target scaler from {target_scaler_path}: {e}")
    else:
        print(f"Warning: Target scaler file not found at {target_scaler_path}")

    if metadata and 'sequence_length' in metadata:
        loaded_components['sequence_length'] = metadata['sequence_length']

    if 'model' not in loaded_components or 'feature_scaler' not in loaded_components or 'target_scaler' not in loaded_components:
        print("\nWarning: Not all essential components (model, feature_scaler, target_scaler) could be loaded.")
        return None

    print("\nLoading process finished.")
    return loaded_components




def predict_single_step_lstm(target_date_str, taux_or_price_target, df_history, 
                             loaded_model_data, 
                             unit_id, device, target_type='price', 
                             weather_data_dict=None, city=metaData['weatherInformation']['default_city']):
    """
    Predicts the price or occupancy rate for a single future date using a trained LSTM model,
    including weather features for the specified date. Uses pre-loaded weather data for fast performance.
    Requires historical data up to the day before the target date to form the input sequence.
    
    Args:
        target_date_str (str): The target date for prediction in 'YYYY-MM-DD' format.
        taux_or_price_target (float): The value for the non-target feature (price or taux_occupation).
        df_history (pd.DataFrame): DataFrame containing historical data for the unit.
        loaded_model_data (dict): Dictionary containing model, scalers, and metadata.
        unit_id (str): The identifier for the unit (e.g., '166', '167').
        device (str): Device to run LSTM model on ('cpu' or 'cuda').
        target_type (str): Type of prediction ('price' or 'taux'). Defaults to 'price'.
        weather_data_dict (dict): Pre-loaded weather data dictionary with dates as keys. 
                                 If None, will fall back to API call.
        city (str): City name for weather data (only used if weather_data_dict is None).
    
    Returns:
        float or None: The predicted value, or None if prediction fails.
    """
    predicting_price = target_type.lower() == 'price'
    prediction_type = "price" if predicting_price else "taux"
    feature_type = "taux" if predicting_price else "price"
    
    target_col = f'total_price_{unit_id}' if predicting_price else f'taux_occupation_{unit_id}'
    feature_col = f'taux_occupation_{unit_id}' if predicting_price else f'total_price_{unit_id}'
    
    print(f"\n--- Predicting {prediction_type} for Unit {unit_id} on {target_date_str} ---")
    try:
        model = loaded_model_data.get('model')
        feature_scaler = loaded_model_data.get('feature_scaler')
        target_scaler = loaded_model_data.get('target_scaler')
        metadata = loaded_model_data.get('metadata')

        if not all([model, feature_scaler, target_scaler, metadata]):
            print("Error: Missing model, scalers, or metadata in loaded_model_data.")
            return None

        model_target_type = metadata.get('target_type')
        if model_target_type and model_target_type.lower() != target_type.lower():
            print(f"Warning: Model was trained to predict {model_target_type} but you're using it to predict {target_type}.")

        sequence_length = metadata.get('sequence_length')
        if sequence_length is None or not isinstance(sequence_length, int) or sequence_length <= 0:
            print(f"Error: Invalid 'sequence_length' ({sequence_length}) found in metadata.")
            return None

        try:
            target_date = pd.to_datetime(target_date_str)
        except ValueError:
            print(f"Error: Invalid target_date_str format: {target_date_str}. Use 'YYYY-MM-DD'.")
            return None

        try:
            week_of_year = target_date.isocalendar().week
        except AttributeError:
            week_of_year = target_date.isocalendar()[1]

        target_features = {
            'dayofweek': target_date.dayofweek,
            'month': target_date.month,
            'year': target_date.year,
            'dayofyear': target_date.dayofyear,
            'weekofyear': int(week_of_year),  
            feature_col: taux_or_price_target  
        }

        try:
            if weather_data_dict is not None:
                weather_df = get_weather_dataframe_for_city_single_date_fast(target_date_str, weather_data_dict)

            else:
                weather_df = get_weather_dataframe_for_city_single_date(target_date_str, city)
            
            if not weather_df.empty:
                weather_row = weather_df.iloc[0]  # Get the first (and only) row
                
                target_features.update({
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
                target_features.update({
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

            target_features.update({
                'temperature_max': 0.0,
                'temperature_min': 0.0,
                'temperature_mean': 0.0,
                'precipitation': 0.0,
                'windspeed_max': 0.0,
                'temperature_range': 0.0
            })

        target_row = pd.DataFrame([target_features], index=[target_date])
        print(f"Generated features for target date: {target_features}")

        if not isinstance(df_history.index, pd.DatetimeIndex):
            print("Error: df_history must have a DatetimeIndex.")
            return None
        df_history = df_history.sort_index()

        last_hist_date_needed = target_date - timedelta(days=1)
        first_hist_date_needed = last_hist_date_needed - timedelta(days=sequence_length - 2) 

        print(f"Fetching history from {first_hist_date_needed.date()} to {last_hist_date_needed.date()}")

        historical_sequence_df = df_history.loc[first_hist_date_needed : last_hist_date_needed]

        if len(historical_sequence_df) < sequence_length - 1:
            print(f"Error: Not enough historical data points found ({len(historical_sequence_df)}) between "
                  f"{first_hist_date_needed.date()} and {last_hist_date_needed.date()}. "
                  f"Need {sequence_length - 1}.")
            return None
        elif len(historical_sequence_df) > sequence_length - 1:
            print(f"Warning: Found more data points ({len(historical_sequence_df)}) than needed ({sequence_length - 1}). Using the latest ones.")
            historical_sequence_df = historical_sequence_df.iloc[-(sequence_length - 1):]

        input_sequence_df = pd.concat([historical_sequence_df, target_row])

        feature_order = metadata.get('feature_names_in') 
        if not feature_order and hasattr(feature_scaler, 'feature_names_in_'):
            feature_order = list(feature_scaler.feature_names_in_) 
        elif not feature_order:
            print("Error: Cannot determine feature order. Missing 'feature_names_in' in metadata and scaler.")
            return None

        print(f"Expected feature order: {feature_order}")

        missing_cols = [col for col in feature_order if col not in input_sequence_df.columns]
        if missing_cols:
            print(f"Error: Missing required feature columns in combined input data: {missing_cols}")
            return None

        try:
            input_sequence_df = input_sequence_df[feature_order]
        except Exception as e:
            print(f"Error reordering columns: {e}")
            return None

        if input_sequence_df.isnull().values.any():
            print("Error: NaNs found in the input sequence before scaling:")
            print(input_sequence_df[input_sequence_df.isnull().any(axis=1)])
            return None

        print(f"Scaling sequence of shape: {input_sequence_df.shape}")
        try:
            input_sequence_scaled = feature_scaler.transform(input_sequence_df)
        except ValueError as e:
            print(f"Error scaling input sequence. Check feature mismatch or dtype issues: {e}")
            print("Input DataFrame dtypes:\n", input_sequence_df.dtypes)
            return None
        except Exception as e:
            print(f"Error during scaling: {e}")
            return None

        input_tensor = torch.tensor(input_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        print(f"Input tensor shape: {input_tensor.shape}")

        model.eval()
        with torch.no_grad():
            prediction_scaled = model(input_tensor)
        print(f"Scaled prediction shape: {prediction_scaled.shape}")

        predicted_value = target_scaler.inverse_transform(prediction_scaled.cpu().numpy())

        final_prediction = predicted_value[0][0]

        print(f"Predicted {prediction_type} (Unit {unit_id}) for {target_date_str}: {final_prediction:.2f}")
        return final_prediction

    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        traceback.print_exc() 
        return None
    

def find_max_price_for_date_lstm(
    target_date_str, 
    unit_id, 
    df_history, 
    device='cuda' if torch.cuda.is_available() else 'cpu', 
    min_value=0.0, 
    max_value=95.0, 
    step=0.01,
    target_type='price',
    config_file_path=metaDataPath,
    modelsName="LSTMModels",
    weather_data_dict=None
):
    """
    Finds the value that maximizes the predicted price or occupancy rate for a single date
    using a trained LSTM model by iterating through a range of values.
    Automatically loads the appropriate model using choose_LSTM_model.

    Args:
        target_date_str (str): The target date for prediction in 'YYYY-MM-DD' format.
        unit_id (str): The identifier for the unit (e.g., '166', '167').
        df_history (pd.DataFrame): DataFrame containing historical data for the unit.
                                   Must have a DatetimeIndex and include necessary features.
        device (str): Device to run LSTM model on ('cpu' or 'cuda').
        min_value (float): Minimum value to test.
        max_value (float): Maximum value to test.
        step (float): Step for iterating values.
        target_type (str): Type of prediction ('price' or 'taux').
        config_file_path (str): Path to the model configuration file.
        modelsName (str): Name of the models section in the config file.
    """
    # Load the model using choose_LSTM_model
    loaded_model_data = choose_LSTM_model(
        target=target_type, 
        unit=unit_id, 
        modelsName=modelsName, 
        config_file_path=config_file_path
    )
    
    if loaded_model_data is None:
        print(f"Error: Could not load model for target='{target_type}', unit='{unit_id}'")
        return None, None
    
    predicting_price = target_type.lower() == 'price'
    prediction_type = "price" if predicting_price else "taux" 
    feature_type = "taux" if predicting_price else "price"
    
    varied_feature = f'taux_occupation' if predicting_price else f'total_price'
    target_col = f'total_price_{unit_id}' if predicting_price else f'taux_occupation_{unit_id}'
    
    best_input_value = None
    max_predicted_value = -float('inf')

    print(f"\nSearching for optimal {feature_type} input that maximizes {prediction_type}")
    print(f"for LSTM model (unit {unit_id}) on {target_date_str}...")
    print(f"Range: {min_value}-{max_value}, step: {step}")

    num_steps = int((max_value - min_value) / step) + 1
    test_values = np.linspace(min_value, max_value, num_steps)

    for test_value in tqdm(test_values, desc=f"Testing {varied_feature} values"):
        predicted_value = predict_single_step_lstm(
            target_date_str=target_date_str,
            taux_or_price_target=test_value,
            df_history=df_history, 
            loaded_model_data=loaded_model_data,
            unit_id=unit_id,
            device=device,
            target_type=target_type,
            weather_data_dict=weather_data_dict
        )
        
        if predicted_value is not None:
            if predicted_value > max_predicted_value:
                max_predicted_value = predicted_value
                best_input_value = test_value
    
    if best_input_value is not None:
        print(f"\nLSTM Search complete. Optimal {varied_feature}: {best_input_value:.2f} -> ")
        print(f"Max {target_col}: {max_predicted_value:.2f}")
        return best_input_value, max_predicted_value
    else:
        print(f"LSTM Search complete. No valid predictions found for {target_col}.")
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

def choose_LSTM_model(target, unit, modelsName="LSTMModels", config_file_path=metaDataPath):
    """
    Loads an LSTM model based on the target and unit using a configuration file.

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
            
            model = load_lstm_model(
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
    

def find_max_output_for_date_range_lstm(
    start_date_str,
    end_date_str,
    unit_id,
    df_166,
    df_167,
    device='cuda' if torch.cuda.is_available() else 'cpu',     
    min_value=30.0,
    max_value=95.0,
    step=1,
    target_type='price',
    config_file_path=metaDataPath, 
    modelsName="LSTMModels",
    weather_data_dict=None
):
    """
    Finds the input that maximizes the predicted output (price or taux) for a range of dates 
    using an LSTM model. Selects df_history based on unit_id.

    Args:
        start_date_str (str): The start date in 'YYYY-MM-DD' format.
        end_date_str (str): The end date in 'YYYY-MM-DD' format.
        unit_id (str): The unit ID (e.g., '166', '167'). This will also determine df_history.
        df_166 (pd.DataFrame): DataFrame containing historical data for unit 166.
        df_167 (pd.DataFrame): DataFrame containing historical data for unit 167.
        device (str): Device to run LSTM model on ('cpu' or 'cuda').
        min_value (float): Minimum input value to test for the feature being varied.
        max_value (float): Maximum input value to test.
        step (float): Step for iterating through input values.
        target_type (str): The type of target to predict ('price' or 'taux').
        config_file_path (str): Path to the JSON configuration file for model selection.
        modelsName (str): Key for LSTM models in the config file (e.g., "LSTMModels").
    """
    
    if unit_id == '166':
        df_history_selected = df_166
    elif unit_id == '167':
        df_history_selected = df_167
    else:
        print(f"Error: Invalid unit_id '{unit_id}'. Cannot select df_history.")
        return {f"Error: Invalid unit_id '{unit_id}' for LSTM range search."}
    
    if df_history_selected is None:
        print(f"Error: df_history for unit_id '{unit_id}' is None.")
        return {f"Error: df_history for unit_id '{unit_id}' is None for LSTM range search."}

    all_dates = get_dates(start_date_str, end_date_str)
    results_for_range = {}

    for target_date_str_loop in all_dates:
        print(f"\nProcessing date: {target_date_str_loop} for LSTM...")
        try:
            best_input, max_output = find_max_price_for_date_lstm(
                target_date_str=target_date_str_loop,
                unit_id=unit_id,
                df_history=df_history_selected, 
                device=device,         
                min_value=min_value,    
                max_value=max_value,    
                step=step,
                target_type=target_type,
                config_file_path=config_file_path,
                modelsName=modelsName,
                weather_data_dict=weather_data_dict
            )
            
            if best_input is not None:
                results_for_range[target_date_str_loop] = {"best_input": best_input, "max_output": max_output}
            
            else:
                results_for_range[target_date_str_loop] = "No valid prediction found for this date."
        
        except Exception as e:
            error_message = f"Error processing date {target_date_str_loop} with LSTM: {str(e)}"
            results_for_range[target_date_str_loop] = error_message
            print(error_message)

    return results_for_range


def find_max_price_for_single_date_lstm(
    target_date_str, 
    unit_id, 
    df_history, 
    device='cuda' if torch.cuda.is_available() else 'cpu', 
    min_value=0.0, 
    max_value=95.0, 
    step=0.01,
    target_type='price',
    config_file_path=metaDataPath,
    modelsName="LSTMModels",
    weather_data_dict=None
):
    """
    Finds the value that maximizes the predicted price or occupancy rate for a single date
    using a trained LSTM model by iterating through a range of values.
    Automatically loads the appropriate model using choose_LSTM_model.

    Args:
        target_date_str (str): The target date for prediction in 'YYYY-MM-DD' format.
        unit_id (str): The identifier for the unit (e.g., '166', '167').
        df_history (pd.DataFrame): DataFrame containing historical data for the unit.
                                   Must have a DatetimeIndex and include necessary features.
        device (str): Device to run LSTM model on ('cpu' or 'cuda').
        min_value (float): Minimum value to test.
        max_value (float): Maximum value to test.
        step (float): Step for iterating values.
        target_type (str): Type of prediction ('price' or 'taux').
        config_file_path (str): Path to the model configuration file.
        modelsName (str): Name of the models section in the config file.
    """

    loaded_model_data = choose_LSTM_model(
        target=target_type, 
        unit=unit_id, 
        modelsName=modelsName, 
        config_file_path=config_file_path
    )
    
    if loaded_model_data is None:
        print(f"Error: Could not load model for target='{target_type}', unit='{unit_id}'")
        return None, None
    
    predicting_price = target_type.lower() == 'price'
    prediction_type = "price" if predicting_price else "taux" 
    feature_type = "taux" if predicting_price else "price"
    
    varied_feature = f'taux_occupation' if predicting_price else f'total_price'
    target_col = f'total_price_{unit_id}' if predicting_price else f'taux_occupation_{unit_id}'
    
    best_input_value = None
    max_predicted_value = -float('inf')

    print(f"\nSearching for optimal {feature_type} input that maximizes {prediction_type}")
    print(f"for LSTM model (unit {unit_id}) on {target_date_str}...")
    print(f"Range: {min_value}-{max_value}, step: {step}")

    num_steps = int((max_value - min_value) / step) + 1
    test_values = np.linspace(min_value, max_value, num_steps)

    for test_value in tqdm(test_values, desc=f"Testing {varied_feature} values"):
        predicted_value = predict_single_step_lstm(
            target_date_str=target_date_str,
            taux_or_price_target=test_value,
            df_history=df_history, 
            loaded_model_data=loaded_model_data,
            unit_id=unit_id,
            device=device,
            target_type=target_type,
            weather_data_dict=weather_data_dict
        )
        
        if predicted_value is not None:
            if predicted_value > max_predicted_value:
                max_predicted_value = predicted_value
                best_input_value = test_value
    
    if best_input_value is not None:
        print(f"\nLSTM Search complete. Optimal {varied_feature}: {best_input_value:.2f} -> ")
        print(f"Max {target_col}: {max_predicted_value:.2f}")
        return best_input_value, max_predicted_value
    else:
        print(f"LSTM Search complete. No valid predictions found for {target_col}.")
        return None, None