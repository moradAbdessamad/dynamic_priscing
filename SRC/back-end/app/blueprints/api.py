from flask import Blueprint #type: ignore
from flask import Flask, request, jsonify #type: ignore
from app.utils.helpers import *
from dotenv import load_dotenv # type: ignore
import os 
from pathlib import Path
import sys
from datetime import datetime
import numpy as np # type: ignore
from app.utils.helpers import send_whatsapp_session_message, send_openai_request, load_xgboost_model, find_max_price_for_date_xgboost, extract_date_unit_from_text
from app.utils.helpersEtract import *

project_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_src_directory not in sys.path:
    sys.path.append(project_src_directory)

from helpFolder import *
import SRC

load_dotenv()
access_token = os.getenv("META_ACCESS_TOKEN")
phone_number_id = os.getenv("PHONE_NUMBER_ID")
recipient_phone = os.getenv("RECIPIENT_PHONE")

api_bp = Blueprint('api', __name__)

load_dotenv()
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")

SRC_DIR = Path(__file__).parent.parent.parent.parent 
MODEL_DIR_RELATIVE_TO_SRC = "TrainedModels/XGBoostModels"
ABSOLUTE_MODEL_PATH = SRC_DIR / MODEL_DIR_RELATIVE_TO_SRC

df_166_global_lstm = None
df_167_global_lstm = None
lstm_data_loaded = False

metadata = SRC.load_model_metadata()

weather_data_dict = load_weather_data_from_json()

def ensure_lstm_data_loaded():
    """
    Ensures the LSTM data is loaded from the path specified in metadata.
    Returns the DataFrames needed for LSTM predictions.
    For V3 models, concatenates weather data.
    """
    global df_166_global_lstm, df_167_global_lstm, lstm_data_loaded
    
    if not lstm_data_loaded:
        try:
            data_csv_file_path = metadata['PMSInformation']['final_csv_path']
            start_date = metadata['PMSInformation']['start_date_range']
            end_date = metadata['PMSInformation']['end_date_range']
            active_version = metadata.get("model_versions", {}).get("active_version", "v1")
            
            if os.path.exists(data_csv_file_path):
                try:
                    print(f"Attempting to load LSTM data from: {data_csv_file_path}")
                    print(f"Using date range: {start_date} to {end_date}")
                    print(f"Using version: {active_version}")
                    
                    df_167_global_lstm, df_166_global_lstm = prepare_the_data(
                        data_csv_file_path,
                        start_date,
                        end_date
                    )

                    if df_166_global_lstm is not None and df_167_global_lstm is not None:

                        if active_version == "v3":
                            try:
                                print("Loading weather data for V3 models...")
                                df_weather = make_weather_data_dataframe()
                                
                                if df_weather is not None:
                                    print("Concatenating weather data with LSTM dataframes...")
                                    df_166_global_lstm = concatenate_on_index(df_166_global_lstm, df_weather)
                                    df_167_global_lstm = concatenate_on_index(df_167_global_lstm, df_weather)
                                    print("Weather data successfully concatenated for V3 models.")
                                else:
                                    print("Warning: Weather data could not be loaded, using base data only.")
                                    
                            except Exception as e:
                                print(f"Error loading weather data for V3: {e}")
                                print("Falling back to base data without weather information.")
                        
                        lstm_data_loaded = True
                        print("DataFrames for LSTM loaded successfully.")
                    else:
                        print("Failed to load LSTM DataFrames: prepare_the_data returned None.")

                except NameError:
                    print("Error: prepare_the_data function not found. Make sure it's imported from helpFolder.")

                except Exception as e:
                    print(f"Error loading data for LSTM: {e}")
            else:
                print(f"LSTM Data CSV file not found at {data_csv_file_path}. LSTM predictions cannot proceed without data.")
        
        except Exception as e:
            print(f"Error loading metadata or extracting values: {e}")
            
    return df_166_global_lstm, df_167_global_lstm


def custom_serializer(obj):
    if isinstance(obj, np.float32):
        return float(obj)  
    raise TypeError("Type not serializable")


@api_bp.route('/', methods=['GET'])
def hello_world():
    return 'Hello, World!'

@api_bp.route("/webhooks", methods=["GET", "POST"])
def verify_webhook():
    try:
        if request.method == "GET":
            mode = request.args.get("hub.mode")
            token = request.args.get("hub.verify_token")
            challenge = request.args.get("hub.challenge")

            if mode == "subscribe" and token == VERIFY_TOKEN:
                return challenge, 200
            else:
                return "Verification failed", 403

        if request.method == "POST":
            data = request.get_json()
            print("Received webhook:", data)

            user_message = None
            try:
                    if (data and isinstance(data, dict) and
                            data.get('object') == 'whatsapp_business_account' and
                            'entry' in data and data['entry'] and
                            isinstance(data['entry'], list) and data['entry'][0] and
                            'changes' in data['entry'][0] and data['entry'][0]['changes'] and
                            isinstance(data['entry'][0]['changes'], list) and data['entry'][0]['changes'][0] and
                            'value' in data['entry'][0]['changes'][0] and data['entry'][0]['changes'][0]['value'] and
                            'messages' in data['entry'][0]['changes'][0]['value'] and 
                            data['entry'][0]['changes'][0]['value']['messages'] and
                            isinstance(data['entry'][0]['changes'][0]['value']['messages'], list) and
                            data['entry'][0]['changes'][0]['value']['messages'][0] and
                            data['entry'][0]['changes'][0]['value']['messages'][0].get('type') == 'text' and
                            'text' in data['entry'][0]['changes'][0]['value']['messages'][0] and
                            'body' in data['entry'][0]['changes'][0]['value']['messages'][0]['text']):
                        
                        user_message = data['entry'][0]['changes'][0]['value']['messages'][0]['text']['body']
                        sender_phone_number = data['entry'][0]['changes'][0]['value']['messages'][0]['from']

                        if user_message:
                            if user_message.lower() in ["hi", "hello", "bonjour", "salut", "slm"]:
                                response_message = (
                                    "Hello! I'm your hotel pricing assistant 🏨\n\n"
                                    "I can help you find the optimal pricing for:\n"
                                    "• Unit 166\n"
                                    "• Unit 167\n\n"
                                    "Please provide:\n"
                                    "1. Date (e.g., \"2025-07-15\")\n"
                                    "2. Unit (166 or 167)\n"
                                    "3. Prediction type (price or taux)\n\n"
                                    "Example: \"Date: 2025-07-15, Unit: 166, Predict: price\""
                                )

                                whatsapp_response = send_whatsapp_session_message(
                                                          phone_number_id=phone_number_id, 
                                                          recipient_phone=sender_phone_number, 
                                                          access_token=access_token, 
                                                          message_text=response_message)
                                
                                return jsonify({"status": "success", "result": f"Welcome message sent {whatsapp_response.status_code if whatsapp_response else 'Failed'}"}), 200
                            
                            else:
                                extracted_info = extract_chatbot_info(user_message)
                                
                                if not extracted_info["valid"]:
                                    response_message = (
                                        "❌ I couldn't understand your request.\n\n"
                                        "Please provide:\n"
                                        "• Date (e.g., \"2025-07-15\")\n"
                                        "• Unit: 166 or 167\n"
                                        "• Prediction type: price or taux\n\n"
                                        "Example: \"Date: 2025-07-15, Unit: 166, Predict: price\""
                                    )
                                else:
                                    target_date = extracted_info["date"]
                                    unit_id = extracted_info["unit_id"]
                                    prediction_type = extracted_info["prediction_type"]
                                    
                                    print(f"Extracted - Date: {target_date}, Unit: {unit_id}, Prediction: {prediction_type}")
                                    
                                    # Get active version and model configuration
                                    active_version = metadata.get("model_versions", {}).get("active_version", "v1")
                                    
                                    if active_version == "v3":
                                        model_section = "XGBoostModelsV3"
                                        try:
                                            weather_data_dict = load_weather_data_from_json(
                                                json_file_path=f"../{metadata['weatherInformation']['archive_json_path']}"
                                            )
                                        except Exception as e:
                                            print(f"Warning: Could not load weather data: {e}")
                                            weather_data_dict = {}
                                    else:
                                        try:
                                            model_section = metadata["model_versions"]["models"][active_version]["xgboost"]
                                        except KeyError:
                                            model_section = "xgboost"
                                        weather_data_dict = {}
                                    
                                    # Set default ranges based on prediction type
                                    if prediction_type == "price":
                                        min_input_value = 0.0
                                        max_input_value = 90.0
                                        step_value = 0.5
                                    else:  # taux
                                        min_input_value = 100.0
                                        max_input_value = 1000.0
                                        step_value = 10.0
                                    
                                    try:
                                        # Always use XGBoost model
                                        if active_version == "v3":
                                            best_input_value, max_predicted_value = find_max_price_for_single_date_xgboost(
                                                target_date_str=target_date,
                                                unit_id=unit_id,
                                                min_value=min_input_value,
                                                max_value=max_input_value,
                                                step=step_value,
                                                target_type=prediction_type,
                                                modelsName=model_section,
                                                weather_data_dict=weather_data_dict
                                            )
                                        else:
                                            best_input_value, max_predicted_value = find_max_price_for_single_date_xgboost(
                                                target_date_str=target_date,
                                                unit_id=unit_id,
                                                min_value=min_input_value,
                                                max_value=max_input_value,
                                                step=step_value,
                                                target_type=prediction_type,
                                                modelsName=model_section
                                            )
                                        
                                        if best_input_value is not None and max_predicted_value is not None:
                                            if prediction_type == "price":
                                                # Calculate price per room for price predictions
                                                room_type = None
                                                for room, unit in metadata["PMSInformation"]["room_map"].items():
                                                    if unit == unit_id:
                                                        room_type = room
                                                        break
                                                
                                                if room_type and room_type in metadata["PMSInformation"]["room_count"]:
                                                    room_count = metadata["PMSInformation"]["room_count"][room_type]
                                                    price_per_room = max_predicted_value / room_count if room_count > 0 else 0
                                                else:
                                                    price_per_room = 0
                                                    print(f"Warning: Could not find room count for unit {unit_id}")
                                                
                                                response_message = (
                                                    f"🏨 **Unit {unit_id} Pricing Prediction**\n"
                                                    f"📅 Date: {target_date}\n"
                                                    f"📊 Optimal occupancy rate: {best_input_value:.1f}%\n"
                                                    f"💰 Maximum predicted price: {max_predicted_value:.2f} MAD\n"
                                                    f"🏠 Price per room: {price_per_room:.2f} MAD"
                                                )
                                            else:  # taux
                                                response_message = (
                                                    f"🏨 **Unit {unit_id} Occupancy Prediction**\n"
                                                    f"📅 Date: {target_date}\n"
                                                    f"💰 Optimal price: {best_input_value:.2f} MAD\n"
                                                    f"📊 Maximum predicted occupancy: {max_predicted_value:.1f}%"
                                                )
                                        else:
                                            response_message = (
                                                f"❌ Sorry, I couldn't generate a prediction for Unit {unit_id} on {target_date}. "
                                                "Please try a different date or check if the model is available."
                                            )
                                    
                                    except Exception as e:
                                        print(f"Prediction error: {e}")
                                        response_message = (
                                            f"❌ Error generating prediction: {str(e)}\n"
                                            "Please try again or contact support."
                                        )
                                
                                print(f"Response message: {response_message}")

                                whatsapp_response = send_whatsapp_session_message(
                                    phone_number_id=phone_number_id, 
                                    recipient_phone=sender_phone_number, 
                                    access_token=access_token,
                                    message_text=response_message
                                )
                                return jsonify({"status": "success", "result": f"Prediction message sent {whatsapp_response.status_code if whatsapp_response else 'Failed'}"}), 200
                        else:
                            print("No user message found in the webhook data.")
                            return jsonify({"status": "received_but_not_actionable", "reason": "no_user_message_found"}), 200
                    
                    else:
                        print("Webhook received, but not a processable user text message or malformed.")
                        return jsonify({"status": "received_but_not_actionable", "reason": "not_a_text_message_or_malformed"}), 200

            except Exception as e:
                print(f"Error processing webhook POST request: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500
                
    except Exception as e:
        print(f"Unhandled exception in /webhooks: {str(e)}") 
        return jsonify({"success": False, "error": str(e)}), 500

def extract_chatbot_info(user_input):
    """
    Extract date, unit, and prediction type from user message
    """
    import re
    from datetime import datetime
    
    # Initialize result
    result = {
        "valid": False,
        "date": None,
        "unit_id": None,
        "prediction_type": None
    }
    
    user_input_lower = user_input.lower()
    
    # Extract date
    date_patterns = [
        r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',  # YYYY-MM-DD or YYYY/MM/DD
        r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b',  # DD-MM-YYYY or MM/DD/YYYY
        r'date[:\s]*["\']?([0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2})["\']?',  # date: "YYYY-MM-DD"
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, user_input_lower)
        if match:
            try:
                date_str = match.group(1)
                # Try to parse and reformat to YYYY-MM-DD
                if '/' in date_str:
                    date_str = date_str.replace('/', '-')
                
                # Handle different date formats
                parts = date_str.split('-')
                if len(parts) == 3:
                    if len(parts[0]) == 4:  # YYYY-MM-DD
                        result["date"] = date_str
                    else:  # DD-MM-YYYY or MM-DD-YYYY
                        if int(parts[0]) > 12:  # DD-MM-YYYY
                            result["date"] = f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
                        else:  # MM-DD-YYYY
                            result["date"] = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
                break
            except:
                continue
    
    # Extract unit ID (166 or 167 only)
    unit_patterns = [
        r'\b(166|167)\b',  # Direct unit numbers
        r'unit[:\s]*["\']?(166|167)["\']?',  # unit: 166
    ]
    
    for pattern in unit_patterns:
        match = re.search(pattern, user_input_lower)
        if match:
            result["unit_id"] = match.group(1) if pattern.startswith(r'\b') else match.group(1)
            break
    
    # Extract prediction type
    if any(word in user_input_lower for word in ["price", "pricing", "cost"]):
        result["prediction_type"] = "price"
    elif any(word in user_input_lower for word in ["taux", "occupancy", "occupation"]):
        result["prediction_type"] = "taux"
    
    # Check if all required fields are present
    if result["date"] and result["unit_id"] and result["prediction_type"]:
        result["valid"] = True
    
    return result


from_date = "2025-05-13"
to_date = "2025-05-17"
total_rooms = 432

@api_bp.route("/calculate_true_price", methods=["POST"])
def calculate_true_price():
    """
    Calculate the true values of lines from the PMS APIs
    """

    try:
        if request.method != "POST":
            return jsonify({"status": "error", "message": "Only POST requests are allowed"}), 405
        
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        if not data.get("from_date") and not data.get("to_date") and not data.get("total_rooms"):
            return jsonify({"status": "error", "message": "from_date, to_date, and total_rooms are required"}), 400
        
        from_date = data.get("from_date")
        to_date = data.get("to_date")
        total_rooms = data.get("total_rooms")

        if not isinstance(from_date, str) or not isinstance(to_date, str) or not isinstance(total_rooms, int):
            return jsonify({"status": "error", "message": "from_date and to_date must be strings, and total_rooms must be an integer"}), 400
        
        result = get_occupancy_data(
            start_date_str=from_date,
            end_date_str=to_date,
            total_rooms_param=total_rooms,
        )

        if result is None:
            return jsonify({"status": "error", "message": "Failed to get occupancy data"}), 500
        
        return jsonify({"status": "success", "result": result}), 200
    
    except Exception as e:
        print(f"Error in calculate_true_price: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/calculate_predicted", methods=["POST"])
def calculate_predicted():
    """
    Calculate the predicted values based on the API's config.
    Uses trained models from "TrainedModels" based on active version from metadata.
    """

    try:
        if request.method != "POST":
            return jsonify({"status": "error", "message": "Only POST requests are allowed"}), 405
        
        data = request.get_json()

        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        required_fields = ["fromDate", "toDate", "models", "targetPredict", 
                           "targetValueMin", "targetValueMax", "unit"]
        
        for field in required_fields:
            if field not in data:
                return jsonify({"status": "error", "message": f"Missing required field: {field}"}), 400

        from_date_str = data["fromDate"]
        to_date_str = data["toDate"]
        model_choice = data["models"]  # "XGBoost", "LSTM", "LinearRegression"
        target_type_predict = data["targetPredict"]  # "price" or "taux"
        min_input_value = float(data["targetValueMin"])
        max_input_value = float(data["targetValueMax"])
        unit_id_str = str(data["unit"])  # unit_id is a string
        
        active_version = metadata.get("model_versions", {}).get("active_version", "v1")
        
        if active_version == "v3":
            model_type_map = {
                "XGBoost": "XGBoostModelsV3",
                "LinearRegression": "LinearRegressionModelsV3",
                "LSTM": "LSTMModelsV3"
            }
        else:
            model_type_map = {
                "XGBoost": "xgboost",
                "LinearRegression": "linear",
                "LSTM": "lstm"
            }
        
        model_type = model_type_map.get(model_choice)
        if not model_type:
            return jsonify({"status": "error", "message": f"Invalid model choice: {model_choice}"}), 400
        
        weather_data_dict = None
        if active_version == "v3":
            try:
                weather_data_dict = load_weather_data_from_json(
                    json_file_path=f"../{metadata['weatherInformation']['archive_json_path']}"
                )
            except Exception as e:
                return jsonify({"status": "error", "message": f"Failed to load weather data: {str(e)}"}), 500
        
        if active_version != "v3":
            try:
                model_section = metadata["model_versions"]["models"][active_version][model_type]
            except KeyError:
                return jsonify({"status": "error", "message": f"Invalid version or model type: {active_version}/{model_type}"}), 400
        else:
            model_section = model_type

        step_value = 1.0 
        if (max_input_value - min_input_value) > 1000: 
            step_value = (max_input_value - min_input_value) / 100 
            if target_type_predict == 'taux' and step_value < 0.1:  # For taux, step shouldn't be too small
                step_value = 0.1
            elif target_type_predict == 'price' and step_value < 100:  # For price, step shouldn't be too small
                step_value = 100

        prediction_results = None
        
        print(f"Received request: model={model_choice}, target={target_type_predict}, unit={unit_id_str}, "
              f"dates={from_date_str}-{to_date_str}, range=[{min_input_value}-{max_input_value}], step={step_value}, "
              f"using version={active_version}, model_section={model_section}")

        if model_choice == "XGBoost":
            try:
                if active_version == "v3":
                    prediction_results = find_max_price_for_date_range_xgboost(
                        start_date_str=from_date_str,
                        end_date_str=to_date_str,
                        unit_id=unit_id_str,
                        min_value=min_input_value,
                        max_value=max_input_value,
                        step=step_value,
                        target_type=target_type_predict,
                        modelsName=model_section,
                        weather_data_dict=weather_data_dict
                    )
                else:
                    prediction_results = find_max_price_for_date_range_xgboost(
                        start_date_str=from_date_str,
                        end_date_str=to_date_str,
                        unit_id=unit_id_str,
                        min_value=min_input_value,
                        max_value=max_input_value,
                        step=step_value,
                        target_type=target_type_predict,
                        modelsName=model_section
                    )
            except NameError:
                return jsonify({"status": "error", "message": "XGBoost range function not found."}), 500
        
        elif model_choice == "LinearRegression":
            try:
                if active_version == "v3":
                    prediction_results = find_max_output_for_date_range_linear_model(
                        start_date_str=from_date_str,
                        end_date_str=to_date_str,
                        unit_id=unit_id_str,
                        min_value=min_input_value,
                        max_value=max_input_value,
                        step=step_value,
                        target_type=target_type_predict,
                        modelsName=model_section,
                        weather_data_dict=weather_data_dict
                    )
                else:
                    prediction_results = find_max_output_for_date_range_linear_model(
                        start_date_str=from_date_str,
                        end_date_str=to_date_str,
                        unit_id=unit_id_str,
                        min_value=min_input_value,
                        max_value=max_input_value,
                        step=step_value,
                        target_type=target_type_predict,
                        modelsName=model_section
                    )
            except NameError:
                return jsonify({"status": "error", "message": "Linear Regression range function not found."}), 500

        elif model_choice == "LSTM":
            current_df_166, current_df_167 = ensure_lstm_data_loaded()
            if current_df_166 is None or current_df_167 is None:
                return jsonify({"status": "error", "message": "LSTM data (df_166 or df_167) could not be loaded."}), 500

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            try:
                if active_version == "v3":
                    prediction_results = find_max_output_for_date_range_lstm(
                        start_date_str=from_date_str,
                        end_date_str=to_date_str,
                        unit_id=unit_id_str,
                        df_166=current_df_166, 
                        df_167=current_df_167, 
                        device=device,
                        min_value=min_input_value,
                        max_value=max_input_value,
                        step=step_value,
                        target_type=target_type_predict,
                        modelsName=model_section,
                        weather_data_dict=weather_data_dict
                    )
                else:
                    prediction_results = find_max_output_for_date_range_lstm(
                        start_date_str=from_date_str,
                        end_date_str=to_date_str,
                        unit_id=unit_id_str,
                        df_166=current_df_166, 
                        df_167=current_df_167, 
                        device=device,
                        min_value=min_input_value,
                        max_value=max_input_value,
                        step=step_value,
                        target_type=target_type_predict,
                        modelsName=model_section
                    )
            except NameError:
                return jsonify({"status": "error", "message": "LSTM range function not found."}), 500
        else:
            return jsonify({"status": "error", "message": f"Unsupported model type: {model_choice}"}), 400

        if prediction_results is None:
            return jsonify({"status": "error", "message": "Prediction failed or model returned no results."}), 500
        
        formatted_results = {}
        for date, result in prediction_results.items():
            if isinstance(result, dict):
                best_input = result.get("best_input")
                max_output = result.get("max_output")
                
                if isinstance(best_input, (np.float32, np.float64)):
                    best_input = float(best_input)
                if isinstance(max_output, (np.float32, np.float64)):
                    max_output = float(max_output)
                
                if target_type_predict == "price":
                    input_label = "best_taux_value"
                    output_label = "max_predicted_price"
                    moyenne_label = "price_per_room"
                    
                    room_type = None
                    for room, unit in metadata["PMSInformation"]["room_map"].items():
                        if unit == unit_id_str:
                            room_type = room
                            break
                    
                    if room_type and room_type in metadata["PMSInformation"]["room_count"]:
                        room_count = metadata["PMSInformation"]["room_count"][room_type]
                        average_price = max_output / room_count if room_count > 0 else 0
                    else:
                        average_price = 0
                        print(f"Warning: Could not find room count for unit {unit_id_str}")
                    
                    formatted_results[date] = {
                        input_label: best_input,
                        output_label: max_output,
                        moyenne_label: round(average_price, 2)
                    }
                else:  
                    input_label = "best_price_value"
                    output_label = "max_predicted_taux"
                    
                    formatted_results[date] = {
                        input_label: best_input,
                        output_label: max_output
                    }
            else:
                formatted_results[date] = result
        
        return jsonify({
            "status": "success", 
            "result": formatted_results,
            "model": model_choice,
            "target_type": target_type_predict,
            "unit": unit_id_str,
            "model_version": active_version
        }), 200

    except Exception as e:
        print(f"Error in calculate_predicted: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    


@api_bp.route("/calculate_predicted_single_date", methods=["POST"])
def calculate_predicted_single_date():
    """
    Calculate the predicted values for single date based on the API's config.
    Uses trained models according to active version in metadata.
    """
    try:
        if request.method != "POST":
            return jsonify({"status": "error", "message": "Only POST requests are allowed"}), 405
        
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        required_fields = ["singleDate", "models", "targetPredict", 
                           "targetValueMin", "targetValueMax", "unit"]
        
        for field in required_fields:
            if field not in data:
                return jsonify({"status": "error", "message": f"Missing required field: {field}"}), 400
        
        single_date_str = data["singleDate"]
        model_choice = data["models"]  # "XGBoost", "LSTM", "LinearRegression"
        target_type_predict = data["targetPredict"]  # "price" or "taux"
        min_input_value = float(data["targetValueMin"])
        max_input_value = float(data["targetValueMax"])
        unit_id_str = str(data["unit"])  # unit_id is a string
        
        active_version = metadata.get("model_versions", {}).get("active_version", "v1")
        
        if active_version == "v3":
            model_type_map = {
                "XGBoost": "XGBoostModelsV3",
                "LinearRegression": "LinearRegressionModelsV3",
                "LSTM": "LSTMModelsV3"
            }
        else:
            model_type_map = {
                "XGBoost": "xgboost",
                "LinearRegression": "linear",
                "LSTM": "lstm"
            }
        
        model_type = model_type_map.get(model_choice)
        if not model_type:
            return jsonify({"status": "error", "message": f"Invalid model choice: {model_choice}"}), 400
        
        weather_data_dict = None
        if active_version == "v3":
            try:
                weather_data_dict = load_weather_data_from_json(
                    json_file_path=f"../{metadata['weatherInformation']['archive_json_path']}"
                )
            except Exception as e:
                return jsonify({"status": "error", "message": f"Failed to load weather data: {str(e)}"}), 500
        
        if active_version != "v3":
            try:
                model_section = metadata["model_versions"]["models"][active_version][model_type]
            except KeyError:
                return jsonify({"status": "error", "message": f"Invalid version or model type: {active_version}/{model_type}"}), 400
        else:
            model_section = model_type
        
        step_value = 1.0 
        if (max_input_value - min_input_value) > 1000: 
            step_value = (max_input_value - min_input_value) / 100 
            if target_type_predict == 'taux' and step_value < 0.1:  # For taux, step shouldn't be too small
                step_value = 0.1
            elif target_type_predict == 'price' and step_value < 100:  # For price, step shouldn't be too small
                step_value = 100

        best_input_value = None
        max_predicted_value = None
        
        print(f"Received request: model={model_choice}, target={target_type_predict}, unit={unit_id_str}, "
              f"date={single_date_str}, range=[{min_input_value}-{max_input_value}], step={step_value}, "
              f"using version={active_version}, model_section={model_section}")
        
        if model_choice == "XGBoost":
            try:
                if active_version == "v3":
                    best_input_value, max_predicted_value = find_max_price_for_single_date_xgboost(
                        target_date_str=single_date_str,
                        unit_id=unit_id_str,
                        min_value=min_input_value,
                        max_value=max_input_value,
                        step=step_value,
                        target_type=target_type_predict,
                        modelsName=model_section,
                        weather_data_dict=weather_data_dict
                    )
                else:
                    best_input_value, max_predicted_value = find_max_price_for_date_xgboost(
                        target_date_str=single_date_str,
                        unit_id=unit_id_str,
                        min_value=min_input_value,
                        max_value=max_input_value,
                        step=step_value,
                        target_type=target_type_predict,
                        modelsName=model_section
                    )
            except NameError:
                return jsonify({"status": "error", "message": "XGBoost single date function not found."}), 500
        
        elif model_choice == "LinearRegression":
            try:
                if active_version == "v3":
                    best_input_value, max_predicted_value = find_max_price_for_single_date_linear_model(
                        target_date_str=single_date_str,
                        unit_id=unit_id_str,
                        min_value=min_input_value,
                        max_value=max_input_value,
                        step=step_value,
                        target_type=target_type_predict,
                        modelsName=model_section,
                        weather_data_dict=weather_data_dict
                    )
                else:
                    best_input_value, max_predicted_value = find_max_price_for_date_linear_model(
                        target_date_str=single_date_str,
                        unit_id=unit_id_str,
                        min_value=min_input_value,
                        max_value=max_input_value,
                        step=step_value,
                        target_type=target_type_predict,
                        modelsName=model_section
                    )
            except NameError:
                return jsonify({"status": "error", "message": "Linear Regression single date function not found."}), 500
        
        elif model_choice == "LSTM":
            current_df_166, current_df_167 = ensure_lstm_data_loaded()
            if current_df_166 is None or current_df_167 is None:
                return jsonify({"status": "error", "message": "LSTM data (df_166 or df_167) could not be loaded."}), 500

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            try:
                df_to_use = current_df_166 if unit_id_str == '166' else current_df_167
                
                if active_version == "v3":
                    best_input_value, max_predicted_value = find_max_price_for_single_date_lstm(
                        target_date_str=single_date_str,
                        unit_id=unit_id_str,
                        df_history=df_to_use,
                        device=device,
                        min_value=min_input_value,
                        max_value=max_input_value,
                        step=step_value,
                        target_type=target_type_predict,
                        modelsName=model_section,
                        weather_data_dict=weather_data_dict
                    )
                else:
                    best_input_value, max_predicted_value = find_max_price_for_date_lstm(
                        target_date_str=single_date_str,
                        unit_id=unit_id_str,
                        df_history=df_to_use,
                        device=device,
                        min_value=min_input_value,
                        max_value=max_input_value,
                        step=step_value,
                        target_type=target_type_predict,
                        modelsName=model_section
                    )
            except NameError:
                return jsonify({"status": "error", "message": "LSTM single date function not found."}), 500

        else:
            return jsonify({"status": "error", "message": f"Unsupported model type: {model_choice}"}), 400
        
        if best_input_value is None or max_predicted_value is None:
            return jsonify({"status": "error", "message": "Prediction failed or model returned no results."}), 500
        
        if isinstance(best_input_value, (np.float32, np.float64)):
            best_input_value = float(best_input_value)
        if isinstance(max_predicted_value, (np.float32, np.float64)):
            max_predicted_value = float(max_predicted_value)
        
        if target_type_predict == "price":
            input_label = "best_taux_value"
            output_label = "max_predicted_price"
            moyenne_label = "price_per_room"
            
            room_type = None
            for room, unit in metadata["PMSInformation"]["room_map"].items():
                if unit == unit_id_str:
                    room_type = room
                    break
            
            if room_type and room_type in metadata["PMSInformation"]["room_count"]:
                room_count = metadata["PMSInformation"]["room_count"][room_type]
                average_price = max_predicted_value / room_count if room_count > 0 else 0
            else:
                average_price = 0
                print(f"Warning: Could not find room count for unit {unit_id_str}")
            
            formatted_result = {
                single_date_str: {
                    input_label: best_input_value,
                    output_label: max_predicted_value,
                    moyenne_label: round(average_price, 2)  
                }
            }
        else:  
            input_label = "best_price_value"
            output_label = "max_predicted_taux"
            
            formatted_result = {
                single_date_str: {
                    input_label: best_input_value,
                    output_label: max_predicted_value
                }
            }
        
        return jsonify({
            "status": "success", 
            "result": formatted_result,
            "model": model_choice,
            "target_type": target_type_predict,
            "unit": unit_id_str,
            "model_version": active_version  
        }), 200
    
    except Exception as e:
        print(f"Error in calculate_predicted_single_date: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500