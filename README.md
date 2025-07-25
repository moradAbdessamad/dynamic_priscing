# Dynamic Pricing Hotel Revenue Management System

A comprehensive machine learning-powered hotel revenue management system that predicts optimal pricing and occupancy rates using real-time data from Property Management Systems (PMS), weather APIs, and event data.

## 🏨 Overview

This project implements an intelligent dynamic pricing system for hotels, specifically designed for hotel revenue optimization. The system integrates multiple data sources and uses advanced machine learning algorithms to predict and optimize room pricing, occupancy rates, and revenue management strategies.

### Key Features

- **Multi-Model ML Pipeline**: Supports XGBoost, Linear Regression, and LSTM models for price and occupancy prediction
- **Real-time PMS Integration**: Connects to Fractal Stay PMS API for live reservation data
- **Weather Integration**: Uses Open-Meteo API for weather-based pricing adjustments
- **Event-Based Pricing**: Integrates PredictHQ events API for demand forecasting
- **Web Application**: Full-stack solution with Flask backend and Angular frontend
- **Multi-Room Type Support**: Handles different room types (Standard, Family, Imaginary rooms)
- **Customer Segment Analysis**: Revenue analysis by customer segments (AVM, TO, OTA, etc.)

## 🏗️ Architecture

### Backend Components
- **Flask API Server**: RESTful API endpoints for predictions and data processing
- **Machine Learning Models**: Three model types with multiple versions (v1, v2, v3)
  - XGBoost Models for price/occupancy prediction
  - Linear Regression for baseline predictions
  - LSTM for time-series forecasting
- **Data Processing Pipeline**: Automated ETL for PMS, weather, and event data
- **Model Training System**: Jupyter notebooks for model development and retraining

### Frontend Components
- **Angular Application**: Modern web interface for hotel management
- **Dynamic Dashboard**: Real-time visualization of pricing recommendations
- **Filtering System**: Advanced filters for date ranges, room types, and segments
- **Prediction Results Display**: Interactive tables and charts for decision support

## 📊 Data Sources

1. **PMS Data (Fractal Stay API)**
   - Real-time reservation data
   - Room occupancy and availability
   - Revenue and pricing history
   - Customer segments and demographics

2. **Weather Data (Open-Meteo API)**
   - Temperature, precipitation, wind speed
   - Historical and forecast data
   - Impact on demand patterns

3. **Events Data (PredictHQ API)**
   - Local events and their impact on demand
   - Event categorization and severity

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Node.js for Angular frontend
node --version
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/moradAbdessamad/dynamic_priscing.git
cd PFEVersion2
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Angular dependencies**
```bash
cd SRC/front-end/angular-app
npm install
```

4. **Environment Setup**
```bash
# Create .env file in SRC/back-end/
NGROK_TOKEN=your_ngrok_token_here
```

### Configuration

Update the metadata configuration in `SRC/metaData/modelMetaData.json`:

```json
{
  "PMSInformation": {
    "api_templates": ["your_pms_api_endpoints"],
    "total_rooms": 432,
    "room_count": {
      "Standard Room": 387,
      "Family Room": 18,
      "Imaginary Room": 67
    }
  }
}
```

## 🔧 Usage

### Running the Backend

```bash
cd SRC/back-end
python run.py
```

The Flask server will start on port 5300 with ngrok tunnel for external access.

### Running the Frontend

```bash
cd SRC/front-end/angular-app
ng serve
```

The Angular app will be available at `http://localhost:4200`

### API Endpoints

- `POST /calculate_predicted` - Get pricing predictions for date ranges
- `POST /calculate_predicted_single_date` - Single date prediction
- `POST /calculate_true_price` - Calculate actual metrics from PMS data
- `GET /webhooks` - Webhook endpoint for real-time updates

### Making Predictions

Example API call for price prediction:

```python
import requests

data = {
    "unit_id": "166",
    "target_date": "2025-07-25",
    "input_value": 75.0,
    "target_type": "price",
    "model_type": "xgboost"
}

response = requests.post("http://localhost:5300/calculate_predicted_single_date", json=data)
result = response.json()
```

## 📁 Project Structure

```
PFEVersion2/
├── SRC/                          # Source code
│   ├── back-end/                 # Flask API server
│   │   ├── app/                  # Application modules
│   │   └── run.py               # Server entry point
│   ├── front-end/               # Angular application
│   │   └── angular-app/         # Angular project
│   ├── helpFolder/              # ML helper functions
│   │   ├── helpersLR.py         # Linear regression utilities
│   │   ├── helpersXGB.py        # XGBoost utilities
│   │   ├── helpersLSTM.py       # LSTM utilities
│   │   └── helpersEtractPMS.py  # PMS data processing
│   ├── Training/                # Model training notebooks
│   └── TrainedModels/           # Saved ML models
├── Data/                        # Data storage
│   ├── PMSEtractedData/         # PMS reservation data
│   ├── WeatherData/             # Weather information
│   ├── SegmentMetricsData/      # Customer segment analytics
│   └── TransformedData/         # Processed datasets
├── Extract/                     # Data extraction notebooks
├── Transforme/                  # Data transformation scripts
└── Config/                      # Configuration files
```

## 🤖 Machine Learning Models

### Model Types and Versions

1. **XGBoost Models (v1, v2, v3)**
   - Price prediction: Units 166, 167
   - Occupancy rate prediction
   - Feature importance analysis

2. **Linear Regression Models (v1, v2, v3)**
   - Baseline predictions
   - Feature interpretability
   - Fast inference

3. **LSTM Models (v1, v2, v3)**
   - Time-series forecasting
   - Sequential pattern learning
   - Long-term trend prediction

### Feature Engineering

- **Temporal Features**: Day of week, month, year, day of year
- **Occupancy Features**: Current and historical occupancy rates
- **Weather Features**: Temperature, precipitation, wind speed
- **Event Features**: Local events impact scores
- **Seasonal Features**: Holiday and peak season indicators

## 📈 Performance Metrics

The system tracks multiple performance indicators:

- **Revenue Metrics**: Total revenue, RevPAR, ADR
- **Occupancy Metrics**: Occupancy percentage, available rooms
- **Prediction Accuracy**: RMSE, MAE, R² scores
- **Customer Metrics**: Segment-wise performance analysis

## 🔄 Data Pipeline

1. **Data Extraction**: Automated fetching from PMS, weather, and events APIs
2. **Data Transformation**: Cleaning, feature engineering, and aggregation
3. **Model Training**: Automated retraining with new data
4. **Prediction Generation**: Real-time pricing recommendations
5. **Performance Monitoring**: Model accuracy tracking and alerts

## 📱 Web Interface Features

- **Real-time Dashboard**: Live pricing and occupancy data
- **Prediction Interface**: Easy-to-use prediction tools
- **Historical Analysis**: Trend analysis and reporting
- **Multi-room Management**: Handle different room types
- **Segment Analysis**: Customer segment performance tracking

## 🛠️ Development

### Adding New Models

1. Create model training notebook in `SRC/Training/`
2. Implement helper functions in `SRC/helpFolder/`
3. Update metadata configuration
4. Add API endpoints in `SRC/back-end/app/blueprints/api.py`

### Running Tests

```bash
# Run data extraction tests
cd atest/
python testOne.py

# Test API endpoints
python -m pytest tests/
```

## 📚 Dependencies

### Python Backend
- Flask, Flask-CORS
- scikit-learn, XGBoost
- PyTorch (for LSTM)
- pandas, numpy
- requests, python-dotenv

### Angular Frontend
- Angular 19.x
- TypeScript
- RxJS for reactive programming

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is part of a Master's thesis (PFE - Projet de Fin d'Études) for dynamic pricing in hotel revenue management.

## 👥 Authors

- **Morad Abdessamad** - [@moradAbdessamad](https://github.com/moradAbdessamad)

## 🙏 Acknowledgments

- Fractal Stay for PMS API access
- Open-Meteo for weather data
- PredictHQ for events data
- Academic supervisors and reviewers

---

**Note**: This system is designed for educational and research purposes as part of a Master's degree project. For production use, ensure proper security measures and API rate limiting are implemented.