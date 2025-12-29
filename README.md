# Anomaly Detection Agent

An intelligent anomaly detection system for time series data, powered by machine learning and AI explanations. This project provides a comprehensive dashboard for uploading CSV data, detecting anomalies, visualizing trends, forecasting, and generating actionable insights using Google's Gemini AI.

## Features

### Backend (FastAPI)
- **Anomaly Detection**: Uses Isolation Forest algorithm to detect anomalies in time series data
- **Severity Classification**: Automatically classifies anomalies as Low, Medium, High, or Critical
- **Root Cause Ranking**: Provides ranked potential causes for detected anomalies
- **AI Explanations**: Generates detailed explanations and recommendations using Gemini AI
- **Chat Interface**: Interactive chatbot for querying dataset insights

### Frontend (Streamlit)
- **CSV Upload**: Easy file upload interface
- **Anomaly Visualization**: Interactive graphs showing detected anomalies
- **Forecasting**: Time series forecasting with Holt-Winters method
- **Trend Decomposition**: Break down time series into trend, seasonality, and residual components
- **Custom Severity Builder**: Adjustable thresholds for anomaly detection
- **PDF Export**: Generate and download anomaly reports
- **Gemini Chat**: Integrated AI chatbot for data analysis questions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/2300032509/anomaly-detection-agent.git
cd anomaly-detection-agent
```

2. Create a virtual environment:
```bash
python -m venv venv2
venv2\Scripts\activate  # On Windows
# or
source venv2/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=models/gemini-2.5-flash
```

## Usage

### Running the Backend
```bash
python main.py
```
The FastAPI server will start on `http://127.0.0.1:8000`

### Running the Frontend
```bash
cd frontend
streamlit run app.py
```
The Streamlit app will open in your browser.

### API Endpoints

#### POST /analyze
Upload a CSV file and detect anomalies.

**Parameters:**
- `file`: CSV file upload
- `time_col`: Name of the time column
- `metric_col`: Name of the metric column to analyze

**Response:**
```json
{
  "anomalies": [
    {
      "timestamp": "2023-01-01",
      "value": 100.5,
      "rolling_mean": 95.2,
      "residual": 5.3,
      "score": 0.85,
      "severity": "High",
      "root_causes_ranked": [["Data Entry Error", 0.92], ...],
      "explanation": "Detailed AI explanation..."
    }
  ]
}
```

#### POST /chat
Ask questions about the dataset using AI.

**Parameters:**
```json
{
  "question": "What are the main trends in this data?",
  "dataset_summary": "Summary of detected anomalies..."
}
```

## Requirements

- Python 3.8+
- FastAPI
- Streamlit
- pandas
- scikit-learn
- google-genai
- matplotlib
- requests
- python-dotenv
- fpdf
- statsmodels (optional, for advanced forecasting)

## Project Structure

```
anomaly-agent-demo/
├── main.py                 # FastAPI backend
├── frontend/
│   └── app.py             # Streamlit frontend
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in repo)
├── .gitignore            # Git ignore rules
├── input.json            # Sample input data
├── output.json           # Sample output data
├── metadata.json         # Metadata
└── README.md             # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - feel free to use this project for your own purposes.

## Support

For issues or questions, please open a GitHub issue in this repository.