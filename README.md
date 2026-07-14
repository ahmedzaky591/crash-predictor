# Crash Predictor 🎯

A machine learning-based tool that predicts the next crash point in crash games based on historical crash data.

## Features

- 🔮 **Predict next crash multiplier** from historical data
- 📊 **Pattern analysis** using time-series forecasting
- 🎨 **Web UI** for easy testing
- 📈 **Training on historical crashes** to improve predictions
- ⚡ **Fast predictions** with scikit-learn

## How it Works

1. **Input**: Provide historical crash multipliers (e.g., `[1.5, 2.3, 1.8, 3.2, 2.1, ...]`)
2. **Process**: The ML model analyzes patterns and trends
3. **Output**: Prediction for the next crash multiplier with confidence score

## Installation

```bash
git clone https://github.com/ahmedzaky591/crash-predictor.git
cd crash-predictor
pip install -r requirements.txt
```

## Usage

### Python API

```python
from predictor import CrashPredictor

# Initialize predictor
predictor = CrashPredictor()

# Historical crash data
crashes = [1.5, 2.3, 1.8, 3.2, 2.1, 1.9, 2.8, 1.6, 2.5, 3.1]

# Get prediction
prediction = predictor.predict(crashes)
print(f"Next crash at: {prediction['multiplier']:.2f}x")
print(f"Confidence: {prediction['confidence']:.2%}")
```

### Web Interface

```bash
python app.py
# Open http://localhost:5000
```

## Project Structure

```
crash-predictor/
├── predictor.py          # Core prediction engine
├── app.py                # Flask web app
├── requirements.txt      # Dependencies
├── templates/
│   └── index.html        # Web UI
└── sample_crashes.json   # Sample training data
```

## Technologies

- **scikit-learn**: Machine learning
- **numpy/pandas**: Data processing
- **Flask**: Web framework
- **Chart.js**: Data visualization

## License

MIT
