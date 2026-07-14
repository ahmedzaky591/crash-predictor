from flask import Flask, render_template, request, jsonify
import json
from predictor import CrashPredictor
import os

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize predictor
predictor = CrashPredictor()

# Load or generate sample data
SAMPLE_DATA_FILE = 'sample_crashes.json'


def load_sample_crashes():
    """Load or create sample crash data."""
    if os.path.exists(SAMPLE_DATA_FILE):
        with open(SAMPLE_DATA_FILE, 'r') as f:
            return json.load(f)
    else:
        # Generate sample data
        sample = [1.5, 2.3, 1.8, 3.2, 2.1, 1.9, 2.8, 1.6, 2.5, 3.1, 
                  2.2, 1.7, 2.9, 2.0, 1.4, 3.5, 2.4, 1.9, 2.6, 2.8,
                  1.5, 2.1, 3.0, 1.8, 2.5, 2.2, 1.9, 2.7, 2.0, 1.6]
        with open(SAMPLE_DATA_FILE, 'w') as f:
            json.dump(sample, f)
        return sample


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/api/train', methods=['POST'])
def train():
    """Train the model with provided crash data."""
    try:
        data = request.json
        crashes = data.get('crashes', [])
        
        if not crashes or len(crashes) < 10:
            return jsonify({
                'success': False,
                'error': 'Need at least 10 crash data points'
            }), 400
        
        result = predictor.train(crashes)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict next crash."""
    try:
        data = request.json
        crashes = data.get('crashes', [])
        
        if not crashes or len(crashes) < 5:
            return jsonify({
                'success': False,
                'error': 'Need at least 5 crash data points to predict'
            }), 400
        
        # Train if not already trained
        if not predictor.is_trained:
            predictor.train(crashes)
        
        prediction = predictor.predict(crashes)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'data': {
                'multiplier': prediction['rounded'],
                'confidence': f"{prediction['confidence']:.2%}"
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Generate multiple predictions in advance."""
    try:
        data = request.json
        crashes = data.get('crashes', [])
        num_predictions = data.get('num_predictions', 5)
        
        if not crashes or len(crashes) < 5:
            return jsonify({
                'success': False,
                'error': 'Need at least 5 crash data points'
            }), 400
        
        if not predictor.is_trained:
            predictor.train(crashes)
        
        predictions = predictor.predict_batch(crashes, num_predictions)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/sample-data', methods=['GET'])
def get_sample_data():
    """Get sample crash data."""
    try:
        sample = load_sample_crashes()
        return jsonify({
            'success': True,
            'crashes': sample,
            'count': len(sample)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats', methods=['POST'])
def get_stats():
    """Get statistics about the crash data."""
    try:
        data = request.json
        crashes = data.get('crashes', [])
        
        if not crashes:
            return jsonify({
                'success': False,
                'error': 'No crash data provided'
            }), 400
        
        import numpy as np
        crashes_arr = np.array(crashes, dtype=float)
        
        stats = {
            'total_crashes': len(crashes),
            'average': float(np.mean(crashes_arr)),
            'min': float(np.min(crashes_arr)),
            'max': float(np.max(crashes_arr)),
            'std_dev': float(np.std(crashes_arr)),
            'median': float(np.median(crashes_arr))
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_trained': predictor.is_trained,
        'version': '1.0.0'
    })


if __name__ == '__main__':
    # Load sample data on startup
    load_sample_crashes()
    app.run(debug=True, port=5000, host='0.0.0.0')
