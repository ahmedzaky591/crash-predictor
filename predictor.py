import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path


class CrashPredictor:
    """
    Machine learning-based crash predictor.
    Analyzes historical crash multipliers to predict the next crash point.
    """
    
    def __init__(self, model_path='model.pkl'):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.is_trained = False
        self.min_samples = 10
        
    def create_features(self, crashes, lookback=5):
        """
        Create features from crash history using a sliding window approach.
        
        Args:
            crashes: List of crash multipliers
            lookback: Number of previous crashes to use as features
            
        Returns:
            X (features), y (targets)
        """
        crashes = np.array(crashes, dtype=float)
        X, y = [], []
        
        for i in range(lookback, len(crashes)):
            # Features: last 'lookback' crashes
            X.append(crashes[i-lookback:i])
            # Target: next crash
            y.append(crashes[i])
        
        return np.array(X), np.array(y)
    
    def train(self, crashes):
        """
        Train the model on historical crash data.
        
        Args:
            crashes: List of historical crash multipliers
            
        Returns:
            Training metrics
        """
        if len(crashes) < self.min_samples:
            raise ValueError(f"Need at least {self.min_samples} crash data points to train")
        
        X, y = self.create_features(crashes)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate training score
        score = self.model.score(X_scaled, y)
        
        return {
            'success': True,
            'samples_trained': len(crashes),
            'r2_score': float(score),
            'message': f'Model trained on {len(crashes)} crash data points'
        }
    
    def predict(self, crashes, return_confidence=True):
        """
        Predict the next crash multiplier.
        
        Args:
            crashes: List of recent crash multipliers (at least 5)
            return_confidence: Whether to return confidence metric
            
        Returns:
            Prediction dict with multiplier and confidence
        """
        if not self.is_trained:
            # Auto-train if not already trained
            self.train(crashes)
        
        if len(crashes) < 5:
            raise ValueError("Need at least 5 crash data points to predict")
        
        # Use last 5 crashes as features
        recent_crashes = np.array(crashes[-5:], dtype=float).reshape(1, -1)
        recent_scaled = self.scaler.transform(recent_crashes)
        
        # Make prediction
        prediction = self.model.predict(recent_scaled)[0]
        
        # Ensure prediction is positive and reasonable
        prediction = max(1.01, float(prediction))
        
        result = {
            'multiplier': prediction,
            'rounded': round(prediction, 2)
        }
        
        if return_confidence:
            # Confidence based on model's prediction uncertainty
            predictions_ensemble = []
            for estimator in self.model.estimators_:
                pred = estimator.predict(recent_scaled)[0]
                predictions_ensemble.append(pred)
            
            std_dev = np.std(predictions_ensemble)
            # Convert std to confidence (lower std = higher confidence)
            confidence = max(0.5, 1.0 - (std_dev / prediction))
            result['confidence'] = float(confidence)
        
        return result
    
    def predict_batch(self, crashes, num_predictions=5):
        """
        Generate multiple predictions in advance.
        
        Args:
            crashes: List of recent crash multipliers
            num_predictions: Number of predictions to generate
            
        Returns:
            List of predictions
        """
        predictions = []
        current_crashes = list(crashes[-5:])  # Start with last 5
        
        for i in range(num_predictions):
            pred = self.predict(current_crashes)
            predictions.append(pred['rounded'])
            # Add prediction to history for next iteration
            current_crashes.append(pred['rounded'])
            current_crashes = current_crashes[-5:]  # Keep only last 5
        
        return predictions
    
    def save_model(self):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        import pickle
        with open(self.model_path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
        return f"Model saved to {self.model_path}"
    
    def load_model(self):
        """Load trained model from disk."""
        import pickle
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.is_trained = True
            return "Model loaded successfully"
        except FileNotFoundError:
            return "No saved model found"


# Utility functions
def load_crash_data(filepath):
    """Load crash data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_crash_data(crashes, filepath):
    """Save crash data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(crashes, f)


if __name__ == "__main__":
    # Example usage
    sample_crashes = [1.5, 2.3, 1.8, 3.2, 2.1, 1.9, 2.8, 1.6, 2.5, 3.1, 
                      2.2, 1.7, 2.9, 2.0, 1.4, 3.5, 2.4, 1.9, 2.6, 2.8]
    
    predictor = CrashPredictor()
    
    # Train
    print("Training model...")
    result = predictor.train(sample_crashes)
    print(result)
    
    # Predict next crash
    print("\nPredicting next crash...")
    prediction = predictor.predict(sample_crashes)
    print(f"Next crash: {prediction['rounded']}x")
    print(f"Confidence: {prediction['confidence']:.2%}")
    
    # Batch predictions
    print("\nBatch predictions (next 5 crashes):")
    batch = predictor.predict_batch(sample_crashes, 5)
    for i, pred in enumerate(batch, 1):
        print(f"  Crash {i}: {pred}x")
