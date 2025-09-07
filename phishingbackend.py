import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from datetime import datetime
import re
import urllib.parse

app = Flask(__name__)
CORS(app)

# Global variables for statistics
prediction_stats = {"safe": 0, "phishing": 0}
prediction_history = []


class SimpleKagglePhishingDetector:
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.model_trained = False

    def load_kaggle_dataset(self, csv_path='phishing_dataset.csv'):
        """Load the Kaggle phishing dataset"""
        try:
            print(f"📊 Loading Kaggle dataset from {csv_path}...")

            # You can download from: https://www.kaggle.com/datasets/akashkr/phishing-website-dataset
            if not os.path.exists(csv_path):
                print("❌ Dataset file not found!")
                print("Please download the Kaggle dataset:")
                print("1. Go to: https://www.kaggle.com/datasets/akashkr/phishing-website-dataset")
                print("2. Download 'phishing_dataset.csv'")
                print("3. Place it in the same folder as this script")
                return None

            df = pd.read_csv(csv_path)
            print(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns")

            # Drop index column if exists
            if 'index' in df.columns:
                df = df.drop(columns=['index'])
            df['Result'] = df['Result'].map({1: 0, -1: 1})  # 0 = safe, 1 = phishing

            print(f"📈 Dataset info:")
            print(f"   - Phishing URLs: {len(df[df['Result'] == 1])}")
            print(f"   - Legitimate URLs: {len(df[df['Result'] == 0])}")
            print(f"   - Features: {len(df.columns) - 1}")

            return df

        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None

    def train_with_kaggle_data(self, csv_path='phishing_dataset.csv'):
        """Train model with Kaggle dataset - Super Simple!"""
        print("🚀 Starting model training with Kaggle data...")

        # Load dataset
        df = self.load_kaggle_dataset(csv_path)
        if df is None:
            return False

        # Prepare features and labels
        # The Kaggle dataset has 'class' column (0=legitimate, 1=phishing)
        # Prepare features and labels
        X = df.drop('Result', axis=1)  # All columns except 'Result'
        y = df['Result']  # Target column

        self.feature_names = list(X.columns)
        print(f"🔢 Features: {self.feature_names[:5]}... (showing first 5)")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"📚 Training set: {len(X_train)} samples")
        print(f"📝 Test set: {len(X_test)} samples")

        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        print("🤖 Training Random Forest model...")
        self.model.fit(X_train, y_train)

        # Evaluate model
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)

        print(f"✅ Training accuracy: {train_accuracy:.4f}")
        print(f"✅ Test accuracy: {test_accuracy:.4f}")

        # Detailed evaluation
        y_pred = self.model.predict(X_test)
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

        # Feature importance
        feature_importance = list(zip(self.feature_names, self.model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        print("\n🔝 Top 10 Most Important Features:")
        for feature, importance in feature_importance[:10]:
            print(f"   {feature}: {importance:.4f}")

        # Save model
        model_path = 'kaggle_phishing_model.joblib'
        joblib.dump(self.model, model_path)
        print(f"💾 Model saved as {model_path}")

        self.model_trained = True

        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'feature_importance': feature_importance[:10]
        }

    def extract_simple_features(self, url):
        """
        Extract features from URL to match Kaggle dataset format
        Note: This is a simplified version. The Kaggle dataset has pre-calculated features.
        """
        features = {}

        try:
            # Parse URL
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url

            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path

            # Basic features (simplified to match some common Kaggle dataset features)
            features['having_IP_Address'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain) else 0
            features['URL_Length'] = 1 if len(url) > 75 else 0
            features['Shortining_Service'] = 1 if any(short in domain for short in ['bit.ly', 'tinyurl', 't.co']) else 0
            features['having_At_Symbol'] = 1 if '@' in url else 0
            features['double_slash_redirecting'] = 1 if '//' in path else 0
            features['Prefix_Suffix'] = 1 if '-' in domain else 0
            features['having_Sub_Domain'] = min(domain.count('.') - 1, 2) if '.' in domain else 0
            features['SSLfinal_State'] = 1 if url.startswith('https://') else 0
            features['Domain_registeration_length'] = 0  # Would need WHOIS data
            features['Favicon'] = 0  # Would need to check favicon
            features['port'] = 1 if ':' in domain and any(c.isdigit() for c in domain.split(':')[-1]) else 0
            features['HTTPS_token'] = 1 if 'https' in domain else 0

            # Add more features with default values (Kaggle dataset has 30 features)
            default_features = [
                'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email',
                'Abnormal_URL', 'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow',
                'Iframe', 'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank',
                'Google_Index', 'Links_pointing_to_page', 'Statistical_report'
            ]

            for feature in default_features:
                if feature not in features:
                    features[feature] = 0

        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return all zeros if extraction fails
            all_features = [
                'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
                'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
                'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
                'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
                'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
                'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
                'Statistical_report'
            ]
            features = {name: 0 for name in all_features}

        return features

    def predict(self, url):
        """Predict if URL is phishing"""
        if not self.model_trained:
            raise ValueError("Model not trained. Please train the model first.")

        # Extract features
        features = self.extract_simple_features(url)

        # Convert to array in correct order
        feature_vector = [features.get(name, 0) for name in self.feature_names]
        feature_array = np.array(feature_vector).reshape(1, -1)

        # Make prediction
        prediction = self.model.predict(feature_array)[0]
        probabilities = self.model.predict_proba(feature_array)[0]

        return {
            'prediction': int(prediction),
            'phishing_probability': float(probabilities[1]),
            'legitimate_probability': float(probabilities[0]),
            'confidence': float(max(probabilities)),
            'features': features
        }


# Initialize detector
detector = SimpleKagglePhishingDetector()


@app.route('/')
def home():
    return jsonify({
        "message": "Simple Kaggle Phishing Detection API",
        "version": "1.0 - Beginner Friendly",
        "model_trained": detector.model_trained,
        "dataset": "Kaggle Phishing Website Dataset",
        "endpoints": {
            "/predict": "POST - Analyze URL for phishing",
            "/stats": "GET - Get prediction statistics",
            "/train": "POST - Train model (requires dataset file)",
            "/health": "GET - Health check"
        }
    })


@app.route('/predict', methods=['POST'])
def predict_url():
   @app.route('/predict', methods=['POST'])
def predict_url():
    try:
        data = request.get_json()
        url = data.get('url', '').strip()

        if not url:
            return jsonify({'error': 'URL is required'}), 400

        # ✅ Auto-train if model not trained yet
        if not detector.model_trained:
            csv_path = 'phishing_dataset.csv'
            if not os.path.exists(csv_path):
                return jsonify({
                    'error': 'Dataset not found. Please download from Kaggle.',
                    'instructions': 'https://www.kaggle.com/datasets/akashkr/phishing-website-dataset'
                }), 503

            # Train the model automatically
            detector.train_with_kaggle_data(csv_path)

        # Make prediction
        result = detector.predict(url)

        # Update statistics
        global prediction_stats, prediction_history
        if result['prediction'] == 1:
            prediction_stats['phishing'] += 1
            label = 'phishing'
        else:
            prediction_stats['safe'] += 1
            label = 'safe'

        # Add to history
        prediction_history.append({
            'url': url,
            'prediction': label,
            'confidence': result['confidence'],
            'timestamp': datetime.now().isoformat()
        })

        # Keep only last 100 predictions
        if len(prediction_history) > 100:
            prediction_history.pop(0)

        # Prepare response
        response = {
            'url': url,
            'is_phishing': result['prediction'] == 1,
            'confidence': result['confidence'],
            'phishing_probability': result['phishing_probability'],
            'legitimate_probability': result['legitimate_probability'],
            'label': label,
            'explanation': {
                'has_ip_address': bool(result['features'].get('having_IP_Address', 0)),
                'long_url': bool(result['features'].get('URL_Length', 0)),
                'uses_shortener': bool(result['features'].get('Shortining_Service', 0)),
                'has_at_symbol': bool(result['features'].get('having_At_Symbol', 0)),
                'has_redirects': bool(result['features'].get('double_slash_redirecting', 0)),
                'suspicious_domain': bool(result['features'].get('Prefix_Suffix', 0)),
                'is_https': bool(result['features'].get('SSLfinal_State', 0)),
                'subdomain_count': result['features'].get('having_Sub_Domain', 0)
            },
            'risk_factors': []
        }

        # Generate risk factors
        features = result['features']
        if features.get('having_IP_Address', 0):
            response['risk_factors'].append('Uses IP address instead of domain name')
        if features.get('URL_Length', 0):
            response['risk_factors'].append('Unusually long URL')
        if features.get('Shortining_Service', 0):
            response['risk_factors'].append('Uses URL shortener service')
        if features.get('having_At_Symbol', 0):
            response['risk_factors'].append('Contains @ symbol (suspicious)')
        if features.get('double_slash_redirecting', 0):
            response['risk_factors'].append('Contains suspicious redirect pattern')
        if features.get('Prefix_Suffix', 0):
            response['risk_factors'].append('Domain contains suspicious characters')
        if not features.get('SSLfinal_State', 0):
            response['risk_factors'].append('Not using secure HTTPS protocol')
        if features.get('having_Sub_Domain', 0) >= 2:
            response['risk_factors'].append('Too many subdomains')

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500



@app.route('/train', methods=['POST'])
def train_model():
    """Train model with Kaggle dataset"""
    try:
        print("🎓 Starting model training...")
        result = detector.train_with_kaggle_data()

        if result:
            return jsonify({
                'status': 'success',
                'message': 'Model trained successfully with Kaggle dataset!',
                'results': {
                    'train_accuracy': result['train_accuracy'],
                    'test_accuracy': result['test_accuracy'],
                    'top_features': [(name, round(importance, 4)) for name, importance in result['feature_importance']]
                }
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Training failed. Please check if dataset file exists.'
            }), 500

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Training error: {str(e)}'
        }), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    total = prediction_stats['safe'] + prediction_stats['phishing']

    return jsonify({
        'total_predictions': total,
        'safe_count': prediction_stats['safe'],
        'phishing_count': prediction_stats['phishing'],
        'safe_percentage': round((prediction_stats['safe'] / total * 100) if total > 0 else 0, 2),
        'phishing_percentage': round((prediction_stats['phishing'] / total * 100) if total > 0 else 0, 2),
        'model_trained': detector.model_trained,
        'dataset_source': 'Kaggle Phishing Website Dataset',
        'recent_predictions': prediction_history[-10:]
    })


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_trained': detector.model_trained,
        'dataset': 'Kaggle',
        'uptime': datetime.now().isoformat()
    })


if __name__ == "__main__":
    import os

    print("🚀 Starting Simple Kaggle Phishing Detection API...")
    print("📚 This version uses the Kaggle Phishing Website Dataset")
    print("=" * 60)

    # Try to train model automatically if dataset exists
    if os.path.exists('phishing_dataset.csv'):
        print("📊 Found dataset file, training model...")
        result = detector.train_with_kaggle_data()
        if result:
            print("✅ Model trained successfully!")
        else:
            print("❌ Automatic training failed")
    else:
        print("📥 Dataset not found. Please download from Kaggle:")
        print("   https://www.kaggle.com/datasets/akashkr/phishing-website-dataset")
        print("   Then call /train endpoint or restart the server")

    # Start Flask server
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🌐 Starting Flask server on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
