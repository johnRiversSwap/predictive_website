from flask import Flask, request, render_template, jsonify, send_from_directory
import joblib
import pandas as pd
import os

app = Flask(__name__)
app.env = 'development'

model = None  # Initialize model as None

def load_model():
    global model
    try:
        directory = os.path.dirname(__file__)  # Get the directory where the script is located
        path_to_model = os.path.join(directory, 'model_rf.joblib')
        model = joblib.load(path_to_model)
        print("Model loaded successfully.")
    except Exception as e:
        model = None
        print(f"Failed to load model: {e}")

load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from POST request
        data = request.get_json()
        print(f"Received data for prediction: {data}")
        
        # Assume data is in a format directly consumable by the model
        df = pd.DataFrame([data])
        
        prediction = model.predict(df)
        
        # Log the prediction
        print(f"Prediction made: {prediction.tolist()}")
        
        # You can format the output as needed; here, we just return the prediction
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)})

# Route to serve the JSON files
@app.route('/data/<filename>')
def serve_json(filename):
    try:
        return send_from_directory('data', f'{filename}.json')
    except Exception as e:
        print(f"Error serving JSON file {filename}: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
