from flask import Flask, request, jsonify
import torch
import numpy as np
import joblib

# Define the PyTorch model
class DietaryRecommendationModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DietaryRecommendationModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)

# Load the pre-trained model
input_dim = 54  # Adjust based on the dataset features
hidden_dim = 128
output_dim = 3
model = DietaryRecommendationModel(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("dietary_model.pth"))
model.eval()

# Load the scaler used during preprocessing
scaler = joblib.load("scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        # Parse input JSON
        input_data = request.get_json()

        # Extract features from the input data
        features = np.array(input_data["features"]).reshape(1, -1)

        # Scale the features using the saved scaler
        scaled_features = scaler.transform(features)

        # Convert scaled features to PyTorch tensor
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32)

        # Make predictions using the model
        with torch.no_grad():
            predictions = model(input_tensor)
            probabilities = predictions.numpy().tolist()[0]

        # Create human-readable recommendations
        recommendations = {
            "Dietary Fiber": f"{probabilities[0]*100:.2f}%",
            "Prebiotics": f"{probabilities[1]*100:.2f}%",
            "Probiotics": f"{probabilities[2]*100:.2f}%"
        }

        return jsonify({
            "success": True,
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

