import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib

# Load the dataset
file_path = "microbiomebalance_optimized_dataset.csv"
data = pd.read_csv(file_path)

# Preprocessing
# Encode categorical variables
le_gender = LabelEncoder()
le_health_status = LabelEncoder()

data["Gender"] = le_gender.fit_transform(data["Gender"])
data["Health_Status"] = le_health_status.fit_transform(data["Health_Status"])

# Select features and target
microbial_features = [col for col in data.columns if "Taxon_" in col]
metadata_features = ["Dietary_Fiber_Intake", "Probiotic_Use", "Age", "Gender"]
target_feature = "Health_Status"

X = data[microbial_features + metadata_features]
y = data[target_feature]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Define the DNN model
class DietaryRecommendationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DietaryRecommendationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)

# Model parameters
input_dim = X_train.shape[1]
hidden_dim = 128
output_dim = len(data[target_feature].unique())

model = DietaryRecommendationModel(input_dim, hidden_dim, output_dim)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 30
batch_size = 64

for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i + batch_size]
        y_batch = y_train_tensor[i:i + batch_size]

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test_tensor)
    y_pred = torch.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=le_health_status.classes_))
torch.save(model.state_dict(), "dietary_model.pth")
joblib.dump(scaler, "scaler.pkl")
