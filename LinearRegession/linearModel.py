import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Linear model prediction function: y = Xw + b
def predict(X):
    return torch.matmul(X, w) + b

# Mean Squared Error loss function
def mse_loss(prediction, target):
    return torch.mean((prediction - target) ** 2)

# Load dataset from CSV file
data = pd.read_csv('Concrete_Data.csv')

# Separate input features (all columns except last)
inputs = data.iloc[:, :-1]

# Separate target variable (last column)
target = data.iloc[:, -1]

# Standardize input features (zero mean, unit variance)
scaler = StandardScaler()
inputs_scaled = scaler.fit_transform(inputs)

# Convert inputs to PyTorch tensor
inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)

# Convert target values to PyTorch tensor
target_tensor = torch.tensor(target.values, dtype=torch.float32)

# Total number of samples
n_values = inputs_tensor.shape[0]

# Generate a random permutation of indices
indices = torch.randperm(n_values)

# Determine split index for 80% training, 20% testing
split_index = int(n_values * 0.8)

# Split indices into training and testing sets
train_indices = indices[:split_index]
test_indices = indices[split_index:]

# Create training data
x_train = inputs_tensor[train_indices]
y_train = target_tensor[train_indices]

# Create testing data
x_test = inputs_tensor[test_indices]
y_test = target_tensor[test_indices]

# Number of input features
number_of_features = x_train.shape[1]

# Initialize weights with gradient tracking enabled
w = torch.randn((number_of_features, 1), requires_grad=True)

# Initialize bias with gradient tracking enabled
b = torch.randn((1,), requires_grad=True)

# Learning rate for gradient descent
learning_rate = 0.01

# Number of training iterations
epochs = 500

# List to store training loss values
train_losses = []

# Training loop
for epoch in range(epochs):

    # Forward pass: compute predictions
    y_pred = predict(x_train)

    # Compute training loss
    loss = mse_loss(y_pred, y_train)

    # Backward pass: compute gradients
    loss.backward()

    # Update weights and bias without tracking gradients
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Reset gradients to zero for next iteration
    w.grad.zero_()
    b.grad.zero_()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch + 1}, MSE: {loss}")

    # Store loss value for visualization
    train_losses.append(loss.item())

# Plot training loss curve
plt.plot(train_losses, label="train_loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training loss")
plt.grid(True)
plt.show()

# Make predictions on test data
y_test_pred = predict(x_test)

# Compute test MSE loss
test_loss = mse_loss(y_test_pred, y_test).item()

# Print test MSE
print(f"Test MSE: {test_loss}")

# Print final training RMSE
print(f"Training Loss (RMSE): {(train_losses[-1]) ** 0.5}")

# Print testing RMSE
print(f"Testing Loss (RMSE): {(test_loss) ** 0.5}")
