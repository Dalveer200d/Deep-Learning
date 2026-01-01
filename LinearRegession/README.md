# Linear Regression from Scratch using PyTorch

This project demonstrates a **from-scratch implementation of Linear Regression using PyTorch**, without relying on high-level abstractions such as `nn.Module` or `torch.optim`. The goal is to build a strong conceptual foundation in **deep learning fundamentals**, including tensor operations, automatic differentiation, and gradient-based optimization.

The model is trained and evaluated on the **Concrete Compressive Strength dataset**.

---

## Dataset

- **Name:** Concrete Compressive Strength Dataset  
- **Problem Type:** Regression  
- **Target Variable:** Concrete compressive strength  
- **Features:** Cement, water, aggregates, age, and other concrete mix properties  

The dataset is loaded directly from the provided CSV file.

---

## Model Overview

- **Model Type:** Linear Regression  
- **Prediction Equation:**
- **Loss Function:** Mean Squared Error (MSE)  
- **Optimization:** Manual Gradient Descent using PyTorch autograd  

This implementation explicitly shows:
- Weight and bias initialization
- Forward pass computation
- Backpropagation (`loss.backward()`)
- Manual parameter updates
- Gradient resetting

---

## Workflow

1. Load dataset using Pandas  
2. Separate input features and target variable  
3. Apply feature scaling using `StandardScaler`  
4. Convert data into PyTorch tensors  
5. Perform random train–test split (80/20)  
6. Train the model using gradient descent  
7. Track and visualize training loss  
8. Evaluate model performance using MSE and RMSE  

---

## How to Run

### Run as Python Script
```bash
python linearModel.py
