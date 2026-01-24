import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. LOAD DATA (Fixed URL)
def load_data():
    url = "https://raw.githubusercontent.com/krishnaik06/Simple-Linear-Regression/master/Salary_Data.csv"
    
    try:
        df = pd.read_csv(url)
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading online data: {e}")
        return None, None

    # X = Features (YearsExperience), y = Target (Salary)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

# 2. MATHEMATICAL FUNCTIONS (From Scratch)
def train_test_split_custom(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    m = len(y)
    indices = np.random.permutation(m)
    test_count = int(m * test_size)
    
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        # Gradient = (1/m) * X.T * errors
        gradient = (1 / m) * X.T.dot(errors)        
        # Update Rule
        theta -= learning_rate * gradient
        # Track Cost
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

def predict(X, theta):
    return X.dot(theta)

# 3. METRICS (Regression Specific)
def r2_score_custom(y_true, y_pred):
    mean_y = np.mean(y_true)
    ss_total = np.sum((y_true - mean_y) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def mse_custom(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 4. MAIN EXECUTION
X, y = load_data()
# Plot Raw Data
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Data Points')
plt.title('Raw Data: Years Experience vs Salary')
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.show()
# B. Preprocessing
X_b = np.c_[np.ones((len(X), 1)), X] 
# C. Split Data
X_train, X_test, y_train, y_test = train_test_split_custom(X_b, y, test_size=0.2)

# D. Train Model
theta = np.zeros(X_train.shape[1]) # Init weights
learning_rate = 0.01
iterations = 1000
print(f"\nTraining Linear Regression (LR={learning_rate}, Iter={iterations})...")
theta_final, cost_history = gradient_descent(X_train, y_train, theta, learning_rate, iterations)

# E. Evaluate
y_pred_train = predict(X_train, theta_final)
y_pred_test = predict(X_test, theta_final)

mse_val = mse_custom(y_test, y_pred_test)
r2_val = r2_score_custom(y_test, y_pred_test)

print("\n" + "="*40)
print("EVALUATION REPORT")
print("="*40)
print(f"Final Weights (Theta): {theta_final}")
print(f"Mean Squared Error (MSE): {mse_val:,.2f}")
print(f"R2 Score (Accuracy):      {r2_val:.4f}")
print("(Confusion Matrix / ROC are not applicable to Regression)")


# 5. FINAL VISUALIZATION
plt.figure(figsize=(14, 5))
# Plot 1: Cost History (Convergence)
plt.subplot(1, 2, 1)
plt.plot(cost_history, color='red', linewidth=2)
plt.title('Training Cost (Loss) over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.grid(True, alpha=0.3)
# Plot 2: Model Fit Line
plt.subplot(1, 2, 2)
# Scatter Test Data
plt.scatter(X_test[:, 1], y_test, color='green', s=100, label='Test Data (Actual)')
# Plot Regression Line
x_range = np.linspace(X[:,0].min(), X[:,0].max(), 100)
x_range_b = np.c_[np.ones((100, 1)), x_range] # Add intercept for prediction
y_line = predict(x_range_b, theta_final)

plt.plot(x_range, y_line, color='blue', linewidth=3, label='Model Prediction Line')
plt.title('Linear Regression Model Fit')
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()