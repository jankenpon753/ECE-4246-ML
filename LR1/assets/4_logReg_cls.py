
def standardize(X):
    # Standardization (Z-score Normalization) is crucial for Logistic Regression
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

def sigmoid(z):
    # Sigmoid Activation: 1 / (1 + e^-z)
    z = np.clip(z, -500, 500) # prevent overflow
    return 1.0 / (1.0 + np.exp(-z))

def compute_cost(X, y, theta):
    # Log Loss (Binary Cross Entropy)
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-15 # prevent log(0)
    h = np.clip(h, epsilon, 1 - epsilon)
    cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def train_logistic_regression(X, y, lr=0.1, epochs=3000):
    # Add Intercept (Bias)
    X_b = np.c_[np.ones((len(X), 1)), X]
    theta = np.zeros(X_b.shape[1])
    m = len(y)
    cost_history = []
    
    for _ in range(epochs):
        # Forward Prop
        z = np.dot(X_b, theta)
        h = sigmoid(z)
        
        # Gradient Calculation
        gradient = np.dot(X_b.T, (h - y)) / m
        
        # Update Weights
        theta -= lr * gradient
        
        # Record Cost
        cost_history.append(compute_cost(X_b, y, theta))
        
    return theta, cost_history

def predict(X, theta, threshold=0.5):
    X_b = np.c_[np.ones((len(X), 1)), X]
    probs = sigmoid(np.dot(X_b, theta))
    preds = (probs >= threshold).astype(int)
    return preds, probs