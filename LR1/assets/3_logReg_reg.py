# 2. LOGISTIC REGRESSION MODEL (From Scratch)
def sigmoid(z):
    # Clip to prevent overflow/underflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.1, epochs=5000):
    # Add Intercept
    X_b = np.c_[np.ones((len(X), 1)), X]
    theta = np.zeros(X_b.shape[1])
    m = len(y)
    cost_history = []
    
    for i in range(epochs):
        # Forward
        z = np.dot(X_b, theta)
        h = sigmoid(z)
        # Gradient (Derivatives)
        gradient = np.dot(X_b.T, (h - y)) / m
        # Update
        theta -= lr * gradient
        # Cost (Log Loss)
        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon) # Safety for log
        cost = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
        cost_history.append(cost)
        
    return theta, cost_history

def predict_proba(X, theta):
    X_b = np.c_[np.ones((len(X), 1)), X]
    return sigmoid(np.dot(X_b, theta))

def predict(X, theta, threshold=0.5):
    probs = predict_proba(X, theta)
    return (probs >= threshold).astype(int)