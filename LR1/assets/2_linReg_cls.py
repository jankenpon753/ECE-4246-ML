# 2. MATHEMATICAL FUNCTIONS (Linear Regression)

def compute_cost_mse(X, y, theta):
    """
    Cost Function: Mean Squared Error (MSE)
    Even though this is classification, Linear Regression uses MSE.
    """
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        # Gradient Calculation
        gradient = (1/m) * X.T.dot(errors)
        # Update
        theta -= learning_rate * gradient
        # Log Cost
        cost_history.append(compute_cost_mse(X, y, theta))
    return theta, cost_history

def predict_class(X, theta, threshold=0.5):
    continuous_pred = X.dot(theta)
    return (continuous_pred >= threshold).astype(int), continuous_pred

# 3. EVALUATION METRICS
def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    acc = (TP + TN) / (TP + TN + FP + FN)
    prec = TP / (TP + FP) if (TP+FP) > 0 else 0
    rec = TP / (TP + FN) if (TP+FN) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec+rec) > 0 else 0
    
    return acc, prec, rec, f1, cm
