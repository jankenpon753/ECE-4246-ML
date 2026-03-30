import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. EVALUATION & HELPER FUNCTIONS
# ==========================================
def train_test_split_custom(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    split = int(len(X) * (1 - test_size))
    return (
        X[indices[:split]],
        X[indices[split:]],
        y[indices[:split]],
        y[indices[split:]],
    )


def standardize(X, mean_val=None, std_val=None):
    if mean_val is None or std_val is None:
        mean_val = np.mean(X, axis=0)
        std_val = np.std(X, axis=0)
        std_val[std_val == 0] = 1
    return (X - mean_val) / std_val, mean_val, std_val


def get_confusion_matrix(y_true, y_pred):
    # Mapping back to 0 and 1 for standard matrix positioning
    y_t = np.where(y_true == -1, 0, 1)
    y_p = np.where(y_pred == -1, 0, 1)

    TP = np.sum((y_t == 1) & (y_p == 1))
    TN = np.sum((y_t == 0) & (y_p == 0))
    FP = np.sum((y_t == 0) & (y_p == 1))
    FN = np.sum((y_t == 1) & (y_p == 0))
    return np.array([[TN, FP], [FN, TP]])


def evaluate_metrics(y_true, y_pred):
    cm = get_confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return accuracy, precision, recall, f1, cm


# ==========================================
# 2. PLOTTING FUNCTIONS
# ==========================================
def plot_cost_history(cost_history):
    plt.figure(figsize=(8, 4))
    plt.plot(cost_history, color="purple", linewidth=2)
    plt.title("SVM Training: Hinge Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Cost (Hinge Loss)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    cm, class_names=["Class -1", "Class +1"], title="Confusion Matrix"
):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="Actual Label",
        xlabel="Predicted Label",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )
    fig.tight_layout()
    plt.show()


def plot_svm_decision_boundary(X, y, model, title):
    plt.figure(figsize=(10, 6))

    # Scatter the data points
    plt.scatter(
        X[y == -1, 0], X[y == -1, 1], color="blue", marker="o", label="Versicolor (-1)"
    )
    plt.scatter(
        X[y == 1, 0], X[y == 1, 1], color="red", marker="x", label="Virginica (+1)"
    )

    # Create a grid to evaluate model
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()])

    # Calculate the decision function (w*x + b) for the grid
    Z = (np.dot(model.w, xy) + model.b).reshape(XX.shape)

    # Plot the decision boundary and margins
    # Z = 0 is the decision boundary; Z = -1 and Z = 1 are the margins
    ax.contour(
        XX,
        YY,
        Z,
        colors=["blue", "black", "red"],
        levels=[-1, 0, 1],
        alpha=0.8,
        linestyles=["--", "-", "--"],
        linewidths=[2, 2, 2],
    )

    plt.title(title)
    plt.xlabel("Petal Length (Standardized)")
    plt.ylabel("Petal Width (Standardized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ==========================================
# 3. SUPPORT VECTOR MACHINE ALGORITHM
# ==========================================
class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.w = None
        self.b = None
        self.cost_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # Ensure labels are strictly -1 and 1
        y_ = np.where(y <= 0, -1, 1)

        for epoch in range(self.epochs):
            # Calculate Hinge Loss for the epoch history
            distances = 1 - y_ * (np.dot(X, self.w) + self.b)
            distances[distances < 0] = 0  # max(0, distance)
            hinge_loss = self.lambda_param * np.sum(self.w**2) + (
                1 / n_samples
            ) * np.sum(distances)
            self.cost_history.append(hinge_loss)

            # Stochastic Gradient Descent updates
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    # Point is classified correctly and lies strictly outside the margin
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    # Point is misclassified or lies inside the margin
                    dw = 2 * self.lambda_param * self.w - y_[idx] * x_i
                    db = -y_[idx]

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)


# ==========================================
# 4. DATA LOADER & MAIN EXECUTION
# ==========================================
def load_binary_iris_for_svm():
    # Load from online repository
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    # Filter for Versicolor and Virginica to make it a hard binary problem
    df = df[df["species"].isin(["versicolor", "virginica"])]

    # SVM REQUIRED PREPROCESSING: Map to -1 and 1
    df["species"] = df["species"].map({"versicolor": -1, "virginica": 1})

    # Select 2 Features for 2D visualization: Petal Length and Petal Width
    X = df[["petal_length", "petal_width"]].values
    y = df["species"].values

    return X, y


def main():
    print("=" * 50)
    print(" SUPPORT VECTOR MACHINE (LINEAR) CLASSIFICATION")
    print("=" * 50)

    # 1. Load Iris Data
    print("Loading Iris Dataset (Versicolor vs Virginica)...")
    X, y_svm = load_binary_iris_for_svm()

    if X is None:
        return

    # 2. Split and Standardize
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y_svm, test_size=0.2)

    # Fit scaler on training data, transform both
    X_train_std, mean_val, std_val = standardize(X_train)
    X_test_std, _, _ = standardize(X_test, mean_val, std_val)

    # 3. Train SVM
    print("\nTraining Linear SVM...")
    svm = LinearSVM(learning_rate=0.01, lambda_param=0.01, epochs=1500)
    svm.fit(X_train_std, y_train)

    # 4. Predict & Evaluate
    y_pred = svm.predict(X_test_std)
    acc, prec, rec, f1, cm = evaluate_metrics(y_test, y_pred)

    print("\n" + "=" * 40)
    print("EVALUATION REPORT")
    print("=" * 40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # 5. Visualizations
    plot_cost_history(svm.cost_history)
    plot_confusion_matrix(
        cm,
        class_names=["Versicolor (-1)", "Virginica (+1)"],
        title="SVM Confusion Matrix",
    )
    plot_svm_decision_boundary(
        X_test_std, y_test, svm, title="SVM Decision Boundary & Margins (Iris Test Set)"
    )


if __name__ == "__main__":
    main()
