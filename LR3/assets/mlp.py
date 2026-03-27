import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


# 1. MLP ALGORITHM (From Scratch)
class MultiLayerPerceptron:
    def __init__(
        self, input_size=2, hidden_size=4, output_size=1, learning_rate=0.2, epochs=5000
    ):
        self.lr = learning_rate
        self.epochs = epochs

        # Initialize Weights and Biases randomly
        np.random.seed(42)
        # Weights from Input to Hidden Layer
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))

        # Weights from Hidden to Output Layer
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

        self.cost_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, a):
        # The derivative of sigmoid is a * (1 - a)
        return a * (1 - a)

    def fit(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)  # Ensure y is a column vector (m, 1)

        for epoch in range(self.epochs):
            # FORWARD PROPAGATION
            # Layer 1 (Hidden)
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self._sigmoid(Z1)

            # Layer 2 (Output)
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self._sigmoid(Z2)

            # Record Cost (Binary Cross Entropy)
            epsilon = 1e-8
            cost = (-1 / m) * np.sum(
                y * np.log(A2 + epsilon) + (1 - y) * np.log(1 - A2 + epsilon)
            )
            self.cost_history.append(cost)

            # BACKWARD PROPAGATION
            # Output Layer Error
            dZ2 = A2 - y
            dW2 = (1 / m) * np.dot(A1.T, dZ2)
            db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

            # Hidden Layer Error
            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self._sigmoid_derivative(A1)
            dW1 = (1 / m) * np.dot(X.T, dZ1)
            db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

            # UPDATE WEIGHTS
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

            if epoch % 1000 == 0:
                print(f"Epoch {epoch} | Cost: {cost:.4f}")

    def predict_proba(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._sigmoid(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self._sigmoid(Z2)
        return A2

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# 2. EVALUATION & HELPER FUNCTIONS
def train_test_split_custom(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    split = int(len(X) * test_size)
    return (
        X[indices[split:]],
        X[indices[:split]],
        y[indices[split:]],
        y[indices[:split]],
    )


def get_confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
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


# 3. PLOTTING FUNCTIONS
def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")

    plt.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        color="red",
        marker="o",
        edgecolors="k",
        label="Class 0 (Test)",
    )
    plt.scatter(
        X[y == 1, 0], X[y == 1, 1], color="blue", marker="x", label="Class 1 (Test)"
    )

    plt.title(title)
    plt.xlabel("Feature 1 (Standardized)")
    plt.ylabel("Feature 2 (Standardized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_cost_history(cost_history):
    plt.figure(figsize=(8, 4))
    plt.plot(cost_history, color="purple", linewidth=2)
    plt.title("MLP Training: Cost vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Log Loss (Cost)")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5, 4))
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


# 4. MAIN EXECUTION
def main():
    print("-" * 50)
    print(" MULTI-LAYER PERCEPTRON (NON-LINEAR CLASSIFICATION)")
    print("-" * 50)

    # 1. Generate Non-Linear Data (Two Moons)
    print("Generating 'Two Moons' dataset...")
    X, y = make_moons(n_samples=1000, noise=0.15, random_state=42)

    # 2. Split and Standardize
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2)

    # Standardize data (Neural networks prefer scaled data)
    mean_val = np.mean(X_train, axis=0)
    std_val = np.std(X_train, axis=0)
    X_train = (X_train - mean_val) / std_val
    X_test = (X_test - mean_val) / std_val

    # 3. Train MLP
    # Architecture: 2 inputs -> 4 hidden neurons -> 1 output
    print(
        f"\nTraining MLP (2 Inputs -> 4 Hidden -> 1 Output) on {len(X_train)} samples..."
    )
    mlp = MultiLayerPerceptron(
        input_size=2, hidden_size=4, output_size=1, learning_rate=0.2, epochs=5000
    )
    mlp.fit(X_train, y_train)

    # 4. Predict & Evaluate
    y_pred = mlp.predict(X_test).flatten()
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
    plot_cost_history(mlp.cost_history)
    plot_confusion_matrix(
        cm, class_names=["Moon 0", "Moon 1"], title="MLP Confusion Matrix"
    )

    # Plot decision boundary specifically over the test set
    plot_decision_boundary(
        X_test, y_test, mlp, title="MLP Decision Boundary (Test Data)"
    )


if __name__ == "__main__":
    main()
