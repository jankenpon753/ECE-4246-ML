import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing


# 1. PERCEPTRON ALGORITHM (From Scratch)
class Perceptron:
    def __init__(self, learning_rate, epochs):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors_history = []  # Track misclassifications per epoch

    def _activation_function(self, z):
        # Step function: returns 1 if z >= 0, else 0
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training loop
        for epoch in range(self.epochs):
            errors = 0
            # The Perceptron updates weights sample by sample
            for idx, x_i in enumerate(X):
                # 1. Calculate linear output: z = (w * x) + b
                linear_output = np.dot(x_i, self.weights) + self.bias
                # 2. Apply step activation function
                y_predicted = self._activation_function(linear_output)

                # 3. Perceptron Update Rule
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

                # Count misclassifications
                if update != 0:
                    errors += 1

            self.errors_history.append(errors)

            # Early stopping if perfectly classified
            if errors == 0:
                print(f"Converged early at epoch {epoch + 1}!")
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activation_function(linear_output)


# 2. HELPER & EVALUATION FUNCTIONS
def standardize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


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
def plot_decision_boundary(X_plot, y, model, title, feature_names, class_names):
    x_min, x_max = X_plot[:, 0].min() - 0.5, X_plot[:, 0].max() + 0.5
    y_min, y_max = X_plot[:, 1].min() - 0.5, X_plot[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    X_grid_plot = np.c_[xx.ravel(), yy.ravel()]

    num_features = model.weights.shape[0]
    X_full_grid = np.zeros((X_grid_plot.shape[0], num_features))
    X_full_grid[:, :2] = X_grid_plot

    Z = model.predict(X_full_grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")

    plt.scatter(
        X_plot[y == 0, 0],
        X_plot[y == 0, 1],
        color="blue",
        marker="o",
        label=class_names[0],
        alpha=0.6,
    )
    plt.scatter(
        X_plot[y == 1, 0],
        X_plot[y == 1, 1],
        color="red",
        marker="x",
        label=class_names[1],
        alpha=0.6,
    )

    plt.title(title)
    plt.xlabel(f"{feature_names[0]} (Standardized)")
    plt.ylabel(f"{feature_names[1]} (Standardized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_errors(errors_history):
    plt.figure(figsize=(8, 4))
    plt.plot(
        range(1, len(errors_history) + 1),
        errors_history,
        marker="o",
        color="green",
        markersize=4,
    )
    plt.title("Perceptron Convergence: Errors vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Number of Misclassifications")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """Plots a beautiful heatmap for the confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    # Set labels and ticks
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="Actual Label",
        xlabel="Predicted Label",
    )

    # Add text annotations inside the boxes
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
    print(" SINGLE LAYER PERCEPTRON (SLP) CLASSIFICATION")
    print("-" * 50)

    global X_std_full, y, slp

    # 1. Load Data
    print("Loading California Housing Dataset...")
    try:
        housing = fetch_california_housing(as_frame=True)
        # Sample the dataset to speed up training (optional, but perceptron on 20k rows is slow)
        df = housing.frame.sample(2000, random_state=42)

        X_full = df.drop("MedHouseVal", axis=1).values
        all_feature_names = list(housing.feature_names)

        # Convert to Binary Classification
        median_value_threshold = housing.target.median()
        y = (df["MedHouseVal"] > median_value_threshold).astype(int).values

        class_names = ["Low Value (0)", "High Value (1)"]

        print(f"Dataset Shape: {X_full.shape}")
        print(f"Target distribution: {np.bincount(y)}")

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Split and Standardize
    X_train, X_test, y_train, y_test = train_test_split_custom(X_full, y, test_size=0.2)

    X_train_std = standardize(X_train)
    X_test_std = standardize(X_test)  # Ideally scale using train stats, simplified here
    X_std_full = standardize(X_full)

    # 3. Train Perceptron
    print("\nTraining Perceptron...")
    slp = Perceptron(learning_rate=0.1, epochs=100)
    slp.fit(X_train_std, y_train)

    # 4. Predict and Evaluate
    y_pred = slp.predict(X_test_std)
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
    # A. Plot Errors
    plot_errors(slp.errors_history)
    # B. Plot Confusion Matrix (ADD THIS LINE)
    print("\nPlotting Confusion Matrix...")
    plot_confusion_matrix(cm, class_names=class_names, title="SLP Confusion Matrix")
    # C. Plot Decision Boundary
    medinc_idx = all_feature_names.index("MedInc")
    houseage_idx = all_feature_names.index("HouseAge")
    X_test_plot = X_test_std[:, [medinc_idx, houseage_idx]]
    plot_feature_names = [
        all_feature_names[medinc_idx],
        all_feature_names[houseage_idx],
    ]

    print(
        f"\nPlotting decision boundary using '{plot_feature_names[0]}' and '{plot_feature_names[1]}'..."
    )
    plot_decision_boundary(
        X_test_plot,
        y_test,
        slp,
        title="SLP Decision Boundary (California Housing)",
        feature_names=plot_feature_names,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()
