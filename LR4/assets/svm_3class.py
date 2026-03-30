import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. EVALUATION & HELPER FUNCTIONS
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


def get_confusion_matrix_3class(y_true, y_pred):
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def evaluate_metrics_multiclass(y_true, y_pred):
    cm = get_confusion_matrix_3class(y_true, y_pred)
    accuracy = np.trace(cm) / np.sum(cm)

    precisions, recalls = [], []
    for i in range(3):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = (
        (2 * macro_precision * macro_recall) / (macro_precision + macro_recall)
        if (macro_precision + macro_recall) > 0
        else 0
    )

    return accuracy, macro_precision, macro_recall, macro_f1, cm


# 2. PLOTTING FUNCTIONS
def plot_multiclass_cost_history(models, class_names):
    plt.figure(figsize=(10, 5))
    colors = ["purple", "teal", "orange"]  # Distinct colors for the 3 curves

    for i, model in enumerate(models):
        plt.plot(
            model.cost_history,
            color=colors[i],
            linewidth=2,
            label=f"{class_names[i]} vs Rest",
        )

    plt.title("Multi-Class SVM Training: Hinge Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Cost (Hinge Loss)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
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


def plot_multiclass_decision_boundary(X, y, model, class_names, title):
    plt.figure(figsize=(10, 6))
    colors = ["purple", "teal", "orange"]
    markers = ["o", "s", "^"]
    # Create mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    # Predict over mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    # Plot filled contours for decision regions
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    # Scatter actual points
    for i in range(3):
        plt.scatter(
            X[y == i, 0],
            X[y == i, 1],
            c=colors[i],
            marker=markers[i],
            edgecolors="k",
            s=60,
            label=class_names[i],
        )

    plt.title(title)
    plt.xlabel("Petal Length (Standardized)")
    plt.ylabel("Petal Width (Standardized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# 3. SVM ALGORITHMS (Base & OvR Wrapper)
class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.w = None
        self.b = None
        self.cost_history = []  # ADDED: Initialize cost history array

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        self.cost_history = []  # Reset on fit

        for epoch in range(self.epochs):
            # ADDED: Calculate Hinge Loss for the epoch history
            distances = 1 - y * (np.dot(X, self.w) + self.b)
            distances[distances < 0] = 0  # max(0, distance)
            hinge_loss = self.lambda_param * np.sum(self.w**2) + (
                1 / n_samples
            ) * np.sum(distances)
            self.cost_history.append(hinge_loss)

            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.w - y[idx] * x_i
                    db = -y[idx]

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def get_raw_score(self, X):
        return np.dot(X, self.w) + self.b


class MultiClassSVM:
    def __init__(self, n_classes=3, learning_rate=0.01, lambda_param=0.01, epochs=1000):
        self.n_classes = n_classes
        # Initialize an array of standard Binary SVMs
        self.models = [
            LinearSVM(learning_rate, lambda_param, epochs) for _ in range(n_classes)
        ]

    def fit(self, X, y):
        for i in range(self.n_classes):
            print(f"  -> Training SVM {i+1} (Class {i} vs Rest)...")
            # Convert target: 1 if class matches 'i', -1 otherwise
            y_binary = np.where(y == i, 1, -1)
            self.models[i].fit(X, y_binary)

    def predict(self, X):
        # Gather scores (w*x + b) from all 3 models
        scores = np.zeros((X.shape[0], self.n_classes))
        for i in range(self.n_classes):
            scores[:, i] = self.models[i].get_raw_score(X)

        # The predicted class is the one with the highest confidence score
        return np.argmax(scores, axis=1)


# 4. MAIN EXECUTION
def load_all_iris():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None
    # Map all 3 classes to 0, 1, 2
    class_names = ["setosa", "versicolor", "virginica"]
    mapper = {name: i for i, name in enumerate(class_names)}
    df["species"] = df["species"].map(mapper)
    # 2 Features for Visualization
    X = df[["petal_length", "petal_width"]].values
    y = df["species"].values
    return X, y, class_names


def main():
    print("=" * 50)
    print(" MULTI-CLASS SVM (One-vs-Rest) CLASSIFICATION")
    print("=" * 50)
    # 1. Load Data
    X, y, class_names = load_all_iris()
    if X is None:
        return
    # 2. Split and Standardize
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2)
    X_train_std, mean_val, std_val = standardize(X_train)
    X_test_std, _, _ = standardize(X_test, mean_val, std_val)
    # 3. Train Multi-Class SVM
    print("\nTraining Multi-Class SVM (One-vs-Rest strategy)...")
    ovr_svm = MultiClassSVM(
        n_classes=3, learning_rate=0.01, lambda_param=0.01, epochs=1500
    )
    ovr_svm.fit(X_train_std, y_train)
    # 4. Predict & Evaluate
    y_pred = ovr_svm.predict(X_test_std)
    acc, prec, rec, f1, cm = evaluate_metrics_multiclass(y_test, y_pred)

    print("\n" + "=" * 40)
    print("EVALUATION REPORT (3-Class)")
    print("=" * 40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (Macro): {prec:.4f}")
    print(f"Recall (Macro):    {rec:.4f}")
    print(f"F1 Score (Macro):  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    # 5. Visualizations. Plot all 3 Cost Histories (One for each SVM trained)
    plot_multiclass_cost_history(ovr_svm.models, class_names)

    plot_confusion_matrix(
        cm, class_names=class_names, title="Multi-Class SVM Confusion Matrix"
    )

    plot_multiclass_decision_boundary(
        X_test_std,
        y_test,
        ovr_svm,
        class_names=class_names,
        title="Multi-Class SVM Decision Regions",
    )


if __name__ == "__main__":
    main()
