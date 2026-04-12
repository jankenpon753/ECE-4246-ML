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
        std_val[std_val == 0] = 1e-8
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


def plot_decision_boundary(X, y, model, class_names, title):
    plt.figure(figsize=(9, 6))
    colors = ["purple", "teal", "yellow"]
    markers = ["o", "s", "^"]

    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # Predict over mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)

    # Plot contours
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")

    # Scatter actual test points
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


# 3. GAUSSIAN NAIVE BAYES ALGORITHM
class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        # Initialize arrays for mean, variance, and prior probabilities
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        # Calculate statistics for each class
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(X.shape[0])

    def _calculate_likelihood(self, class_idx, x):
        """Gaussian Probability Density Function (PDF)"""
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        # Add small epsilon to variance to prevent division by zero
        numerator = np.exp(-((x - mean) ** 2) / (2 * var + 1e-8))
        denominator = np.sqrt(2 * np.pi * var + 1e-8)
        return numerator / denominator

    def predict(self, X):
        y_pred = [self._predict_sample(x) for x in X]
        return np.array(y_pred)

    def _predict_sample(self, x):
        posteriors = []

        # Calculate posterior probability for each class
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            # Sum of logs is used instead of multiplying probabilities to prevent underflow
            posterior = np.sum(np.log(self._calculate_likelihood(idx, x) + 1e-8))
            posterior = prior + posterior
            posteriors.append(posterior)

        # Return class with highest posterior probability
        return self.classes[np.argmax(posteriors)]


# 4. DATA LOADER & MAIN EXECUTION
def load_all_iris():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

    class_names = ["setosa", "versicolor", "virginica"]
    mapper = {name: i for i, name in enumerate(class_names)}
    df["species"] = df["species"].map(mapper)

    # 2 Features for Visualization
    X = df[["petal_length", "petal_width"]].values
    y = df["species"].values
    return X, y, class_names


def main():
    print("=" * 50)
    print(" GAUSSIAN NAIVE BAYES CLASSIFICATION")
    print("=" * 50)

    # 1. Load Data
    X, y, class_names = load_all_iris()
    if X is None:
        return

    # 2. Split and Standardize
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2)
    X_train_std, mean_val, std_val = standardize(X_train)
    X_test_std, _, _ = standardize(X_test, mean_val, std_val)

    # 3. Train Naive Bayes
    print("\nTraining Gaussian Naive Bayes Model...")
    nb_model = GaussianNaiveBayes()
    nb_model.fit(X_train_std, y_train)

    # 4. Predict & Evaluate
    y_pred = nb_model.predict(X_test_std)
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

    # 5. Visualizations
    plot_confusion_matrix(
        cm, class_names=class_names, title="Naive Bayes Confusion Matrix"
    )

    plot_decision_boundary(
        X_test_std,
        y_test,
        nb_model,
        class_names=class_names,
        title="Gaussian Naive Bayes Decision Regions",
    )


if __name__ == "__main__":
    main()
