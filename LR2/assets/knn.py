import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. DISTANCE METRICS
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))


# 2. DATA LOADING & PREPROCESSING
def load_iris_3class():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print("Failed to load data.")
        return None, None
    # 0: Setosa, 1: Versicolor, 2: Virginica
    unique_species = df["species"].unique()
    mapper = {name: i for i, name in enumerate(unique_species)}
    df["species"] = df["species"].map(mapper)
    # Normalize features to [0, 1]
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df[feature_cols] = (df[feature_cols] - df[feature_cols].min()) / (
        df[feature_cols].max() - df[feature_cols].min()
    )
    X = df[["sepal_length", "sepal_width"]].values
    # X = df[['sepal_length', 'petal_width']].values
    y = df["species"].values

    return X, y, list(unique_species)


def train_test_split_custom(X, y, test_size=0.2, seed=9):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    split = int(len(X) * test_size)
    test_idx, train_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# 3. KNN ALGORITHM (Raw Implementation)


class KNN:
    def __init__(self, k, metric):
        self.k = k
        self.metric = metric
        self.dist_func = (
            euclidean_distance if metric == "euclidean" else manhattan_distance
        )
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # KNN is "lazy": just store the training data
        self.X_train = X
        self.y_train = y

    def _predict_sample(self, x):
        # 1. Calculate distances to all training points
        distances = [self.dist_func(x, x_train) for x_train in self.X_train]
        # 2. Sort and get indices of K nearest neighbors
        k_indices = np.argsort(distances)[: self.k]
        # 3. Get labels of neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # 4. Vote (Mode) return most common label
        most_common = np.bincount(k_nearest_labels).argmax()
        # 5. Probability (for ROC): Fraction of neighbors matching prediction
        prob = np.bincount(k_nearest_labels, minlength=3) / self.k

        return most_common, prob

    def predict(self, X):
        predictions = [self._predict_sample(x) for x in X]
        # Unpack predictions (classes) and probabilities
        y_pred = np.array([p[0] for p in predictions])
        y_probs = np.array([p[1] for p in predictions])
        return y_pred, y_probs


# 4. EVALUATION METRICS (Multi-Class)
def confusion_matrix_3class(y_true, y_pred):
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix_3class(y_true, y_pred)
    # Accuracy: Sum of Diagonal / Total
    accuracy = np.trace(cm) / np.sum(cm)
    # Per-Class Precision, Recall, F1
    precisions = []
    recalls = []

    for i in range(3):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        precisions.append(p)
        recalls.append(r)

    # Macro Average (Simple average of per-class scores)
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = (
        2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
        if (macro_precision + macro_recall) > 0
        else 0
    )

    return accuracy, macro_precision, macro_recall, macro_f1, cm


def get_roc_curve_ovr(y_test, y_probs, class_id):
    # One-vs-Rest ROC for a specific class
    y_binary = (y_test == class_id).astype(int)
    probs_binary = y_probs[:, class_id]

    thresholds = np.linspace(0, 1.1, 20)  # 0 to 1
    tpr_list, fpr_list = [], []

    for t in thresholds:
        pred_bin = (probs_binary >= t).astype(int)

        TP = np.sum((y_binary == 1) & (pred_bin == 1))
        TN = np.sum((y_binary == 0) & (pred_bin == 0))
        FP = np.sum((y_binary == 0) & (pred_bin == 1))
        FN = np.sum((y_binary == 1) & (pred_bin == 0))

        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list


# 5. VISUALIZATION FUNCTIONS
def plot_decision_boundary(model, X, y, title="KNN Decision Boundary"):
    # Create mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

    # Predict for whole mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z, _ = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    # Scatter points
    colors = ["purple", "teal", "yellow"]
    labels = ["Setosa", "Versicolor", "Virginica"]
    for i in range(3):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=colors[i], edgecolors="k", label=labels[i])

    plt.title(title)
    plt.xlabel("Length")
    plt.ylabel("Width")
    plt.legend()


# 6. MAIN EXECUTION
# A. Load
X, y, class_names = load_iris_3class()
# B. Split
X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
# C. Train Model
k_neighbors = 5
# metric = "euclidean"
metric = "manhattan"
knn = KNN(k=k_neighbors, metric=metric)
knn.fit(X_train, y_train)
print(f"Training KNN (k={k_neighbors}, metric={metric})...")
# D. Predict
y_pred, y_probs = knn.predict(X_test)
# E. Evaluate
acc, prec, rec, f1, cm = calculate_metrics(y_test, y_pred)

print("\nEVALUATION REPORT (3-Class Iris)")
print("=" * 40)
print(f"Accuracy:  {acc:.2%}")
print(f"Precision (Macro): {prec:.2%}")
print(f"Recall (Macro):    {rec:.2%}")
print(f"F1 Score (Macro):  {f1:.4f}")
print("-" * 20)
print("Confusion Matrix:")
print(cm)

# F. Plotting
plt.figure(figsize=(18, 5))

# Plot 1: Confusion Matrix
plt.subplot(1, 3, 1)
plt.imshow(cm, cmap="Blues", alpha=0.7)
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
plt.title("Confusion Matrix")
plt.xticks(range(3), class_names)
plt.yticks(range(3), class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Plot 2: ROC Curves (One-vs-Rest)
plt.subplot(1, 3, 2)
colors = ["purple", "teal", "orange"]
for i in range(3):
    fpr, tpr = get_roc_curve_ovr(y_test, y_probs, i)
    # Sort for plotting line correctly
    sorted_indices = np.argsort(fpr)
    fpr = np.array(fpr)[sorted_indices]
    tpr = np.array(tpr)[sorted_indices]

    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"{class_names[i]} vs Rest")

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC (One-vs-Rest)")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Decision Boundary
plt.subplot(1, 3, 3)
plot_decision_boundary(
    knn, X_test, y_test, title=f"KNN Decision Boundary (k={k_neighbors})"
)

plt.tight_layout()
# plt.savefig("./images/output/knn_euclidean_results.png")
plt.savefig("./images/output/knn_manhattan_results.png")
plt.show()
