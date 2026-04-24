import numpy as np
import matplotlib.pyplot as plt


# 1. HELPER FUNCTIONS
def gaussian(x, mean, sigma):
    # Gaussian fuzzy membership function
    return np.exp(-((x - mean) ** 2) / (2 * sigma**2))


def r2_score(y_true, y_pred):
    # #Calculate R-squared score#
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


# 2. MAIN EXECUTION
def main():
    print("=" * 50)
    print(" NEURO-FUZZY SYSTEM (EXPANDED DATASET)")
    print("=" * 50)

    # 1. Generate a larger, non-linear dataset
    # 100 points, non-linear curve: y = x + 2*sin(x) + noise
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y = X + 2 * np.sin(X) + np.random.normal(0, 0.3, 100)  # Added slight noise

    # 2. Initialize Neuro-Fuzzy Parameters
    num_rules = 5  # Using 5 rules to cover the complex curve

    # Spread the Gaussian means evenly across the X range
    means = np.linspace(X.min(), X.max(), num_rules)
    # Set sigma (width) so they overlap smoothly
    sigmas = np.ones(num_rules) * 1.5
    # Initialize consequent weights (w) to zeros
    weights = np.zeros(num_rules)

    learning_rate = 0.05
    epochs = 1000
    mse_history = []

    print(f"Training started with {len(X)} samples and {num_rules} fuzzy rules...")

    # 3. Training Loop
    for epoch in range(epochs):
        epoch_error = 0

        for i in range(len(X)):
            # Step 1 & 2: Fuzzification (Compute memberships and firing strengths)
            firing_strengths = np.array(
                [gaussian(X[i], means[j], sigmas[j]) for j in range(num_rules)]
            )
            total_firing = np.sum(firing_strengths) + 1e-8  # Prevent division by zero

            # Step 3: Defuzzification (Compute weighted average output)
            y_pred = np.sum(firing_strengths * weights) / total_firing

            # Step 4: Calculate Error
            error = y[i] - y_pred
            epoch_error += error**2

            # Step 5: Update Parameters (Gradient update for consequent weights)
            for j in range(num_rules):
                weights[j] += (
                    learning_rate * error * (firing_strengths[j] / total_firing)
                )

        # Record Mean Squared Error for the epoch
        mse_history.append(epoch_error / len(X))

        if epoch % 200 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | MSE: {mse_history[-1]:.4f}")

    print("\nTraining completed.")
    print("Final Rule Weights:")
    for j in range(num_rules):
        print(f"  Rule {j+1} (Center={means[j]:.1f}): w = {weights[j]:.4f}")

    # 4. Final Predictions & Evaluation
    y_final_preds = np.zeros(len(X))
    for i in range(len(X)):
        firing_strengths = np.array(
            [gaussian(X[i], means[j], sigmas[j]) for j in range(num_rules)]
        )
        y_final_preds[i] = np.sum(firing_strengths * weights) / (
            np.sum(firing_strengths) + 1e-8
        )

    final_mse = np.mean((y - y_final_preds) ** 2)
    final_mae = np.mean(np.abs(y - y_final_preds))
    r2 = r2_score(y, y_final_preds)

    print("\n" + "=" * 40)
    print("EVALUATION REPORT")
    print("=" * 40)
    print(f"Mean Squared Error (MSE): {final_mse:.4f}")
    print(f"Mean Absolute Error (MAE): {final_mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # 5. Visualizations
    plt.figure(figsize=(15, 5))

    # Plot A: Error History
    plt.subplot(1, 3, 1)
    plt.plot(mse_history, color="purple", linewidth=2)
    plt.title("Neuro-Fuzzy Learning Curve")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)

    # Plot B: Fuzzy Membership Functions
    plt.subplot(1, 3, 2)
    x_range = np.linspace(X.min(), X.max(), 200)
    for j in range(num_rules):
        mf = gaussian(x_range, means[j], sigmas[j])
        plt.plot(x_range, mf, label=f"Rule {j+1}")
    plt.title("Gaussian Membership Functions")
    plt.xlabel("Input (X)")
    plt.ylabel("Membership Degree")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot C: Predictions vs Actual
    plt.subplot(1, 3, 3)
    plt.scatter(X, y, color="red", alpha=0.6, label="Actual Data")
    plt.plot(X, y_final_preds, color="blue", linewidth=2, label="Neuro-Fuzzy Fit")
    plt.title("System Output vs Target")
    plt.xlabel("Input (X)")
    plt.ylabel("Output (y)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
