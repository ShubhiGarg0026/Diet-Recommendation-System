import matplotlib.pyplot as plt
import numpy as np


# -------------------------------
# INDIVIDUAL MODEL GRAPH
# -------------------------------
def plot_single_model(model_name, values):
    labels = ["Calories", "Protein", "Carbs", "Fats"]

    plt.figure()
    plt.bar(labels, values)
    plt.title(f"{model_name} Prediction")
    plt.ylabel("Values")

    filename = f"{model_name}_graph.png"
    plt.savefig(filename)
    plt.close()

    return filename


# -------------------------------
# COMBINED COMPARISON GRAPH
# -------------------------------
def plot_all_models_comparison(models_output):
    labels = ["Calories", "Protein", "Carbs", "Fats"]
    x = np.arange(len(labels))

    plt.figure(figsize=(10, 6))

    for i, (model_name, values) in enumerate(models_output.items()):
        plt.plot(x, values, marker='o', label=model_name)

    plt.xticks(x, labels)
    plt.title("Model Comparison")
    plt.ylabel("Values")
    plt.legend()
    plt.grid()

    plt.savefig("comparison_all_models.png")
    plt.close()


# -------------------------------
# BEST MODEL SELECTION
# -------------------------------
def choose_best_model(models_output, user_input):
    """
    Compare models based on closeness to input
    (simple distance metric)
    """

    best_model = None
    best_score = float("inf")

    for model_name, values in models_output.items():
        score = np.linalg.norm(np.array(values) - np.array(user_input[:4]))

        if score < best_score:
            best_score = score
            best_model = model_name

    return best_model