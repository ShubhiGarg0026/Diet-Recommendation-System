import matplotlib.pyplot as plt


def plot_comparison(user, match_row):

    labels = ["Age", "BMI", "BP", "Cholesterol", "Sugar"]

    user_values = [float(x) for x in user]

    match_values = [
        float(match_row["Age"]),
        float(match_row["BMI"]),
        float(match_row["Blood_Pressure_Systolic"]),
        float(match_row["Cholesterol_Level"]),
        float(match_row["Blood_Sugar_Level"])
    ]

    x = range(len(labels))

    plt.figure(figsize=(8, 5))
    plt.plot(x, user_values, marker='o', label="User")
    plt.plot(x, match_values, marker='o', label="Matched")

    plt.xticks(x, labels)
    plt.legend()
    plt.title("User vs Predicted Health Profile")
    plt.grid()

    plt.savefig("comparison.png")
    plt.close()