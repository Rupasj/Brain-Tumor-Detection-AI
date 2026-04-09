import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def evaluate(model, X_test, y_test):
    # -------- CREATE OUTPUT FOLDER --------
    os.makedirs("outputs", exist_ok=True)

    # -------- PREDICTIONS --------
    probs = model.predict(X_test)

    # Adjustable threshold (important for medical tasks)
    threshold = 0.4
    preds = (probs > threshold).astype("int32")

    # -------- CLASSIFICATION REPORT --------
    report = classification_report(y_test, preds)

    print("📊 Classification Report:\n")
    print(report)

    # -------- CONFUSION MATRIX --------
    cm = confusion_matrix(y_test, preds)

    print("🧮 Confusion Matrix:\n")
    print(cm)

    # -------- SAVE RESULTS --------
    with open("outputs/results.txt", "w") as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    return report, cm