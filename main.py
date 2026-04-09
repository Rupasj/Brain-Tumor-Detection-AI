from src.train import train_model
from src.evaluate import evaluate

if __name__ == "__main__":
    model, X_test, y_test = train_model()
    evaluate(model, X_test, y_test)