from src.data_loader import load_data
from src.preprocess import split_data
from src.model import build_model

import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def train_model():
    # -------- LOAD DATA --------
    X, y = load_data("data/raw")

    # -------- SPLIT DATA --------
    X_train, X_test, y_train, y_test = split_data(X, y)

    # -------- DATA AUGMENTATION --------
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    datagen.fit(X_train)

    # -------- BUILD MODEL --------
    model = build_model()

    # -------- CLASS WEIGHTS (IMPORTANT) --------
    # Helps model focus more on tumor class
    class_weight = {0: 1, 1: 3}

    # -------- CREATE DIRECTORIES --------
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    # -------- CALLBACKS --------
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        "models/best_model.keras",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # -------- TRAIN --------
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=30,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, checkpoint],
        class_weight=class_weight   # 🔥 KEY IMPROVEMENT
    )

    # -------- SAVE FINAL MODEL --------
    model.save("models/brain_tumor_model.keras")

    # -------- PLOT ACCURACY --------
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])

    plt.savefig("outputs/plots/accuracy.png")
    plt.close()

    # -------- PLOT LOSS (NEW) --------
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])

    plt.savefig("outputs/plots/loss.png")
    plt.close()

    return model, X_test, y_test