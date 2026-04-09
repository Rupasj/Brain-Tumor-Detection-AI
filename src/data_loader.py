import os
import cv2
import numpy as np

IMG_SIZE = 224

def load_data(data_dir):
    X = []
    y = []

    categories = ["yes", "no"]

    for category in categories:
        path = os.path.join(data_dir, category)
        label = categories.index(category)

        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                X.append(image)
                y.append(label)
            except:
                pass

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    X = np.array(X)
    X = preprocess_input(X)
    y = np.array(y)

    return X, y