# Code when I use Common KFold

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score

x = []
y = []

dataset_path = 'dataset'

classe = {
    "Drowsy": 0,
    "Non Drowsy": 1
}

# Limit the number of images to process for each class
limit = 10000

# Loop through each class and read the images
for classe_nome, label in classe.items():
    folder = os.path.join(dataset_path, classe_nome)
    archives = os.listdir(folder)

    archives = archives[:limit]

    # Loop through each image in the class folder
    for archive in archives:
        path = os.path.join(folder, archive)
        img = cv2.imread(path)

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (32, 32))
            img = img.flatten()

            x.append(img)
            y.append(label)

# Convert the lists to numpy arrays
x = np.array(x)
y = np.array(y)

# Normalize the pixels to the range [0, 1]
x = x.astype(np.float32) / 255.0

print("X format: ", x.shape)
print("Y format: ", y.shape)

# ----------------

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Define the list between 1 and 20
k_values = list(range(1, 21))

results = []

# Loop through each value of k and execute the KNN with KFold
for k in k_values:
    model = KNeighborsClassifier(
        n_neighbors=k
    )

    scores = cross_val_score(
        model,
        x,
        y,
        cv=kf,
        scoring='accuracy'
    )

    media = scores.mean()
    results.append(media)

    print(f"K={k} | Accuracy: {media * 100:.2f}%")

# Find the best accuracy obtained
best_accuracy = max(results)

# Find the K value corresponding to the best accuracy
best_k = k_values[results.index(best_accuracy)]

print(
    f"\nBest K: {best_k} | Best Accuracy: {best_accuracy * 100:.2f}%"
)