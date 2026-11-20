# Code when I use Common KFold

import os
import cv2
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score


x = []
y = []

dataset_path = "dataset"

classes = {
    "Drowsy": 0,
    "Non Drowsy": 1
}

limit = 10000
seed = 42

for class_name, label in classes.items():
    folder = os.path.join(dataset_path, class_name)
    archives = os.listdir(folder)

    rng = np.random.RandomState(seed)
    rng.shuffle(archives)

    archives = archives[:limit]

    for archive in archives:
        path = os.path.join(folder, archive)
        image = cv2.imread(path)

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (32, 32))
        image = image.flatten()

        x.append(image)
        y.append(label)

x = np.array(x)
y = np.array(y)

x = x.astype(np.float32) / 255.0

print("Formato de X:", x.shape)
print("Formato de Y:", y.shape)

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=seed
)

k_values = range(1, 21)
results = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(
        model,
        x,
        y,
        cv=kfold,
        scoring="accuracy"
    )

    average = scores.mean()
    standard_deviation = scores.std()

    results.append({
        "k": k,
        "accuracy": average,
        "std": standard_deviation
    })

    print(
        f"K={k} | "
        f"Acurácia: {average * 100:.2f}% | "
        f"Desvio: {standard_deviation * 100:.2f}%"
    )

best_result = max(
    results,
    key=lambda result: result["accuracy"]
)

print("\nMelhor resultado")
print(f"K: {best_result['k']}")
print(f"Acurácia: {best_result['accuracy'] * 100:.2f}%")
print(f"Desvio padrão: {best_result['std'] * 100:.2f}%")