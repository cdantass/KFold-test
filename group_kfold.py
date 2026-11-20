# Code when I use GroupKFold

import os
import re
import cv2
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold, cross_val_score


x = []
y = []
groups = []

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

        participant_match = re.match(r"^([A-Za-z]+)", archive)

        if participant_match is None:
            print(f"Participante não identificado: {archive}")
            continue

        participant = participant_match.group(1).lower()

        x.append(image)
        y.append(label)
        groups.append(participant)

x = np.array(x)
y = np.array(y)
groups = np.array(groups)

x = x.astype(np.float32) / 255.0

print("Formato de X:", x.shape)
print("Formato de Y:", y.shape)
print("Participantes:", len(np.unique(groups)))

group_kfold = GroupKFold(n_splits=5)
k_values = range(1, 21)
results = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(
        model,
        x,
        y,
        groups=groups,
        cv=group_kfold,
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

best_result = max(results, key=lambda result: result["accuracy"])

print("\nMelhor resultado")
print(f"K: {best_result['k']}")
print(f"Acurácia: {best_result['accuracy'] * 100:.2f}%")
print(f"Desvio padrão: {best_result['std'] * 100:.2f}%")