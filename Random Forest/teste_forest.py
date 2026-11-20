import os
import re
import cv2
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GroupKFold, cross_val_score

SEED = 42
limit = 10000
dataset_path = 'dataset'

classe = {"Drowsy": 0, "Non Drowsy": 1}

x = []
y = []
groups = []

for nome, label in classe.items():
    folder = os.path.join(dataset_path, nome)
    archives = os.listdir(folder)
    rng = np.random.RandomState(SEED)
    rng.shuffle(archives)
    archives = archives[:limit]

    for archive in archives:
        img = cv2.imread(os.path.join(folder, archive))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (32, 32))
        x.append(img.flatten())
        y.append(label)
        groups.append(re.match(r'^([A-Za-z]+)', archive).group(1).lower())

x = np.array(x).astype(np.float32) / 255.0
y = np.array(y)
groups = np.array(groups)

print("X format:", x.shape)
print("Y format:", y.shape)
print("Participantes únicos:", len(np.unique(groups)))
print()

kf = GroupKFold(n_splits=5)
folds = list(kf.split(x, y, groups))

modelos = {
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
    "Naive Bayes": GaussianNB(),
}

for k in range(1, 11):
    modelos["KNN (k=" + str(k) + ")"] = KNeighborsClassifier(n_neighbors=k)

best_accuracy = 0
best_std = 0
best_model = ""

for nome, modelo in modelos.items():
    scores = cross_val_score(modelo, x, y, cv=folds, scoring='accuracy')
    accuracy = scores.mean()
    std = scores.std()
    print(f"{nome} | Accuracy: {accuracy * 100:.2f}% | Std: {std * 100:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_std = std
        best_model = nome

print(
    f"\nMelhor algoritmo: {best_model} | "
    f"Accuracy: {best_accuracy * 100:.2f}% | "
    f"Std: {best_std * 100:.2f}%"
)