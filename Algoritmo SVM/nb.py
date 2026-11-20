# Code when I use Naive Bayes
import os
import re
import cv2
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

x = []
y = []
groups = []

dataset_path = "dataset"

classe = {
    "Drowsy": 0,
    "Non Drowsy": 1
}

# Limit the number of images to process for each class
limit = 10000

SEED = 42

# Loop through each class and read the images
for classe_nome, label in classe.items():

    folder = os.path.join(dataset_path, classe_nome)

    archives = os.listdir(folder)
    rng = np.random.RandomState(SEED)
    rng.shuffle(archives)

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

            # Participant code
            participant = re.match(
                r'^([A-Za-z]+)',
                archive
            ).group(1).lower()

            groups.append(participant)

# Convert the lists to numpy arrays
x = np.array(x)
y = np.array(y)
groups = np.array(groups)

# Normalize the pixels to the range [0,1]
x = x.astype(np.float32) / 255.0

print("X format: ", x.shape)
print("Y format: ", y.shape)
print("Participantes únicos: ", len(np.unique(groups)))

kf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

# Create the Naive Bayes model
model = GaussianNB()

# Execute StratifiedGroupKFold Cross Validation
scores = cross_val_score(
    model,
    x,
    y,
    groups=groups,
    cv=kf,
    scoring='accuracy'
)

print("\nScores:")
for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: {score * 100:.2f}%")

print(f"\nMean Accuracy: {scores.mean() * 100:.2f}%")
print(f"Standard Deviation: {scores.std() * 100:.2f}%")