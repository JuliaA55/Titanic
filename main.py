import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("data/train.csv")

df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]
df.dropna(inplace=True)

df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

X = df.drop('Survived', axis=1)
y = df['Survived']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("🔁 Навчання...")
history = model.fit(X_train, y_train, epochs=50, batch_size=16,
                    validation_data=(X_val, y_val), verbose=1)

loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n✅ Точність на валідації: {acc:.4f}")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Точність')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Похибка')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("titanic_training.png")
plt.show()
print("📊 Графік збережено як titanic_training.png")


def predict_passenger(pclass, sex, age, sibsp, parch, fare, embarked):
    sex = 1 if sex.lower() == 'male' else 0

    embark_mapping = {'c': 0, 'q': 1, 's': 2}
    embarked = embark_mapping.get(embarked.lower(), 2) 

    passenger = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    passenger_scaled = scaler.transform(passenger)

    probability = model.predict(passenger_scaled)[0][0]
    prediction = "вижив" if probability >= 0.5 else "не вижив"

    print(f"\n🧍‍♂️ Ймовірність виживання: {probability:.2%} → Пасажир {prediction}")


predict_passenger(3, 'male', 22, 1, 0, 7.25, 's') 
