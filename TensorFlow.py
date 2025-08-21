import tensorflow as tf
print("TensorFlow version:", tf.__version__)
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
data = pd.read_csv(r"D:\StudentsPerformance.csv")
print(data.head())
X = data.drop('math score', axis=1)
y = data['math score']
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

print("Train dataset and test dataset prepared.")
print("Number of batches in train_dataset:", len(list(train_dataset)))
from keras import layers, models

model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  
])

model.summary()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("Model compiled successfully!")
import matplotlib.pyplot as plt
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=20
)

history.history['mae']
history.history['val_mae']
history.history['loss']
history.history['val_loss']

plt.plot(history.history['mae'], label='train_mae')
plt.plot(history.history['val_mae'], label='val_mae')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

import numpy
sample_index = 1
x_sample = X_test[sample_index].reshape(1, -1)
y_true = y_test.iloc[sample_index]
y_pred = model.predict(x_sample)

print(f"True value: {y_true}, Predicted: {y_pred[0][0]}") 

plt.bar(['True', 'Predicted'], [y_true, y_pred[0][0]])
plt.title('Test Sample Prediction')
plt.show()

y_pred_finetune = model.predict(x_sample)
plt.bar(['True', 'Predicted Fine-Tune'], [y_true, y_pred_finetune[0][0]])
plt.title('Test Sample Prediction - Fine-Tune Model')
plt.show()