import pandas as pd
import numpy as np
import tensorflow as tf

print(tf.__version__)

# Load your CSV files, skipping the first row (header)
train_data = pd.read_csv('fashion-mnist_train.csv')
test_data = pd.read_csv('fashion-mnist_test.csv')

print('yaha aagya 1')

# Extract labels and pixel values
train_labels = train_data['label'].values
print(train_labels)
train_images = train_data.drop('label', axis=1).values.reshape(-1, 28, 28, 1)

test_labels = test_data['label'].values
test_images = test_data.drop('label', axis=1).values.reshape(-1, 28, 28, 1)

print('yaha aagya 2')

# Normalize pixel values to [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

print('yaha aagya 3')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

print('yaha aagya 4')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print('yaha aagya 5')

model.summary()

print('yaha aagya 6')

model.fit(train_images, train_labels, epochs=5)

print('yaha aagya 7')

test_loss = model.evaluate(test_images, test_labels)



print('yaha aagya 8')

model.save('fashion_mnist_cnn_model.keras')
