import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- 1. Load and Pre-process the Data ---
data = pd.read_csv('fer2013.csv')

# Convert the pixel string into a numpy array
pixels = data['pixels'].tolist()
faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(48, 48)
    faces.append(face.astype('float32'))

faces = np.asarray(faces)
faces = np.expand_dims(faces, -1) # Add a channel dimension for the CNN

# Normalize the pixel values
faces /= 255.0

emotions = pd.get_dummies(data['emotion']).to_numpy()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.2, random_state=42)

# --- 2. Define the CNN Model Architecture ---
model = tf.keras.models.Sequential([
    # First Convolutional Block
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    # Second Convolutional Block
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    # Third Convolutional Block
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    # Flatten the results to feed into a Dense layer
    tf.keras.layers.Flatten(),

    # Dense (Fully Connected) Layers
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    # Output Layer - 7 neurons for 7 emotions
    tf.keras.layers.Dense(7, activation='softmax')
])

# --- 3. Compile the Model ---
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 4. Train the Model ---
print("\nStarting model training...")
# This will take some time depending on your hardware
history = model.fit(X_train, y_train,
                    batch_size=64,
                    epochs=30, # Start with 30 epochs, can increase later
                    validation_data=(X_test, y_test))

# --- 5. Save the Trained Model ---
print("\nTraining complete. Saving model...")
model.save('my_emotion_model.h5')
print("Model saved as my_emotion_model.h5")