# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Load your dataset (replace 'your_dataset.csv' with your file)
# Example:
data = pd.read_csv('Final_cleaned_preprocessed_DataSet.csv')
features = data.drop(columns=['isFraud'])
target = data['isFraud']

# For demonstration, creating a synthetic dataset
from sklearn.datasets import make_classification
features, target = make_classification(n_samples=1200, n_features=11, n_classes=2, random_state=42)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.25, random_state=42)

# Normalize features by scaling values between 0 and 1
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Initialize the neural network model
network = Sequential([
    Dense(72, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(36, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

# Compile the network
network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Configure early stopping
stop_training = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the network with early stopping
training_results = network.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=40, callbacks=[stop_training])

# Save the trained model in .keras format
network.save('optimized_model.keras')

print("Model training is complete. The model is saved as 'optimized_model.keras'.")

