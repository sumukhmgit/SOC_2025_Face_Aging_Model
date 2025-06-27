import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])

# TODO: Compile with adam + mse
model.compile(optimizer='adam', loss='mse')
model.summary()



X = np.random.randn(100, 5)
y = np.random.randn(100, 1)

#TODO: Train the model for 3 epochs
model.fit(X, y, epochs=3)

X_test = np.random.randn(5, 5)



# 🚧 TODO: Make predictions
predictions = model.predict(X_test)

print("Predictions:", predictions)



# Save the model
model.save("my_model.keras")#write here

# TODO: Load it again
new_model = tf.keras.models.load_model("my_model.keras")
print("Model reloaded successfully!")