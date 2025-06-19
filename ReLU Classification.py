•	import numpy as np
•	from tensorflow.keras import Sequential
•	from tensorflow.keras.layers import Dense
•	
•	# Example input
•	X = np.random.rand(10, 4)  # 10 samples, each with 4 features
•	
•	# Define the model
•	model = Sequential([
•	    Dense(units=25, activation='relu', input_shape=(4,)),
•	    Dense(units=15, activation='relu'),
•	    Dense(units=1, activation='sigmoid')
•	])
•	
•	# Compile the model
•	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
•	
•	# Generate random binary labels
•	Y = np.random.randint(0, 2, (10, 1))  # 10 samples, each with a binary label
•	
•	# Train the model
•	model.fit(X, Y, epochs=10)
•	
•	# Predict output
•	output = model.predict(X)
•	print(f"Predicted Output: {output}")
