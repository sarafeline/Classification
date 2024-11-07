import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# Directory path
data_dir = 'C:\\Users\\user\\Documents\\Training Data\\CatsAndDogs'

# Image Data Generator to process data_dir as a bunch of images
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

# Training Data
train_data = datagen.flow_from_directory(directory=data_dir,
                                         target_size=(224, 224),
                                         batch_size=32,
                                         class_mode='binary',
                                         subset='training')

# Validation Data
val_data = datagen.flow_from_directory(directory=data_dir,
                                       target_size=(224, 224),
                                       batch_size=32,
                                       class_mode='binary',
                                       subset='validation')

# Model definition using pre-trained VGG16
base_model = VGG16(include_top=False, input_shape=(224, 224, 3))
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, validation_data=val_data, epochs=10)

# Example prediction
sample_image = 'C:\\Users\\user\\Downloads\\cats\\kitty.jpg'  # Replace with actual image path
from tensorflow.keras.preprocessing import image
img = image.load_img(sample_image, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

prediction = model.predict(img_array)
print(f"Predicted Class: {'Dog' if prediction[0] >= 0.5 else 'Cat'}")
