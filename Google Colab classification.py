import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from google.colab import files
import zipfile
import os
import shutil

# Upload the cats and dogs zip files
print("Please upload the 'cats' zip file:")
uploaded_cats = files.upload()

print("Please upload the 'dogs' zip file:")
uploaded_dogs = files.upload()

# Create a directory for the dataset
data_dir = '/content/CatsAndDogs'
os.makedirs(data_dir, exist_ok=True)
cats_dir = os.path.join(data_dir, 'cats')
dogs_dir = os.path.join(data_dir, 'dogs')
os.makedirs(cats_dir, exist_ok=True)
os.makedirs(dogs_dir, exist_ok=True)

# Extract the uploaded zip files into the respective directories
for filename in uploaded_cats.keys():
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(cats_dir)

for filename in uploaded_dogs.keys():
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(dogs_dir)

# Verify the directory structure
print(f'Cats directory: {os.listdir(cats_dir)[:5]}')  # Listing only first 5 files for brevity
print(f'Dogs directory: {os.listdir(dogs_dir)[:5]}')

# Image Data Generator
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

# Training data
train_data = datagen.flow_from_directory(directory=data_dir,
                                         target_size=(224, 224),
                                         batch_size=32,
                                         class_mode='binary',
                                         subset='training')

# Validation data
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
# Upload a sample image for prediction
uploaded_sample = files.upload()
sample_image = list(uploaded_sample.keys())[0]

# Load and preprocess the sample image
from tensorflow.keras.preprocessing import image
img = image.load_img(sample_image, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Predict the class of the sample image
prediction = model.predict(img_array)
print(f"Predicted Class: {'Dog' if prediction[0] >= 0.5 else 'Cat'}")
