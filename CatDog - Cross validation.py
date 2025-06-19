import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

# Directory path
data_dir = r'C:\Users\user\Documents\Training Data\CatsAndDogs'

# Image Data Generator with data augmentation
datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                             validation_split=0.2,
                             rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True)

# K-Fold Cross Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
accuracy_per_fold = []
loss_per_fold = []

# Placeholder for the number of images
num_images = len(os.listdir(os.path.join(data_dir, 'Cats'))) + len(os.listdir(os.path.join(data_dir, 'Dogs')))

# Split the data indices for cross-validation
for train_index, val_index in kf.split(np.arange(num_images)):
    print(f'Training fold {fold_no}...')

    # Training and Validation Generators. Batch_size should be less than the total number of images you have, because you’re splitting the umages in batches to improve memory usage
    train_generator = datagen.flow_from_directory(data_dir,
                                                  target_size=(224, 224),
                                                  batch_size=32,
                                                  class_mode='binary',
                                                  subset='training',
                                                  shuffle=True)

    val_generator = datagen.flow_from_directory(data_dir,
                                                target_size=(224, 224),
                                                batch_size=32,
                                                class_mode='binary',
                                                subset='validation',
                                                shuffle=True)

    # Initialize VGG16 model
    base_model = VGG16(include_top=False, input_shape=(224, 224, 3))
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model on the current fold
    history = model.fit(train_generator, validation_data=val_generator, epochs=10)

    # Store the results
    loss_per_fold.append(history.history['val_loss'][-1])
    accuracy_per_fold.append(history.history['val_accuracy'][-1])

    fold_no += 1

# Average performance metrics
print(f'Average Validation Loss: {np.mean(loss_per_fold)}')
print(f'Average Validation Accuracy: {np.mean(accuracy_per_fold)}')

# Example prediction
sample_image_path ='C:\\Users\\user\\Downloads\\cats\\kitty.jpg'  # Replace with actual image path
img = image.load_img(sample_image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

prediction = model.predict(img_array)
print(f"Predicted Class: {'Dog' if prediction[0] >= 0.5 else 'Cat'}")
