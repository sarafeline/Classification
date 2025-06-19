import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.optimizers import Adam

# Setup directories and parameters
data_dir = 'C:\\Users\\user\\Documents\\Training Data\\CatsAndDogs'
loss_threshold = 0.4
folds = 2

# Image Data Generator
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)
data_gen = datagen.flow_from_directory(data_dir, target_size=(224, 224), batch_size=32, class_mode='binary')
n_samples = data_gen.n

# Model creation function
def create_model(learning_rate=0.0001):
    base_model = VGG16(include_top=False, input_shape=(224, 224, 3))
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Cross-validation function
def run_kfold(folds, learning_rate=0.0001):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    losses = []
    for train_idx, val_idx in kf.split(np.arange(n_samples)):
        model = create_model(learning_rate)
        train_data = datagen.flow_from_directory(data_dir, target_size=(224, 224), batch_size=32, class_mode='binary', subset='training')
        val_data = datagen.flow_from_directory(data_dir, target_size=(224, 224), batch_size=32, class_mode='binary', subset='validation')
        history = model.fit(train_data, validation_data=val_data, epochs=10, verbose=1)
        losses.append(history.history['val_loss'][-1])
    return np.mean(losses)

# Initial run of cross-validation
avg_loss = run_kfold(folds)
print(f"Initial average loss: {avg_loss}")

# If average loss is high, update learning rate and retrain the model
if avg_loss > loss_threshold:
    print("High average loss detected, reducing learning rate.")
    new_learning_rate = 0.00001  # Reduce the learning rate
    avg_loss = run_kfold(folds, learning_rate=new_learning_rate)
    print(f"New average loss after updating learning rate: {avg_loss}")

# Final training and prediction if loss is acceptable
if avg_loss <= loss_threshold:
    final_model = create_model(learning_rate=new_learning_rate)
    final_data = datagen.flow_from_directory(data_dir, target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=True)
    final_model.fit(final_data, epochs=10)
    
    # Make a prediction on a sample image
    sample_image_path = 'C:\\Users\\user\\Downloads\\cats\\kitty.jpg'
    from tensorflow.keras.preprocessing import image
    img = image.load_img(sample_image_path, target_size=(224, 224))
    img_array = np.expand_dims(preprocess_input(image.img_to_array(img)), axis=0)
    prediction = final_model.predict(img_array)
    print(f"Predicted Class: {'Dog' if prediction[0] >= 0.5 else 'Cat'}")
else:
    print("High average loss persists. Consider further tuning.")
