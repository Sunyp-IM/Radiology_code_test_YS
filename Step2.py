#### Orgnizing and labeling the data for AI modeling

import os
import json
import shutil
from sklearn.model_selection import train_test_split

## Fing the image files with cancer
with open('all_graphs.json', 'r') as f:
    graphs_dict = json.load(f)


# Directories for cancer and no_cancer images
base_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
cancer_dir = os.path.join(base_dir, "2025_Programming_Test/Radiology_code_test_YS/cancer_modeling/cancer")
no_cancer_dir = os.path.join(base_dir, "2025_Programming_Test/Radiology_code_test_YS/cancer_modeling/no_cancer")
os.makedirs(cancer_dir, exist_ok=True)
os.makedirs(no_cancer_dir, exist_ok=True)

# Filter cancer and non-cancer images based on the graph database
original_image_dir = os.path.join(base_dir, 'ProgrammingTest_Data/png')
cancer_images = []
no_cancer_images = []
for patient_id, graph_data in graphs_dict.items():
    edges = graph_data.get("links", [])
    image_file = None
    has_cancer = False

    # Find the image file and check for a cancer diagnosis
    for edge in edges:
        if edge.get("relationship") == "HAS_IMAGE":
            image_file = edge.get("target")
        if edge.get("relationship") == "HAS_DIAGNOSIS" and "cancer" in edge.get("target", "").lower():
            has_cancer = True

    # Add the image path to the appropriate list
    if image_file:
        image_path = os.path.join(original_image_dir, image_file)
        if os.path.exists(image_path):
            if has_cancer:
                cancer_images.append(image_path)
            else:
                no_cancer_images.append(image_path)

# Copy images to cancer and no_cancer directories
for image_path in cancer_images:
    shutil.copy(image_path, os.path.join(cancer_dir, os.path.basename(image_path)))

for image_path in no_cancer_images:
    shutil.copy(image_path, os.path.join(no_cancer_dir, os.path.basename(image_path)))

# Dataset split directories
split_base_dir = os.path.join(base_dir, "2025_Programming_Test/Radiology_code_test_YS/cancer_modeling/split_dataset")
train_dir = os.path.join(split_base_dir, "train")
val_dir = os.path.join(split_base_dir, "val")
test_dir = os.path.join(split_base_dir, "test")

# Create subdirectories for train, val, and test
for split in [train_dir, val_dir, test_dir]:
    os.makedirs(os.path.join(split, "cancer"), exist_ok=True)
    os.makedirs(os.path.join(split, "no_cancer"), exist_ok=True)

# Get the file paths in the cancer and no_cancer directories
cancer_images = [os.path.join(cancer_dir, f) for f in os.listdir(cancer_dir) if os.path.isfile(os.path.join(cancer_dir, f))]
no_cancer_images = [os.path.join(no_cancer_dir, f) for f in os.listdir(no_cancer_dir) if os.path.isfile(os.path.join(no_cancer_dir, f))]

# Split datasets (80% train, 10% val, 10% test)
train_cancer, temp_cancer = train_test_split(cancer_images, test_size=0.2, random_state=42)
val_cancer, test_cancer = train_test_split(temp_cancer, test_size=0.5, random_state=42)

train_no_cancer, temp_no_cancer = train_test_split(no_cancer_images, test_size=0.2, random_state=42)
val_no_cancer, test_no_cancer = train_test_split(temp_no_cancer, test_size=0.5, random_state=42)

# Helper function to copy files to their respective directories
def copy_files(file_list, destination_dir):
    for file_path in file_list:
        shutil.copy(file_path, destination_dir)

# Copy files to train, val, test directories
copy_files(train_cancer, os.path.join(train_dir, "cancer"))
copy_files(val_cancer, os.path.join(val_dir, "cancer"))
copy_files(test_cancer, os.path.join(test_dir, "cancer"))

copy_files(train_no_cancer, os.path.join(train_dir, "no_cancer"))
copy_files(val_no_cancer, os.path.join(val_dir, "no_cancer"))
copy_files(test_no_cancer, os.path.join(test_dir, "no_cancer"))

print("Dataset preparation complete.")



#### Train the AI model

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os


# Image preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Load ResNet50 model pre-trained on ImageNet
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers for binary classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

# Save the trained model
model.save("cancer_model.h5")

print("Model training complete and saved to 'cancer_model.h5'")



#### Evaluate the model

# Load the test dataset
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
