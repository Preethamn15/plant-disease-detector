##import numpy as np
##import os
##from glob import glob
##import matplotlib.pyplot as plt
##import seaborn as sns
##from keras.layers import Dense, Flatten
##from keras.models import Model, load_model
##from keras.applications.vgg16 import VGG16, preprocess_input
##from tensorflow.keras.preprocessing.image import ImageDataGenerator
##from sklearn.metrics import confusion_matrix, classification_report
##from tensorflow.keras.callbacks import ModelCheckpoint
##
### Define dataset path
##data_path = './dataset/'
##
### Image size and parameters
##IMAGE_SIZE = [128, 128]
##epochs = 50
##batch_size = 128
##
### Load VGG16 model without top layers
##vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
##for layer in vgg.layers:
##    layer.trainable = False
##
### Custom layers
##x = Flatten()(vgg.output)
##prediction = Dense(len(burn_classes), activation='softmax')(x)
##model = Model(inputs=vgg.input, outputs=prediction)
##model.summary()
##
### Compile model
##model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
##
### Data augmentation with validation split
##gen = ImageDataGenerator(
##    rotation_range=20,
##    width_shift_range=0.1,
##    height_shift_range=0.1,
##    shear_range=0.1,
##    zoom_range=0.2,
##    horizontal_flip=True,
##    vertical_flip=True,
##    rescale=1./255,
##    preprocessing_function=preprocess_input,
##    validation_split=0.2  
##)
##
### Create generators with subset
##train_generator = gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, batch_size=batch_size, class_mode='categorical', subset="training")
##valid_generator = gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, batch_size=batch_size, class_mode='categorical', subset="validation", shuffle=False)
##
### Model checkpoint
##checkpoint = ModelCheckpoint('best_vgg16_model.h5', save_best_only=True, monitor='val_loss', mode='min')
##
### Train model
##r = model.fit(train_generator, validation_data=valid_generator, epochs=epochs, callbacks=[checkpoint])
##
### Save final model
##model.save("vgg16_trained_model.h5")
##print("Model saved as vgg16_trained_model.h5")
##
### Plot accuracy
##plt.plot(r.history['accuracy'], label='Train Accuracy')
##plt.plot(r.history['val_accuracy'], label='Validation Accuracy')
##plt.title('Model Accuracy')
##plt.ylabel('Accuracy')
##plt.xlabel('Epoch')
##plt.legend()
##plt.savefig('VGG16_model_accuracy.png')
##plt.show()
##
### Plot loss
##plt.plot(r.history['loss'], label='Train Loss')
##plt.plot(r.history['val_loss'], label='Validation Loss')
##plt.title('Model Loss')
##plt.ylabel('Loss')
##plt.xlabel('Epoch')
##plt.legend()
##plt.savefig('VGG16_model_loss.png')
##plt.show()
##
### Load best model
##best_model = load_model('best_vgg16_model.h5')
##valid_generator.reset()
##Y_pred = best_model.predict(valid_generator, verbose=1)
##y_pred = np.argmax(Y_pred, axis=1)
##
### Get true labels
##y_true = valid_generator.classes
##class_labels = list(valid_generator.class_indices.keys())
##
### Compute confusion matrix
##cm = confusion_matrix(y_true, y_pred)
##
### Classification Report
##class_report = classification_report(y_true, y_pred, target_names=class_labels)
##print("\nClassification Report:")
##print(class_report)
##
### Save classification report
##with open("classification_report.txt", "w") as file:
##    file.write(class_report)
##
### Plot confusion matrix
##plt.figure(figsize=(10,8))
##sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
##plt.xlabel('Predicted Label')
##plt.ylabel('True Label')
##plt.title('Confusion Matrix')
##plt.savefig('VGG16_confusion_matrix.png')
##plt.show()









import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Dense, Flatten
from keras.models import Model, load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import ModelCheckpoint

# Define dataset path
data_path = './dataset/'

# Image size and parameters
IMAGE_SIZE = [128, 128]
epochs = 50
batch_size = 128

# Load VGG16 model without top layers
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False

# Data augmentation with validation split
gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255,
    preprocessing_function=preprocess_input,
    validation_split=0.2  
)

# Create generators with subset
train_generator = gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, batch_size=batch_size, class_mode='categorical', subset="training")
valid_generator = gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, batch_size=batch_size, class_mode='categorical', subset="validation", shuffle=False)

# Get number of classes dynamically
num_classes = len(train_generator.class_indices)

# Custom layers
x = Flatten()(vgg.output)
prediction = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model checkpoint
checkpoint = ModelCheckpoint('best_vgg16_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# Train model
r = model.fit(train_generator, validation_data=valid_generator, epochs=epochs, callbacks=[checkpoint])

# Save final model
model.save("vgg16_trained_model.h5")
print("Model saved as vgg16_trained_model.h5")

# Plot accuracy
plt.plot(r.history['accuracy'], label='Train Accuracy')
plt.plot(r.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('VGG16_model_accuracy.png')
plt.show()

# Plot loss
plt.plot(r.history['loss'], label='Train Loss')
plt.plot(r.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('VGG16_model_loss.png')
plt.show()

# Load best model
best_model = load_model('best_vgg16_model.h5')
valid_generator.reset()
Y_pred = best_model.predict(valid_generator, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)

# Get true labels
y_true = valid_generator.classes
class_labels = list(valid_generator.class_indices.keys())

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Classification Report
class_report = classification_report(y_true, y_pred, target_names=class_labels)
print("\nClassification Report:")
print(class_report)

# Save classification report
with open("classification_report.txt", "w") as file:
    file.write(class_report)

# Plot confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('VGG16_confusion_matrix.png')
plt.show()
