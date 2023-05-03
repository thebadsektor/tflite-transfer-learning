import os
import tensorflow as tf
import tensorflow_hub as hub
from tflite_model_maker import ImageClassifierDataLoader as DataLoader
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
import matplotlib.pyplot as plt

# !pip uninstall tensorflow -y
# !pip install tensorflow==2.10.0
# py -3 -m venv .venv
# source .venv/Scripts/activate
# pip freeze > requirements.txt
# pip install pipenv
# pipenv install -r requirements.txt
# pip install --no-cache-dir tflite_model_maker

# Load your image dataset
image_path = "/data/train/raw/spoon"
data = DataLoader.from_folder(image_path)
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

# Load your existing MobileNetV2 model
base_model_path = "/data/models/object_labeler.tflite"

# Load the base model and get the feature extraction layer
base_model = tf.keras.models.load_model(base_model_path)
feature_extractor = base_model.get_layer('AvgPool_0a_7x7')

# Check if the layer is an instance of AveragePool2D
if not isinstance(feature_extractor, tf.keras.layers.AveragePooling2D):
    raise ValueError("The layer name 'AvgPool_0a_7x7' does not correspond to an AveragePool2D layer. Please check the layer name in Netron.")

# Freeze the top few layers of the base model
freeze_layers = 5 # Set this to the number of top layers you want to freeze
for layer in feature_extractor.layers[:freeze_layers]:
    layer.trainable = False

# Create a new model using transfer learning with your existing model
num_classes = len(train_data.index_to_label)  # Get the number of classes in your new dataset

# Remove the last few layers of the base model
base_model.layers.pop()  # Remove the last layer (likely the output layer)
base_model.layers.pop()  # Remove the second to last layer (likely the fully connected layer)

# Add new layers to match the number of classes in your new dataset
x = base_model.layers[-1].output
x = tf.keras.layers.Dense(128, activation='relu')(x)  # Add a new fully connected layer
x = tf.keras.layers.Dropout(0.2)(x)  # Add dropout for regularization
output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)  # Add a new output layer

# Create a new model with the updated layers
new_model = tf.keras.Model(inputs=base_model.inputs, outputs=output_layer)

# Train the new model
model_spec = image_classifier.ModelSpec(uri="https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4")
model_spec.input_image_shape = [224, 224]
model = image_classifier.create(
    train_data,
    model_spec=model_spec,
    validation_data=validation_data,
    epochs=10,
    use_augmentation=True,
    dropout_rate=0.2,
    warmup_steps=100,
    base_model=new_model
)

# Evaluate the new model's performance on the test dataset
loss, accuracy = model.evaluate(test_data)

# Export the new model
model.export(export_dir="/data/models/", export_format=ExportFormat.TFLITE)

# Quantize the model for optimization
config = QuantizationConfig.for_integer()  # This will quantize the model to uint8
model.export(export_dir="/data/models/", tflite_filename="object_labeler_uint8.tflite", quantization_config=config)
