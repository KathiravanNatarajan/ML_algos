#####################################################################################
"""
Convolutional Neural Networks (CNNs):
•	AlexNet
•	VGGNet (VGG16, VGG19)
•	GoogLeNet (Inception v1)
•	ResNet (Residual Networks)
•	DenseNet (Densely Connected Convolutional Networks)

Transformers:
•	Vision Transformer (ViT)
•	Data-efficient Image Transformers (DeiT)
•	Swin Transformer

Efficient Models:
•	MobileNet (MobileNetV1, V2, V3)
•	EfficientNet (B0-B7)
•	NASNet (Neural Architecture Search Network)

Ensemble and Hybrid Models:
•	Xception (Extreme Inception)
•	RegNet (Designing Network Design Spaces)
•	ConvNeXt

"""

######################################################################################

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.applications import VGG16, VGG19
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications import DenseNet121, ResNet50, MobileNetV3Large
from keras.applications import EfficientNetB0, Xception
import tensorflow_addons as tfa
from tensorflow.keras.applications import vit


def AlexNet(input_shape=(224, 224, 3), classes=1000):
    model = Sequential()
    model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    
    model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    
    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
 
    return model
	
def vgg16(input_shape=(224, 224, 3), classes=1000):
	# VGG16
	vgg16_model = VGG16(include_top=False, weights=None, input_shape=(224, 224, 3), classes=1000)
	model = models.Sequential()
	model.add(base_model)
	model.add(layers.Flatten())
	model.add(layers.Dense(256, activation='relu'))
	model.add(layers.Dense(classes, activation='softmax'))
	return model 
	
def vgg19(input_shape=(224, 224, 3), classes=1000):
	# VGG19
	vgg19_model = VGG19(include_top=False, weights=None, input_shape=(224, 224, 3), classes=1000)
	model = models.Sequential()
	model.add(base_model)
	model.add(layers.Flatten())
	model.add(layers.Dense(256, activation='relu'))
	model.add(layers.Dense(classes, activation='softmax'))
	return model 
	

def inception_v3(input_shape=(224, 224, 3), num_classes=1000): 

	base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

	# Add new layers on top of the base model
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation='relu')(x)  # Add a fully connected layer
	predictions = Dense(num_classes, activation='softmax')(x)  # Add the final output layer

	# Create a new model
	model = Model(inputs=base_model.input, outputs=predictions)
	return model

def resnet(input_shape=(224, 224, 3), num_classes=1000): 
	# Load the ResNet50 model, excluding the top fully connected layers
	base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

	# Create a new model and add the base ResNet50 model
	model = models.Sequential()
	model.add(base_model)

	# Add custom layers on top of the base model
	model.add(layers.GlobalAveragePooling2D())  # Global average pooling to reduce dimensions
	model.add(layers.Dense(256, activation='relu'))  # Fully connected layer with 256 units
	model.add(layers.Dropout(0.5))  # Dropout for regularization
	model.add(layers.Dense(10, activation='softmax'))  # Assuming 10 classes for the classification task
	return model 
	
	
def densenet(input_shape=(224, 224, 3), num_classes=1000): 
	# Load the DenseNet121 model, excluding the top fully connected layers
	base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

	# Create a new model and add the base DenseNet121 model
	model = models.Sequential()
	model.add(base_model)

	# Add custom layers on top of the base model
	model.add(layers.GlobalAveragePooling2D())  # Global average pooling to reduce dimensions
	model.add(layers.Dense(256, activation='relu'))  # Fully connected layer with 256 units
	model.add(layers.Dropout(0.5))  # Dropout for regularization
	model.add(layers.Dense(10, activation='softmax'))  # Assuming 10 classes for the classification task
	return model 
	
def mobilenetv3(input_shape=(224, 224, 3), num_classes=1000): 
	base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

	# Create a new model and add the base MobileNetV3Large model
	model = models.Sequential()
	model.add(base_model)

	# Add custom layers on top of the base model
	model.add(layers.GlobalAveragePooling2D())  # Global average pooling to reduce dimensions
	model.add(layers.Dense(256, activation='relu'))  # Fully connected layer with 256 units
	model.add(layers.Dropout(0.5))  # Dropout for regularization
	model.add(layers.Dense(10, activation='softmax'))  # Assuming 10 classes for the classification task
	return model 
	
def efficientnet(input_shape=(224, 224, 3), num_classes=1000): 
	# Load the EfficientNetB0 model, excluding the top fully connected layers
	base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

	# Create a new model and add the base EfficientNetB0 model
	model = models.Sequential()
	model.add(base_model)

	# Add custom layers on top of the base model
	model.add(layers.GlobalAveragePooling2D())  # Global average pooling to reduce dimensions
	model.add(layers.Dense(256, activation='relu'))  # Fully connected layer with 256 units
	model.add(layers.Dropout(0.5))  # Dropout for regularization
	model.add(layers.Dense(10, activation='softmax'))  # Assuming 10 classes for the classification task
	return model 
	
def nasnet(input_shape, num_classes=1000): 
	# Load the NASNetLarge model, excluding the top fully connected layers
	# image shape restrictions 
	base_model = NASNetLarge(weights='imagenet', include_top=False, input_shape=(331, 331, 3))

	# Create a new model and add the base NASNetLarge model
	model = models.Sequential()
	model.add(base_model)

	# Add custom layers on top of the base model
	model.add(layers.GlobalAveragePooling2D())  # Global average pooling to reduce dimensions
	model.add(layers.Dense(256, activation='relu'))  # Fully connected layer with 256 units
	model.add(layers.Dropout(0.5))  # Dropout for regularization
	model.add(layers.Dense(10, activation='softmax'))  # Assuming 10 classes for the classification task
	return model 
	
def exception(input_shape, num_classes=1000): 
	# Load the Xception model, excluding the top fully connected layers
	base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

	# Create a new model and add the base Xception model
	model = models.Sequential()
	model.add(base_model)

	# Add custom layers on top of the base model
	model.add(layers.GlobalAveragePooling2D())  # Global average pooling to reduce dimensions
	model.add(layers.Dense(256, activation='relu'))  # Fully connected layer with 256 units
	model.add(layers.Dropout(0.5))  # Dropout for regularization
	model.add(layers.Dense(10, activation='softmax'))  # Assuming 10 classes for the classification task
	return model 
	
def convnext(input_shape, num_classes=1000): 
	base_model = tfa.applications.ConvNeXt(input_shape=input_shape, include_top=include_top, weights=weights)
  
	# Create a new model and add the base ConvNeXt model
	model = models.Sequential()
	model.add(base_model)

	# Add custom layers on top of the base model
	model.add(layers.GlobalAveragePooling2D())  # Global average pooling to reduce dimensions
	model.add(layers.Dense(256, activation='relu'))  # Fully connected layer with 256 units
	model.add(layers.Dropout(0.5))  # Dropout for regularization
	model.add(layers.Dense(10, activation='softmax'))  # Assuming 10 classes for the classification task
	return model 

def vit(input_shape, num_classes=1000): 
	# Load the Vision Transformer (ViT) model, excluding the top fully connected layers
	base_model = vit.ViT16B(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

	# Create a new model and add the base ViT model
	model = models.Sequential()
	model.add(base_model)

	# Add custom layers on top of the base model
	model.add(layers.GlobalAveragePooling2D())  # Global average pooling to reduce dimensions
	model.add(layers.Dense(256, activation='relu'))  # Fully connected layer with 256 units
	model.add(layers.Dropout(0.5))  # Dropout for regularization
	model.add(layers.Dense(10, activation='softmax'))  # Assuming 10 classes for the classification task
	return model 
"""
# ----------------------------------------------------------------------------------------
model = AlexNet(input_shape=(224, 224, 3), classes=1000)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
"""
