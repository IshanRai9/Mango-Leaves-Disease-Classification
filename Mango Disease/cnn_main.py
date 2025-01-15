import torch
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt  # Fix: Import plt for image display


import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms



# Path to your saved model
model_path = 'CNN_MODEL.pth'  # Ensure this is the correct path to your model

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 14 * 14, 128)  # Adjust based on pooling
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 14 * 14)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the trained model
model = CNNModel(num_classes=len(os.listdir('dataset/train')))  # Ensure the number of classes is correct
model.load_state_dict(torch.load(model_path))
model.eval()  # Set model to evaluation mode

# Device setup (same as the training setup)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the image transformation
img_size = 128  # Same size as used in training
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Optional normalization
])

# Function to predict the class of an input image
def predict_image(image_path):
    # Load the image
    image = Image.open(image_path)
    image = image.convert("RGB")  # Ensure the image is in RGB format

    # Apply transformations
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move the image to the same device as the model
    image = image.to(device)

    # Make a prediction
    with torch.no_grad():  # No need to calculate gradients for inference
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)

    return predicted_class.item()

# Function to display the image and its prediction
def show_prediction(image_path):
    # Predict the class
    predicted_class = predict_image(image_path)

    # Load the class names (from training set)
    class_names = sorted(os.listdir('dataset/train'))  # Sort to ensure consistent order
    class_name = class_names[predicted_class]

    # Display the image and prediction
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(f'Predicted: {class_name}')
    plt.axis('off')
    plt.show()

# Example usage:
# Take input from the user (file path)
image_path = input("Enter the path to the image: ")
show_prediction(image_path)
