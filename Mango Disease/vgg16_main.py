import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt  # Fix: Import plt for image display

# Define the VGG16 model with Transfer Learning (same as used during training)
class VGG16TransferLearning(nn.Module):
    def __init__(self, num_classes):
        super(VGG16TransferLearning, self).__init__()
        self.vgg16 = models.vgg16(weights='IMAGENET1K_V1')  # Use current torchvision behavior

        # Freeze the feature layers of VGG16
        for param in self.vgg16.features.parameters():
            param.requires_grad = False

        # Modify the classifier layer to match the number of classes
        self.vgg16.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg16(x)

# Load the trained model (with 8 output classes, as per the checkpoint)
model = VGG16TransferLearning(num_classes=8)  # Set num_classes to 8

# Load model weights
model.load_state_dict(torch.load('VGG16_MODEL.pth'))
model.eval()

# Move the model to the correct device (CUDA or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the same transformations used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Change to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to predict the class of an image
def predict_image(image_path):
    # Open the image using PIL
    image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format
    
    # Apply the transformations
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Move the image to the same device as the model
    image = image.to(device)

    # Make the prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
    
    # Get the predicted class label
    predicted_class = predicted.item()
    
    # Map the predicted class index back to the class name
    class_names = ['Anthracnose', 'Bacterrial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']  # Replace with your actual class names
    predicted_class_name = class_names[predicted_class]
    
    return predicted_class_name

# Function to display the image and its prediction
def show_prediction(predicted_class,image_path):
   
   

    # Load the class names (from training set)
    class_names = sorted(os.listdir('dataset/train'))  # Sort to ensure consistent order
    class_name=predicted_class

    # Display the image and prediction
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(f'Predicted: {class_name}')
    plt.axis('off')
    plt.show()

# Take input from the user (file path)
image_path = input("Enter the path to the image: ")

# Predict the class
predicted_class = predict_image(image_path)
print(f'The predicted class for the image is: {predicted_class}')



show_prediction(predicted_class,image_path)
