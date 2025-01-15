import torch
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt  # Fix: Import plt for image display

# Device setup: Automatically use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image preprocessing (same as during training)
img_size = 128  # Image size (128x128)
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model (same architecture as during training)
model = models.resnet18(pretrained=False)  # We don't need to load pretrained weights here

# Correct the output layer to match the number of classes from your training (8 classes in this case)
num_classes = 8  # This should match the number of classes you trained on
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load the saved model
# Make sure the model file 'DEEP_MLP.pth' exists in the correct directory
model.load_state_dict(torch.load('Deep_MLP2.pth', map_location=device))  # Adjust path if necessary
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Function to predict class for an input image
def predict_image(image_path):
    # Open and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Make prediction
    with torch.no_grad():
        outputs = model(img)
        _, predicted_class = torch.max(outputs, 1)

    # Load the class labels
    # Replace this with your actual class names
    class_names = ['Anthracnose', 'Bacterrial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']  # Replace with your actual class names

    predicted_label = class_names[predicted_class.item()]

    return predicted_label



# Function to display the image and its prediction
def show_prediction(predicted_class,image_path):
   
   

    # Load the class names (from training set)
    class_names = sorted(os.listdir('archive/train'))  # Sort to ensure consistent order
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



