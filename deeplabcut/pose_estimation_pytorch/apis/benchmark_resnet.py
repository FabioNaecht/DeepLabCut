import imageio as iio
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import time

# Load pre-trained ResNet-50 model
model = resnet50(pretrained=True)
model.eval()  # Set to evaluation mode

# Set up the device
# device mps (macbook here!)
device = "mps"
# device = "cpu"
model.to(device)

# Define image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert image to PIL format
    transforms.Resize((224, 224)),  # Resize to 224x224 for ResNet-50
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet standards
])

# Initialize video reader
video_path = "/Users/fabionaecht/Documents/PhD/dlc/tensorflow_pytorch/refined/20230424_nn200_tip-sam/test-videos/2023_04_22_15_00_tracking_trial_control_fabio_image_stack1.avi"
video_path = "/Users/fabionaecht/Documents/PhD/dlc/tensorflow_pytorch/refined/20230424_nn200_tip-sam/test-videos/2023_04_22_15_00_tracking_trial_control_fabio_image_stack1.avi"
video_reader = iio.get_reader(video_path)
frame_number_total = video_reader.count_frames()
print(f"Total number of frames: {frame_number_total}")

# Process first 10 frames and measure prediction time
for frame_number in range(10):
    print(f"Processing frame #{frame_number}")

    # Read the current frame
    image = video_reader.get_data(frame_number)

    # Apply transformations
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)

    # Measure prediction time
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    end_time = time.time()

    # Calculate time taken in milliseconds
    execution_time_ms = (end_time - start_time) * 1000
    print(f"Prediction time for frame {frame_number}: {execution_time_ms:.2f} ms")
