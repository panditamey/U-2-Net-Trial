import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from model import U2NET  # Ensure the model import path is correct
import os

model_name='u2net'
model_dir = os.path.join(os.getcwd(), 'saved_models', model_name+'_human_seg', model_name + '_human_seg.pth')

# Normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

def generate_mask(image_path, model_path, output_mask_path):
    # Load U2Net model
    print("Loading U2Net model...")
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    net.eval()

    # Transform for U2Net
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # Get original image size
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.cuda() if torch.cuda.is_available() else input_tensor

    # Forward pass through U2Net
    with torch.no_grad():
        d1, *_ = net(input_tensor)

    # Normalize and get prediction
    pred = normPRED(d1[:, 0, :, :])

    # Convert prediction to mask (0 or 255)
    mask = (pred > 0.5).float() * 255.0  # Thresholding to create binary mask
    mask = mask.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU

    # Resize the mask back to the original image size
    mask_image = Image.fromarray(mask.astype(np.uint8))  # Convert mask to image
    mask_image = mask_image.resize(original_size, Image.BILINEAR)  # Resize to original dimensions

    # Save the mask as an image
    mask_image.save(output_mask_path)
    print(f"Mask saved to: {output_mask_path}")

def main():
    image_path = 'catch.jpg'  # Change to your input image path
    model_path = model_dir  # Change to your model path
    output_mask_path = 'catch_res2.png'  # Path for the output mask image
    generate_mask(image_path, model_path, output_mask_path)

if __name__ == "__main__":
    main()
