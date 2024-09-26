import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from model import U2NET  # Ensure the model import path is correct
import os

model_name = 'u2net'
model_dir = os.path.join(os.getcwd(), 'saved_models', model_name+'_human_seg', model_name + '_human_seg.pth')

# Normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

# Generate mask for a single frame
def generate_mask_for_frame(frame, model):
    # Transform for U2Net
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
    ])

    # Convert frame (OpenCV format) to PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    original_size = image.size  # Get original image size
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.cuda() if torch.cuda.is_available() else input_tensor

    # Forward pass through U2Net
    with torch.no_grad():
        d1, *_ = model(input_tensor)

    # Normalize and get prediction
    pred = normPRED(d1[:, 0, :, :])

    # Convert prediction to mask (0 or 255)
    mask = (pred > 0.5).float() * 255.0  # Thresholding to create binary mask
    mask = mask.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU

    # Resize the mask back to the original image size
    mask_image = Image.fromarray(mask.astype(np.uint8))  # Convert mask to image
    mask_image = mask_image.resize(original_size, Image.BILINEAR)  # Resize to original dimensions

    # Convert back to OpenCV format
    mask_frame = cv2.cvtColor(np.array(mask_image), cv2.COLOR_GRAY2BGR)
    return mask_frame

# Calculate the center point of the person from the mask frame
def get_person_center(mask_frame):
    gray_mask = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Find the largest contour which represents the person
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Calculate center point of bounding box
    center_x = x + w // 2
    center_y = y + h // 2
    return center_x, center_y

# Smooth the center point using Exponential Moving Average (EMA)
def smooth_center(prev_center, new_center, alpha=0.2):
    if prev_center is None:
        return new_center
    smoothed_x = int(alpha * new_center[0] + (1 - alpha) * prev_center[0])
    smoothed_y = int(alpha * new_center[1] + (1 - alpha) * prev_center[1])
    return smoothed_x, smoothed_y

# Crop frame based on center point to fit a 9:16 frame window
def crop_frame(frame, center_x, center_y, output_width, output_height):
    frame_height, frame_width = frame.shape[:2]
    
    # Calculate top-left corner of the crop area
    x1 = max(0, center_x - output_width // 2)
    y1 = max(0, center_y - output_height // 2)
    
    # Make sure the crop area doesn't exceed frame boundaries
    x2 = min(frame_width, x1 + output_width)
    y2 = min(frame_height, y1 + output_height)
    
    # Adjust top-left corner if necessary to ensure desired size
    x1 = max(0, x2 - output_width)
    y1 = max(0, y2 - output_height)
    
    # Crop the frame
    cropped_frame = frame[y1:y2, x1:x2]
    return cropped_frame

def generate_cropped_video(video_path, model_path, output_video_path, duration=10):
    # Load U2Net model
    print("Loading U2Net model...")
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    net.eval()

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Calculate number of frames to process
    num_frames_to_process = int(fps * duration)

    # Define the 9:16 frame size (e.g., 720x1280 or scaled proportionally)
    output_height = min(1280, original_height)
    output_width = int(output_height * 9 / 16)

    # Define codec and create VideoWriter for output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

    frame_count = 0
    prev_center = None
    # for frame_count in tqdm(range(num_frames_to_process), desc="Processing Frames", unit="frame"):
    while cap.isOpened() and frame_count < num_frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        # Generate mask for the frame
        mask_frame = generate_mask_for_frame(frame, net)
        
        # Get center point of the person
        person_center = get_person_center(mask_frame)
        
        if person_center:
            # Smooth the center point using Exponential Moving Average
            center_x, center_y = smooth_center(prev_center, person_center)
            prev_center = (center_x, center_y)
        else:
            # If no person detected, use center of the frame as fallback
            center_x, center_y = original_width // 2, original_height // 2

        # Crop frame based on center point
        cropped_frame = crop_frame(frame, center_x, center_y, output_width, output_height)
        
        # Write the cropped frame to the output video
        out.write(cropped_frame)

        frame_count += 1
        print(f"Processed frame {frame_count}/{num_frames_to_process}")

    # Release resources
    cap.release()
    out.release()
    print(f"Cropped video saved to: {output_video_path}")

def main():
    video_path = 'output_clip_18.mp4'  # Change to your input video path
    model_path = model_dir  # Change to your model path
    output_video_path = 'output_frame_video.mp4'  # Path for the output video
    duration = 28  # Duration in seconds to process
    generate_cropped_video(video_path, model_path, output_video_path, duration)

if __name__ == "__main__":
    main()