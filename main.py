import os
import cv2
import numpy as np
import glob
import torch
from torch.autograd import Variable
from torchvision import transforms
from model import U2NET  # Ensure the model import path is correct

# Normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

# Get video frames for the first 10 seconds
def get_all_frames(video_path, max_duration=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(fps * max_duration)

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

# Function to detect the ball based on color
def detect_ball(frame):
    # Convert to HSV color space for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the color range for the basketball (adjust these values)
    lower_color = np.array([10, 100, 100])  # Lower range for orange color
    upper_color = np.array([25, 255, 255])  # Upper range for orange color

    # Create a mask for the basketball
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Find contours of the detected ball
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour which is assumed to be the basketball
        ball_contour = max(contours, key=cv2.contourArea)
        # Get the center of the ball
        M = cv2.moments(ball_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return np.array([cX, cY])
    
    return None

# Process video using U2Net model
def process_video(video_path, output_video_path, model_path):
    # Load U2Net model
    print("Loading U2Net model...")
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    net.eval()

    # Transform for U2Net
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((320, 320)), transforms.ToTensor()])

    # Get video frames for 10 seconds
    frames = get_all_frames(video_path, max_duration=10)
    
    output_frames = []

    # Process each frame
    for i, frame in enumerate(frames):
        print(f"Processing frame {i + 1}/{len(frames)}")
        ball_position = detect_ball(frame)

        if ball_position is not None:
            print(f"Ball detected at: {ball_position}")  # Debugging output
            cropped_frame = crop_to_ball(frame, ball_position)
            output_frames.append(cropped_frame)
        else:
            print("No ball detected, skipping frame.")

    # Check if output_frames is not empty
    if output_frames:
        save_output_video(output_frames, output_video_path, fps=30)
    else:
        print("No output frames to save.")

# Crop frame around the ball
def crop_to_ball(frame, ball_position, output_size=(1080, 1920)):
    height, width, _ = frame.shape
    crop_width, crop_height = output_size
    x_center, y_center = ball_position

    # Ensure the cropping area is within the frame
    x_start = max(0, x_center - crop_width // 2)
    y_start = max(0, y_center - crop_height // 2)
    
    # Adjust for boundaries
    x_end = min(width, x_start + crop_width)
    y_end = min(height, y_start + crop_height)

    # If cropping exceeds the bounds, adjust the start positions
    if x_end - x_start < crop_width:
        if x_start == 0:
            x_end = min(crop_width, width)
        else:
            x_start = max(0, x_end - crop_width)

    if y_end - y_start < crop_height:
        if y_start == 0:
            y_end = min(crop_height, height)
        else:
            y_start = max(0, y_end - crop_height)

    return frame[y_start:y_end, x_start:x_end]

# Save the output video
def save_output_video(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR
    out.release()

def main():
    video_path = 'input.mp4'  # Change to your input video path
    model_path = 'u2net.pth'  # Change to your model path
    output_video_path = 'output_video.mp4'  # Path for the combined output video
    process_video(video_path, output_video_path, model_path)

if __name__ == "__main__":
    main()
