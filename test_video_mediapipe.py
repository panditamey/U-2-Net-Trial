import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from model import U2NET  # Ensure the model import path is correct
import mediapipe as mp
import os
from tqdm import tqdm

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

# Detect persons using MediaPipe and track their movement
def detect_and_track_movement(frame, prev_landmarks, alpha=0.2):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    movement_scores = []
    centers = []

    if results.pose_landmarks:
        for landmark_set in results.pose_landmarks.landmark:
            # Get the position of the landmark
            cx, cy = int(landmark_set.x * frame.shape[1]), int(landmark_set.y * frame.shape[0])
            centers.append((cx, cy))
            
            # Calculate movement score if previous landmarks exist
            if prev_landmarks is not None:
                prev_cx, prev_cy = prev_landmarks
                movement = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                movement_scores.append(movement)
            else:
                movement_scores.append(0)
    
    # Get the person index with the highest movement score
    if len(movement_scores) > 0:
        max_movement_index = np.argmax(movement_scores)
        max_center = centers[max_movement_index]
    else:
        max_center = (frame.shape[1] // 2, frame.shape[0] // 2)  # Default to center

    return max_center, results.pose_landmarks if results.pose_landmarks else None

def generate_cropped_video(video_path, model_path, output_video_path, duration=10):
    print("Loading U2Net model...")
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    net.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    num_frames_to_process = int(fps * duration)

    output_height = min(1280, original_height)
    output_width = int(output_height * 9 / 16)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

    prev_center = None
    prev_landmarks = None
    for frame_count in tqdm(range(num_frames_to_process), desc="Processing Frames", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        mask_frame = generate_mask_for_frame(frame, net)
        person_center = get_person_center(mask_frame)

        # Detect and track movement using MediaPipe
        max_center, prev_landmarks = detect_and_track_movement(frame, prev_landmarks)

        if person_center:
            # Smooth the center point using Exponential Moving Average
            center_x, center_y = smooth_center(prev_center, max_center)
            prev_center = (center_x, center_y)
        else:
            center_x, center_y = original_width // 2, original_height // 2

        cropped_frame = crop_frame(frame, center_x, center_y, output_width, output_height)
        out.write(cropped_frame)

    cap.release()
    out.release()
    print(f"Cropped video saved to: {output_video_path}")

def main():
    video_path = 'match.mp4'
    model_path = model_dir
    output_video_path = 'output_frame_video.mp4'
    duration = 10
    generate_cropped_video(video_path, model_path, output_video_path, duration)

if __name__ == "__main__":
    main()
