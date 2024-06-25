import cv2
import os
import shutil
from datetime import datetime, timedelta

from .settings import *

shutil.rmtree(OUTPUT_FRAMES_DIR, ignore_errors=True)
os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

def overlay_images(background, overlay, position=(0, 0), alpha=0.5):
    """
    Overlay the overlay image on the background image at a specific position with transparency alpha.
    
    Args:
    background (numpy.ndarray): The background image.
    overlay (numpy.ndarray): The overlay image.
    position (tuple): The (x, y) position to place the overlay on the background.
    alpha (float): The transparency of the overlay image (0: completely transparent, 1: completely opaque).
    
    Returns:
    numpy.ndarray: The resulting image after overlay.
    """
    x, y = position
    h, w = overlay.shape[0], overlay.shape[1]
    
    # Ensure overlay image does not exceed background image size
    if x + w > background.shape[1] or y + h > background.shape[0]:
        raise ValueError("Overlay image exceeds background image size at the specified position.")
    
    # Create a region of interest (ROI) on the background image
    roi = background[y:y+h, x:x+w]
    
    # Blend images using alpha transparency
    blended = cv2.addWeighted(roi, 1 - alpha, overlay, alpha, 0)
    
    # Place blended result back into the background image
    background[y:y+h, x:x+w] = blended
    
    return background

def is_overlap(rect1, rect2):
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2
    
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

def calculate_time(frame_no, start_time, fps):
    total_seconds = frame_no / fps
    time_delta = timedelta(seconds=total_seconds)
    frame_time = start_time + time_delta
    return frame_time.strftime('%H:%M:%S')

def put_object_to_frame(start_time, fps):
    start_time = datetime.strptime(start_time, "%H:%M:%S")
    k = 0
    background = cv2.imread(BACKGROUND_PATH)
    for file_name in os.listdir(DATA_OF_FRAME_DIR):
        background_img = background.copy()
        with open(f"{DATA_OF_FRAME_DIR}/" + file_name, "r", encoding="utf-8") as file:
            content = file.read().split("\n")
        
        rectangles = []
        for image_file in content:
            if image_file:
                overlay_img = cv2.imread(image_file)
                coor_file = image_file.replace(OBJECT_IMAGE_PATH, OBJECT_COORDINATE_PATH).replace(".jpg", ".txt")
                position_string = open(coor_file, "r").read()
                ymin, ymax, xmin, xmax, class_name, frame_no = position_string.split("\t")
                ymin, ymax, xmin, xmax, frame_no = int(ymin), int(ymax), int(xmin), int(xmax), int(frame_no)
                position = (xmin, ymin)
                
                current_rect = (xmin, ymin, xmax, ymax)
                alpha = 1.0
                
                for rect in rectangles:
                    if is_overlap(current_rect, rect):
                        alpha = 0.5
                        break
                
                rectangles.append(current_rect)
                background_img = overlay_images(background_img, overlay_img, position, alpha)
                
                # Calculate time and overlay it
                frame_time = calculate_time(frame_no, start_time, fps)
                text_position = (xmin + (xmax - xmin) // 2, ymin + (ymax - ymin) // 2)
                cv2.putText(background_img, frame_time, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        cv2.imwrite(f"{OUTPUT_FRAMES_DIR}/{k:05d}.jpg", background_img)
        k += 1
