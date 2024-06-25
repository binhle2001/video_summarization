import cv2
import os
import shutil
from .settings import *

shutil.rmtree(BACKGROUND_DIR, ignore_errors=True)
os.makedirs(BACKGROUND_DIR, exist_ok=True)

def extract_background_mog2(video_path):
    """
    Trích xuất nền từ video giám sát bằng phương pháp MOG2.
    
    Args:
    video_path (str): Đường dẫn tới video giám sát.
    
    Returns:
    numpy.ndarray: Ảnh nền được trích xuất.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Không thể mở video")

    # Sử dụng BackgroundSubtractorMOG2
    fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=16, detectShadows=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fgmask = fgbg.apply(frame)

    # Trích xuất nền
    background = fgbg.getBackgroundImage()
    
    cap.release()
    cv2.imwrite(BACKGROUND_PATH, background)
    return background




# Lưu ảnh nền



