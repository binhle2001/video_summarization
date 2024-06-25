import cv2
import os
from .settings import *

def make_video_from_frames(image_folder = OUTPUT_FRAMES_DIR, output_video = OUTPUT_VIDEO_PATH, fps = 15):

    # Lấy danh sách các tệp ảnh và sắp xếp chúng
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort()

    # Đảm bảo rằng danh sách ảnh không rỗng
    if not images:
        print("No images found in the directory.")
        exit()

    # Đọc kích thước của ảnh đầu tiên
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Đặt cấu hình video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec cho file .mp4
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Ghi từng ảnh vào video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Giải phóng tài nguyên
    video.release()
    cv2.destroyAllWindows()
    print(f"Video saved as {output_video}")
