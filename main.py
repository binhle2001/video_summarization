import cv2
from source.tracking_object import track_objects
from source.concate_to_video import make_video_from_frames
from source.extracting_background import extract_background_mog2
from source.filter_object import clean_object
from source.getting_object_for_frame import get_object
from source.putting_object_to_frame import put_object_to_frame



def main(video_path, start_time):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    extract_background_mog2(video_path)
    track_objects(cap)
    clean_object()
    get_object(max_object_in_a_frame=50)
    put_object_to_frame(start_time=start_time, fps=fps)
    make_video_from_frames(fps = fps)
    
if __name__ == "__main__":
    video_path = "output_video.mp4"
    start_time = "09:30:00"
    main(video_path, start_time)
