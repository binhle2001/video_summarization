import os
import shutil

from .settings import *

shutil.rmtree(DATA_OF_FRAME_DIR, ignore_errors=True)
os.makedirs(DATA_OF_FRAME_DIR, exist_ok=True)

def overlap(bbox1, bbox2):
    ymin1, ymax1, xmin1, xmax1 = bbox1
    ymin2, ymax2, xmin2, xmax2 = bbox2
    if xmin1 < xmax2 and xmax1 > xmin2 and ymin1 < ymax2 and ymax1 > ymin2:
        return True
    return False
def get_bbox(coordinate_file_path):
    with open(coordinate_file_path, 'r', encoding="utf-8") as file:
        content = file.read()
    ymin, ymax, xmin, xmax, class_name, frame_no = content.split('\t')
    xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
    return (ymin, ymax, xmin, xmax)


def get_object(max_object_in_a_frame = 50):
    objects = os.listdir(OBJECT_COORDINATE_PATH)
    frame_all_object = {obj:len(os.listdir(f"{OBJECT_COORDINATE_PATH}/" + obj)) for obj in sorted(objects, key=int)}
    objects = sorted(objects, key=int)
    check_object = [i for i in frame_all_object]
    num_object = len(frame_all_object)
    frame_no = 0
    while num_object > 0: 
        rectangles = []
        k = 0
        f = open(f"{DATA_OF_FRAME_DIR}/{frame_no:05d}.txt", "w", encoding='utf-8')
        for object_id in objects:
            if k == max_object_in_a_frame: 
                break
            if frame_all_object[object_id] == 0:
                continue
            if object_id not in check_object and frame_all_object[object_id] >= 0:
                image_file_path = f"{OBJECT_IMAGE_PATH}/" + object_id + "/" + os.listdir(f"{OBJECT_IMAGE_PATH}/{object_id}")[0 - frame_all_object[object_id]] + "\n"
                f.write(image_file_path)
                coordinate_file = f"{OBJECT_COORDINATE_PATH}/" + object_id + "/" + os.listdir(f"{OBJECT_COORDINATE_PATH}/{object_id}")[0 - frame_all_object[object_id]]
                box = get_bbox(coordinate_file)
                rectangles.append(box)
                k += 1
                frame_all_object[object_id] -= 1
                if frame_all_object[object_id] == 0:
                    num_object -= 1
        for object_id in check_object:
            if k >= max_object_in_a_frame:
                break
            is_overlap = False
            coordinate_file = f"{OBJECT_COORDINATE_PATH}/" + object_id + "/" + os.listdir(f"{OBJECT_COORDINATE_PATH}/{object_id}")[0 - frame_all_object[object_id]]
            box = get_bbox(coordinate_file)
            for rec in rectangles:
                if overlap(box, rec):
                    is_overlap = True
                    break
            if not is_overlap:
                check_object.remove(object_id)
                image_file_path = f"{OBJECT_IMAGE_PATH}/" + object_id + "/" + os.listdir(f"{OBJECT_IMAGE_PATH}/{object_id}")[0 - frame_all_object[object_id]] + "\n"
                f.write(image_file_path)
                k += 1
                frame_all_object[object_id] -= 1
                rectangles.append(box)
                if frame_all_object[object_id] == 0:
                    num_object -= 1
        f.close()
        frame_no += 1


