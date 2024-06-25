import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import models
from torchvision import transforms
from PIL import Image
import torch
import os
import shutil
from sklearn.metrics.pairwise import cosine_similarity
from .settings import *


model = YOLO("model_detect_vehicle.pt", task="detect")
model.to("cuda")


# Xóa các thư mục dữ liệu cũ
shutil.rmtree(OBJECT_IMAGE_PATH, ignore_errors=True)
shutil.rmtree(OBJECT_COORDINATE_PATH, ignore_errors=True)
os.makedirs("data_image", exist_ok=True)
os.makedirs("data_coordinate", exist_ok=True)

feature_map = [None for _ in range(MAX_OBJECT)]
image_map = [None for _ in range(MAX_OBJECT)]
coordinate_webcam = [(0, 0) for _ in range(MAX_OBJECT)]
check_object = {}
resnet18 = models.resnet50(pretrained=True)
resnet18.eval()
preprocess = transforms.Compose(
    [
        transforms.Resize(RESNET_RESIZE),
        transforms.CenterCrop(RESNET_CENTERCROP),
        transforms.ToTensor(),
        transforms.Normalize(mean=RESNET_NORMALIZE_MEAN, std=RESNET_NORMALIZE_STD),
    ]
)



def calculate_center(xmin, ymin, xmax, ymax):
    return (xmin + xmax) // 2, (ymin + ymax) // 2

def expand_bounding_box(xmin, ymin, xmax, ymax, expand_ratio=0.2):
    width = xmax - xmin
    height = ymax - ymin
    x_expand = int(width * expand_ratio / 2)
    y_expand = int(height * expand_ratio / 2)
    xmin = max(0, xmin - x_expand)
    ymin = max(0, ymin - y_expand)
    xmax = xmin + width + 2 * x_expand
    ymax = ymin + height + 2 * y_expand
    return xmin, ymin, xmax, ymax

def embed_image_to_extract_features(image):  # embedding image 
    image = Image.fromarray(image)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        embeddings = resnet18(input_batch)
    return embeddings.numpy()

def get_index(detection, check_object):
    cosine_threshold = COSINE_SIMILARITY_THRESHOLD
    matching_threshold = TEMPLATE_MATCHING_THRESHOLD
    cosine_list = []
    distance_list = []
    template_matching_list = []
    img, xmin, ymin, xmax, ymax, class_name = detection
    x_center, y_center = calculate_center(xmin, ymin, xmax, ymax)
    feature_img = embed_image_to_extract_features(img)

    if class_name in OSCILLATE_RANGE_DICT:
        OSCILLATE_RANGE = OSCILLATE_RANGE_DICT[class_name]
    else:
        OSCILLATE_RANGE = OSCILLATE_RANGE_DICT["default"]

    for i in range(len(feature_map)):
        if feature_map[i] is None:
            break

        if i in check_object and check_object[i] is None:
            continue
        if i not in check_object or check_object[i] < MISSING_OBJECT_THRESHOLD:
            cosine = cosine_similarity(feature_img, feature_map[i]) if feature_map[i] is not None else 0
            if cosine > cosine_threshold:
                cosine_list.append((cosine, i))
        if i in check_object and check_object[i] == MISSING_OBJECT_THRESHOLD:
            cosine = cosine_similarity(feature_img, feature_map[i]) if feature_map[i] is not None else 0

            if cosine > COSINE_SIMILARITY_THRESHOLD_CHECKED:
                cosine_list.append((cosine, i))
        x_old, y_old = coordinate_webcam[i]
        if x_old != 0 and y_old != 0:
            distance = np.sqrt((x_center - x_old) ** 2 + (y_center - y_old) ** 2)

            if distance < OSCILLATE_RANGE:
                distance_list.append((distance, i))
        if image_map[i] is None:
            continue
        if i not in check_object or check_object[i] <= MISSING_OBJECT_THRESHOLD:
            
            if img.shape[0] <= image_map[i].shape[0] and img.shape[1] <= image_map[i].shape[1]:
                h, w, _ = img.shape
                template_ = img[int(TEMPLATE_CROP_RATIO[0] * h): int(TEMPLATE_CROP_RATIO[1] * h ), int(TEMPLATE_CROP_RATIO[0] * w): int(TEMPLATE_CROP_RATIO[1] * w ), :]   
     
                min_val = template_matching(image_map[i], template_)

                if min_val < matching_threshold:
                    template_matching_list.append((min_val, i))
            elif img.shape[0] > image_map[i].shape[0]  and img.shape[1] > image_map[i].shape[1] :  
                image = img
                h, w, _ = image_map[i].shape
                template_ = image_map[i][int(TEMPLATE_CROP_RATIO[0] * h): int(TEMPLATE_CROP_RATIO[1] * h ), int(TEMPLATE_CROP_RATIO[0] * w): int(TEMPLATE_CROP_RATIO[1] * w ), :]    

                min_val = template_matching(image, template_)

                if min_val < matching_threshold :
                    template_matching_list.append((min_val, i))
            else:
                try:
                    try:
                        h, w, _ = img.shape
                        template_ = img[int(TEMPLATE_CROP_RATIO[0] * h): int(TEMPLATE_CROP_RATIO[1] * h ), int(TEMPLATE_CROP_RATIO[0] * w): int(TEMPLATE_CROP_RATIO[1] * w ), :]   
                    
                        min_val = template_matching(image_map[i], template_)

                        if min_val < matching_threshold:
                            template_matching_list.append((min_val, i))
                    except:
                        
                        image = img
                        h, w, _ = image_map[i].shape
                        template_ = image_map[i][int(TEMPLATE_CROP_RATIO[0] * h): int(TEMPLATE_CROP_RATIO[1] * h ), int(TEMPLATE_CROP_RATIO[0] * w): int(TEMPLATE_CROP_RATIO[1] * w ), :]    

                        min_val = template_matching(image, template_)
     
                        if min_val < matching_threshold:
                            template_matching_list.append((min_val, i))
                except:
                    pass
                

        
    cosine_list.sort(reverse=True)
    distance_list.sort()
    template_matching_list.sort()


    distance_indices = {i for _, i in distance_list}
    template_matching_indices = {i for _, i in template_matching_list}
    for cosine, i in cosine_list:
        if i in distance_indices:
            return i, False
        
    for i in template_matching_indices:
        if i in distance_indices:
            return i, True
    return len(check_object), False
        
            

def template_matching(image, template, threshold = 0.2):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    res3 = cv2.matchTemplate(image_gray, template_gray, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res3)
    # if k == 404:
    #     cv2.imshow("ccc", template)
    #     print(min_val)
    #     cv2.waitKey(0)
    return min_val


def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)

    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

# Non-Maximum Suppression (NMS) để hợp nhất các bounding box bị overlap
def non_max_suppression(detections, iou_threshold=0.6, overlap_threshold=TEMPLATE_CROP_RATIO[1]):
    if len(detections) == 0:
        return []

    boxes = np.array([[d[6], d[7], d[8], d[9]] for d in detections])
    scores = np.array([1.0] * len(detections))  # Giả sử tất cả các bounding box có điểm số bằng nhau
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0, nms_threshold=iou_threshold)
    indices = indices.flatten().tolist() if len(indices) > 0 else []

    merged_detections = []
    for i in indices:
        keep = True
        for j in indices:
            if i != j and iou(boxes[i], boxes[j]) > overlap_threshold:
                keep = False
                break
        if keep:
            merged_detections.append(detections[i])

    return merged_detections




def track_objects(cap):
    k = 0
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        detections = []

        for result in results:
            for r in result.boxes.data.tolist():
                x1_old, y1_old, x2_old, y2_old, score, class_id = r
                x1_old, y1_old, x2_old, y2_old, class_id = int(x1_old), int(y1_old), int(x2_old), int(y2_old), int(class_id)
                if abs(x2_old - x1_old) < 80 and abs(y2_old - y1_old) < 80: 
                    continue
                img_check = frame[y1_old:y2_old, x1_old:x2_old]
                class_name = model.names[class_id]
                detections.append([img_check, x1_old, y1_old, x2_old, y2_old, class_name])

        # detections = non_max_suppression(detections)  # Áp dụng NMS để hợp nhất các bounding box

        for i in check_object:
            if check_object[i] != 0 and check_object[i] is not None:
                check_object[i] -= 1
        cv2.waitKey(0)
        for detection in detections:
            index, template_checking = get_index(detection, check_object)
            # if k == 586:
                # cv2.imshow("image", image_map[index])
            os.makedirs(f"{OBJECT_IMAGE_PATH}/{index}", exist_ok=True)
            os.makedirs(f"{OBJECT_COORDINATE_PATH}/{index}", exist_ok=True)

            cv2.imwrite(f"{OBJECT_IMAGE_PATH}/{index}/{k:08d}.jpg", detection[0])

            check_object[index] = MISSING_OBJECT_THRESHOLD
                
            coordinate_webcam[index] = calculate_center(detection[1], detection[2], detection[3], detection[4])
            feature_map[index] = embed_image_to_extract_features(detection[0])
            # if image_map[index] is None:
            image_map[index] = detection[0]
            # else:
            #     if (detection[1].shape[0] >= image_map[index].shape[0] and detection[1].shape[1] >= image_map[index].shape[1]):
            #         image_map[index] = detection[1]
                    
                
            with open(f"{OBJECT_COORDINATE_PATH}/{index}/{k:08d}.txt", "w", encoding="utf-8") as file:
                file.write(f"{detection[2]}\t{detection[4]}\t{detection[1]}\t{detection[3]}\t{detection[-1]}\t{frame_no}")
            
        
            k += 1

        for i in list(check_object):
            if check_object[i] == 0:
                check_object[i] = None
        frame_no += 1
    cap.release()
