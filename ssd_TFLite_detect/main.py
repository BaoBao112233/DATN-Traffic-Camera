import cv2
import numpy as np
# Import packages
import os
import argparse
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter
from ssdDetect import polygon_calculate
import json
from tqdm import tqdm

OUTPUT_LEDS = [0,0,0]  
CHECK_CAM = False

saved_video_folder = './Video_saved'
saved_video_path = saved_video_folder + '/' + 'output.mp4'

polygons_folder = './polygon_folder'
polygon_file = polygons_folder + '/' + 'polygon.json'

results_folder = './results_folder'
result_file = results_folder + '/' + 'results.json'
prediction_file = results_folder + '/' + 'predictions.json'

models_folder = './All_Model_detect'
model_file = models_folder + '/Sample_TFLite_model' + '/' + 'detect.tflite'
label_file = models_folder + '/Sample_TFLite_model' + '/' + "labelmap.txt"

parser = argparse.ArgumentParser()

parser.add_argument('--video_path', help="Path of video, image or ID webcam (Default: ID webcam)", default=0)
parser.add_argument('--saved_video_path', help="Path save output file", default=saved_video_path)
parser.add_argument('--polygon_path', help="Path of polygon json file", default=polygon_file)
parser.add_argument('--results_path', help="Path of results json file", default=result_file)
args = parser.parse_args()

VIDEO_PATH = args.video_path

def save_to_json(boxes, scores, classes, filename):
    data = []
    for box, score, cls in zip(boxes, scores, classes):
        data.append({
            'box': [float(coord) for coord in box],
            'score': float(score),
            'class': int(cls)
        })
    
    if len(data) != 0:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

def non_max_suppression(boxes, scores, iou_threshold):
    keep_boxes = []
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    while sorted_indices:
        current_index = sorted_indices.pop(0)
        keep_boxes.append(current_index)
        
        remaining_indices = []
        for index in sorted_indices:
            iou = compute_iou(boxes[current_index], boxes[index])
            if iou <= iou_threshold:
                remaining_indices.append(index)
        
        sorted_indices = remaining_indices
    
    return keep_boxes

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def non_max_suppression_per_class(boxes, scores, classes, iou_threshold):
    unique_classes = set(classes)
    final_boxes = []
    final_scores = []
    final_classes = []

    for cls in unique_classes:
        cls_indices = [i for i, c in enumerate(classes) if c == cls]
        cls_boxes = [boxes[i] for i in cls_indices]
        cls_scores = [scores[i] for i in cls_indices]
        
        keep_indices = non_max_suppression(cls_boxes, cls_scores, iou_threshold)
        final_boxes.extend([cls_boxes[i] for i in keep_indices])
        final_scores.extend([cls_scores[i] for i in keep_indices])
        final_classes.extend([cls] * len(keep_indices))

    return final_boxes, final_scores, final_classes

# Create a tracker based on tracker name
trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
def createTrackerByName(trackerType):

    if trackerType == trackerTypes[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.legacy.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)
    return tracker

# Convert time
def seconds_to_hhmmss(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{round(hours)}:{round(minutes)}:{round(seconds)}"

def detect_camera(videostream,imW,imH, fps, fourcc, total_frames, result_queue_cam):
    # ... (your existing code for camera detection)
    # Assuming PointsInfor is the result from camera detection
    global CHECK_CAM

    min_conf_threshold = float(0.55)
    JSON_PATH = polygon_file
    RESULT_JSON_PATH = result_file

    pkg = importlib.util.find_spec('tflite_runtime')

    #Get path to current working directory
    CWD_PATH = os.getcwd()
    # path json polygon
    JSON_PATH = os.path.join(CWD_PATH,JSON_PATH)
    # print("JSON path : ",JSON_PATH)

    PATH_TO_CKPT = os.path.join(CWD_PATH,model_file)
    PATH_TO_LABELS = os.path.join(CWD_PATH,label_file)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    if labels[0] == '???':
        del(labels[0])
    
    # print(labels,'\n')
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Check output layer name to determine if this model was created with TF2 or TF1,
    # because outputs are ordered differently for TF2 and TF1 models
    outname = output_details[0]['name']

    if ('StatefulPartitionedCall' in outname): # This is a TF2 x
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else: # This is a TF1 model
        boxes_idx, classes_idx, scores_idx = 0, 1, 2
    
    # Get Polygon_calculate
    polygon_cal = polygon_calculate(JSON_PATH,imW,imH)

    with open(JSON_PATH) as json_file:
        data = json.load(json_file)

    # Create VideoWriter object
    out = cv2.VideoWriter(saved_video_path, fourcc, fps, (imW, imH))

    # detect frame return boxes
    def detect_ssd(frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()
        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects
        #print("shape Detect : ",boxes.shape,classes.shape,scores.shape,"\n")
        # print("class : ",classes,"\n")
        boxes_new = []
        classes_new = []
        scores_new = []
        centroid_new = []

        boxes_news = []
        classes_news = []
        scores_news = []
        IOUs = []

        class_checks = [0,1,2,3,5,6,10,15,16,17,18,19,20,21,22]
        #class_checks = [0,3,4]
        for i in range(len(scores)):
            if classes[i] in class_checks:
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                    
                    # scale boxes - values (0,1) to size width height
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    # if(polygon_cal.area_box((xmin,ymin,xmax,ymax),limit_area)):
                    boxes_news.append((xmin,ymin,xmax,ymax))
                    classes_news.append(classes[i])
                    scores_news.append(scores[i])
        # print("SHAPE CALASS: ",classes_new,"|||", scores_new)
        for i in range(len(boxes_news)):
            for j in range(len(boxes_news)):
                if i == j:
                    continue
                IOUs.append(compute_iou(boxes_news[i], boxes_news[j]))

        iou_threshold = max(IOUs) - min(IOUs)
        boxes_new, scores_new, classes_new = non_max_suppression_per_class(boxes_news, scores_news, classes_news, iou_threshold)

        for box in boxes_new:
            centroid_new.append([int((box[0]+box[2])//2),int((box[1]+box[3])//2)])

        return boxes_new,classes_new,scores_new,centroid_new

    boxes, classes,scores ,centroids_old = [],[],[],[]

    trackerType = trackerTypes[4]  
    # Create MultiTracker object
    multiTracker = cv2.legacy.MultiTracker_create()

    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        count = 0
        num_frame_to_detect = 8
        frame_id = 0
        while(True):
            # Acquire frame and resize to expected shape [1xHxWx3]
            ret, frame = videostream.read()
            if not ret:
                break

            frame_old, frame = polygon_cal.cut_frame_polygon(frame)

            # get updated location of objects in subsequent frames
            success, boxes_update = multiTracker.update(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if count == num_frame_to_detect:
                controids = polygon_cal.centroid(boxes_update)
                PointsInfor = polygon_cal.check_result(controids,centroids_old)
                key = seconds_to_hhmmss(frame_id+1/fps) # Can update about dayline
                value = PointsInfor
                result_queue_cam[key] = value
                CHECK_CAM = True
                count = 0

            if count == 0:
                boxes, classes, scores, centroids_old = detect_ssd(frame)
                save_to_json(boxes, scores, classes, results_folder+'/'+f'prediction_{frame_id}.json')
                if len(boxes) ==0:
                    count = num_frame_to_detect-1

                multiTracker = cv2.legacy.MultiTracker_create()
                # Initialize MultiTracker
                for bbox in boxes:
                    box_track = (bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1])
                    multiTracker.add(createTrackerByName(trackerType), frame, box_track)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

                    for _class, score in zip(classes, scores):
                        cv2.putText(frame, f'{labels[int(_class)]} {score:.2f}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                for point in centroids_old:
                    cv2.circle(frame, point, 5, (0, 0, 255), -1)

                for point in data['left']:
                    frame = cv2.circle( frame, (point[0], point[1]), 3, (255,0,0), -1)
                
                for point in data['right']:
                    frame = cv2.circle( frame, (point[0], point[1]), 3, (0,255,0), -1)

                frame = cv2.polylines(frame, [np.int32(data['left'])], False, (255,0, 0), thickness=1)
                frame = cv2.polylines(frame, [np.int32(data['right'])], False, (0,255, 0), thickness=1)

                frame = cv2.circle(frame, (data['POINT_RIGHT'][0], data['POINT_RIGHT'][1]), 5, (0, 255, 255), -1)
                frame = cv2.circle(frame, (data['POINT_LEFT'][0], data['POINT_LEFT'][1]), 5, (255, 0, 255), -1)
                # cv2.imshow('Object detector', frame)

            count+=1
            frame_id+=1
            out.write(frame)
            pbar.update(1)

    with open(RESULT_JSON_PATH, 'w') as json_file:
        json.dump(result_queue_cam, json_file, indent=4)
    print(f"Result has been saved to {RESULT_JSON_PATH}")
    videostream.release()
    out.release()
    cv2.destroyAllWindows()


def main_process():
    global CHECK_CAM
    global VIDEO_PATH
    
    result_queue_cam = {}
    imW,imH = 1280,720
    videostream = cv2.VideoCapture(VIDEO_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    total_frames = int(videostream.get(cv2.CAP_PROP_FRAME_COUNT))
    if not videostream.isOpened():
        print("Error")
        exit()
    detect_camera(videostream,imW,imH, 30, fourcc, total_frames, result_queue_cam)

if __name__ == "__main__":
    # Run the main process
    main_process()
