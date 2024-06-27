# import libraries
import argparse
import cv2
import numpy as np
import json

polygons_folder = './polygon_folder'
polygon_file = polygons_folder + '/' + 'polygon.json'

# Define and parse input arguments
parser = argparse.ArgumentParser()

parser.add_argument('--video_path', help="Path of video or image or ID webcam (default: ID webcam)", default=0)
parser.add_argument('--json_path', help="Path of JSON file saved", default=polygon_file)

args = parser.parse_args()

# Open file
video_path = args.video_path

video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print("Error: Could not open video.")
    exit()
video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_width =int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

if video_height >= 720:
    video_height = int(video_height * (720/video_height))

if video_width >= 1280:
    video_width = int(video_width * (1280/video_width))

# click left mouse to add point for polygon of dict points left and right 
def handle_point_click(event, x, y, flags, points):
    if event == cv2.EVENT_LBUTTONDOWN:
        points['left'].append([x, y])
    if event == cv2.EVENT_RBUTTONDOWN:
        points['POINT_LEFT'] = [x, y]

# draw polygon left and right
def draw_polygon (frame, points):
    for point in points['left']:
        frame = cv2.circle( frame, (point[0], point[1]), 3, (255,0,0), -1)
    
    for point in points['right']:
        frame = cv2.circle( frame, (point[0], point[1]), 3, (0,255,0), -1)

    frame = cv2.polylines(frame, [np.int32(points['left'])], False, (255,0, 0), thickness=1)
    frame = cv2.polylines(frame, [np.int32(points['right'])], False, (0,255, 0), thickness=1)
    return frame


POINTS = {}
POINTS["area"]= []
POINTS['left'] =  []
POINTS['right'] =  []
POINTS['POINT_RIGHT']= [0,0]
POINTS['POINT_LEFT']= [0,0]
point_medial = []

print("draw right road \n")

paused = False

while video.isOpened():
    if not paused:
        ret, frame = video.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (video_width, video_height))

        # draw polygon
        frame = draw_polygon(frame_resized, POINTS)
        frame = cv2.circle(frame, (POINTS['POINT_RIGHT'][0], POINTS['POINT_RIGHT'][1]), 5, (0, 255, 255), -1)
        frame = cv2.circle(frame, (POINTS['POINT_LEFT'][0], POINTS['POINT_LEFT'][1]), 5, (255, 0, 255), -1)

        cv2.imshow("Draw Polygon", frame_resized)
        
        cv2.setMouseCallback('Draw Polygon', handle_point_click, POINTS)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'): # Press 'a' to draw biggest area
        print("Draw area")
        POINTS['area'] = POINTS['left']
        POINTS['left'] = []
    
    elif key == ord('q'): # Press 'q' to quit 
        break
    
    elif key == ord('p'): # Press 'p' to draw right point
        POINTS['POINT_RIGHT'] = POINTS['POINT_LEFT']
    
    elif key == ord('t'): # Press 't' to draw right area
        print("draw left road \n")
        POINTS['right'] = POINTS['left'] 
        POINTS['left'] = []
    
    elif key == ord('s'):  # Press 's' to pause
        paused = True

    elif key == ord('r'):  # Press 'r' to resume
        paused = False

# Clean up
cv2.destroyAllWindows()
# video.stop()

print('POINT LEFT : ',POINTS['left'])
print('\n')
print('POINT RIGHT : ',POINTS['right'])

POINTS['size_width'] = video_width
POINTS['size_height'] = video_height
print('\n')
print("SIZE SCREEN : ",POINTS['size_width'] ,POINTS['size_height'] )

# # Specify the file path where you want to save the JSON data
JSON_PATH = args.json_path

# Write the dictionary to a JSON file
with open(JSON_PATH, 'w') as json_file:
    json.dump(POINTS, json_file)
