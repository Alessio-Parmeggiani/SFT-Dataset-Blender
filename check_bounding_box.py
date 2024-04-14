"""cripts that given an image path, show the image with the drawing of the bounding box contained in a file path.
Bounding box are in YLO format: ID,x_center,y_center,width,height
e.g. 
6 0.6021241830065359 0.445556640625 0.00980392156862745 0.00830078125
1 0.7757352941176471 0.951904296875 0.008986928104575163 0.00732421875
4 0.6501225490196079 0.95654296875 0.015931372549019607 0.013671875
"""

import cv2
import numpy as np

def show_bounding_box(image_path, bounding_box_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Read the bounding box
    with open(bounding_box_path, 'r') as f:
        bounding_box = f.readlines()
    
    # Draw the bounding box
    for box in bounding_box:
        box = box.split()
        print("drawing box: ", box)
        x_center = float(box[1])
        y_center = float(box[2])
        width = float(box[3])
        height = float(box[4])
        
        x1 = int((x_center - width/2) * image.shape[1])
        y1 = int((y_center - height/2) * image.shape[0])
        x2 = int((x_center + width/2) * image.shape[1])
        y2 = int((y_center + height/2) * image.shape[0])
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    

    #resize , half size
    image = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
    # Show the image
    cv2.imshow('Bounding Box', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test the function
image_path = r"C:\Users\alessio\Desktop\DatasetFlight\datasets\train\img_0.jpg"
bounding_box_path = image_path.replace('jpg', 'txt')
show_bounding_box(image_path, bounding_box_path)
