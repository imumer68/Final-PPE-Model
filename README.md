**YOLOv8 Object Detection with Streamlit**
****
This project demonstrates how to perform object detection using YOLOv8 and visualize the results using Streamlit. The model is fine-tuned to detect Personal Protective Equipment (PPE) in images and videos.

_**Requirements**_

Python 
Streamlit
OpenCV
Ultralytics YOLO
Install the required packages using:

pip install streamlit opencv-python ultralytics

_**Usage**_

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Run the Streamlit app using the following command:

streamlit run app.py

4. Upload an image or video file containing scenes with PPE.
5. The model will detect PPE items and draw bounding boxes around them.
6. View the processed image or video with the detected objects highlighted.

_**Model**_

The YOLOv8 model (best.pt) is fine-tuned to detect the following classes:

Hardhat
Mask
NO-Hardhat  
NO-Mask
NO-Safety Vest
Person
Safety Cone
Safety Vest
machinery
vehicle
