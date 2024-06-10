import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile

# Load your model
model = YOLO('best.pt')

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]
})


# VideoTransformer for applying YOLO detection
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = YOLO('best.pt')

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Process frame using YOLO model
        results = self.model.predict(source=image, save=False, show=False)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                cls = int(box.cls.item())
                conf = box.conf.item()
                label = f'{self.model.names[cls]}: {conf:.2f}'  # Use model.names for class labels
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return image


# Streamlit App
st.title('Real-time YOLO Object Detection')

# Option to choose between uploaded file and webcam
option = st.radio('Choose Input Source:', ('Upload File', 'Webcam'))

if option == 'Upload File':
    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
    if uploaded_file is not None:
        file_type = uploaded_file.type
        if file_type in ["image/jpg", "image/jpeg", "image/png"]:
            # If the uploaded file is an image
            image = Image.open(uploaded_file)
            frame = np.array(image)
            results = model.predict(source=frame, save=False, show=False)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                    cls = int(box.cls.item())
                    conf = box.conf.item()
                    label = f'{model.names[cls]}: {conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            st.image(frame, caption='Processed Image', use_column_width=True)

        elif file_type in ["video/mp4", "video/avi", "video/mov"]:
            # If the uploaded file is a video
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            output_path = 'output.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model.predict(source=frame_rgb, save=False, show=False)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        coords = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                        cls = int(box.cls.item())
                        conf = box.conf.item()
                        label = f'{model.names[cls]}: {conf:.2f}'
                        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                stframe.image(frame_rgb, channels="RGB")

            cap.release()
            out.release()

            # Provide download link for processed video
            with open(output_path, 'rb') as f:
                st.download_button('Download Processed Video', f, file_name='processed_video.mp4')

elif option == 'Webcam':
    webrtc_streamer(key="sample",
                    mode=WebRtcMode.SENDRECV,
                    video_transformer_factory=VideoTransformer,
                    rtc_configuration=RTC_CONFIGURATION)
