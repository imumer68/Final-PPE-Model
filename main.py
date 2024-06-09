import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, VideoFrame, WebRtcMode

# Load your model
model = YOLO('best.pt')

# Function to process the frame and perform predictions
def process_frame(frame):
    results = model.predict(source=frame, save=False, show=False)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            cls = int(box.cls.item())
            conf = box.conf.item()
            label = f'{model.names[cls]}: {conf:.2f}'  # Use model.names for class labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

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
            processed_frame = process_frame(frame)
            st.image(processed_frame, caption='Processed Image', use_column_width=True)

        elif file_type in ["video/mp4", "video/avi", "video/mov"]:
            # If the uploaded file is a video
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = process_frame(frame_rgb)
                out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                stframe.image(processed_frame, channels="RGB")

            cap.release()
            out.release()

            # Provide download link for processed video
            with open(output_path, 'rb') as f:
                st.download_button('Download Processed Video', f, file_name='processed_video.mp4')

elif option == 'Webcam':
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.frame = None

        def recv(self, frame: VideoFrame) -> VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = process_frame(img)
            return VideoFrame.from_ndarray(img, format="rgb24")

    ctx = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False})

    if ctx.video_processor:
        st.write("Press the 'Download Recorded Webcam Video' button after stopping the video stream.")

    # Provide download link for webcam video after processing is done
    if st.button('Download Recorded Webcam Video'):
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 480))

        while True:
            if ctx.video_processor and ctx.video_processor.frame is not None:
                frame = ctx.video_processor.frame
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                break

        out.release()
        with open(output_path, 'rb') as f:
            st.download_button('Download Recorded Webcam Video', f, file_name='webcam_output.mp4')
