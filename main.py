import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load the YOLO model
model = YOLO('best.pt')

class ObjectDetector(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        results = self.model.predict(img_bgr)

        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result.tolist()
            cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img_bgr, f'{self.model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

# Streamlit App
st.title('Real-time YOLO Object Detection')

option = st.radio('Choose Input Source:', ('Upload File', 'Webcam'))

if option == 'Upload File':
    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
    if uploaded_file is not None:
        if uploaded_file.type in ["image/jpeg", "image/png"]:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            processed_frame = ObjectDetector().transform(av.VideoFrame.from_ndarray(frame, format="bgr24"))
            st.image(processed_frame, caption='Processed Image', use_column_width=True)
        elif uploaded_file.type in ["video/mp4", "video/avi", "video/mov"]:
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
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = ObjectDetector().transform(av.VideoFrame.from_ndarray(frame_rgb, format="bgr24"))
                out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                stframe.image(processed_frame, channels="RGB")

            cap.release()
            out.release()

            with open(output_path, 'rb') as f:
                st.download_button('Download Processed Video', f, file_name='processed_video.mp4')

elif option == 'Webcam':
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=ObjectDetector,
        media_stream_constraints={"video": True, "audio": False},
    )

    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.model = model

    # Placeholder for the download button
    st_placeholder = st.empty()

    if webrtc_ctx.state.playing:
        st_placeholder.info("Processing webcam feed...")
    else:
        st_placeholder.empty()
