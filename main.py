import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
import av

st.title("WebRTC Video Stream")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Add any image processing here
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def app():
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.video_receiver:
        st.write("WebRTC connection established.")
    else:
        st.write("Waiting for WebRTC connection...")

if __name__ == "__main__":
    app()

st.write("This is a simple Streamlit app to stream video using WebRTC.")
