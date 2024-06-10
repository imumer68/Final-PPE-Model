import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

st.title("WebRTC Video Stream")

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {
            "urls": ["stun:stun.l.google.com:19302"],
        },
        {
            "urls": ["turn:numb.viagenie.ca"],
            "username": "your_username",  # Replace with your TURN server username
            "credential": "your_password"  # Replace with your TURN server credential
        },
    ]
})

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Add any image processing here
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if webrtc_ctx.state.playing:
    st.write("WebRTC connection established.")
else:
    st.write("Waiting for WebRTC connection...")

st.write("This is a simple Streamlit app to stream video using WebRTC.")
