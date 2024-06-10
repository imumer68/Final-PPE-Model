import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("WebRTC Video Stream")

def video_frame_callback(frame):
    # This function is called for each video frame
    img = frame.to_ndarray(format="bgr24")
    return img

webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)

st.write("This is a simple Streamlit app to stream video using WebRTC.")
