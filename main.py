import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.title("WebRTC Test")

def app():
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False}
    )

    if webrtc_ctx.video_receiver:
        st.write("WebRTC connection established.")
    else:
        st.write("Waiting for WebRTC connection...")

if __name__ == "__main__":
    app()
