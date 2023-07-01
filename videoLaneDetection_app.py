import streamlit as st
import cv2

video_file = st.file_uploader('video', type = ['mp4'])
cap = cv2.VideoCapture(video_file)
