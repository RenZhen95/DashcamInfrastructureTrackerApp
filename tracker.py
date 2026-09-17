import cv2
import pandas as pd
from ultralytics import YOLO

import streamlit as st

def trackObjects(file_name):
    """
    Track objects in given MP4 video.

    Parameters
    ----------
    file_name : str
    - Path to MP4 file

    placeholder : Streamlit placeholder

    Returns
    -------
    df : pandas.DataFrame
    - Objects tracked
    """
    # Load the pre-trained YOLO model
    model = YOLO('yolov8n.pt')

    # Open the video
    cap = cv2.VideoCapture(file_name)

    # Read from metadata in MP4 file
    vidWidth  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vidHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps       = int(cap.get(cv2.CAP_PROP_FPS)) 

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup VideoWriter to save the output
    fourcc = cv2.VideoWriter_fourcc(*'VP80') # Open-source codec acquired by Google
    out_path = "output_demo.webm"
    out = cv2.VideoWriter(out_path, fourcc, fps, (vidWidth, vidHeight))

    # 2. UI Elements for Loading
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Store bounding boxes coordinates, confidence scores, object classes
    df = pd.DataFrame(
        columns=[
            "Detection", "Frame", "Class", "ID", "Confidence", 
            "xTopLeft", "yTopLeft", "xBottomRight", "yBottomRight"
        ],
        index=["Detection"]
    )
    frame_count = 1
    detection_count = 1

    while cap.isOpened():

        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO detection on the frame with object-tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        # Update Progress Bar
        progress = int((frame_count / total_frames) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count} of {total_frames}...")
        
        # YOLO automatically draws bounding boxes on the frame for you
        annotated_frame = results[0].plot()

        # Convert BGR (OpenCV format) to RGB (Streamlit format)
        # annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        # placeholder.image(annotated_frame, channels="RGB")

        # Save the frame
        out.write(annotated_frame)
    
        # Extract box-related data
        boxes = results[0].boxes
    
        print(f"Number of boxes: {len(boxes)}")
        coordinates = boxes.xyxy.cpu().numpy()
        classes     = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
    
        # For instances when the model is not confident enough to assign an ID
        if not boxes.id is None:
            ids = boxes.id.cpu().numpy()
        else:
            ids = [-1 for i in range(len(boxes))]
    
        for i in range(len(boxes)):
            detection = {
                "Detection": detection_count,
                "Frame": frame_count, "Class": classes[i],
                "ID": ids[i],
                "Confidence": round(confidences[i], 3),
                "xTopLeft": round(coordinates[i][0], 3),
                "yTopLeft": round(coordinates[i][1], 3),
                "xBottomRight": round(coordinates[i][2], 3),
                "yBottomRight": round(coordinates[i][3], 3)
            }
            df.loc[df.shape[0]] = detection
            detection_count += 1
    
        frame_count += 1
    
    cap.release()
    out.release()

    # Clear the loading text
    status_text.empty()
    progress_bar.empty()

    # Set Detection to be index
    df.set_index(["Detection"])

    return df, out_path
