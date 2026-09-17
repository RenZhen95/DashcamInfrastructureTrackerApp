import os
import tempfile
import streamlit as st

from tracker import trackObjects

st.title("Dashcam Infrastructure Tracker")
st.write("Upload a dashcam video or use the built-in demo to automatically detect and track vehicles and infrastructure.")

_filepath = None

# App UI Setup: Create two columns for the upload vs. demo option
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Video (MP4)", type=['mp4'])

with col2:
    st.write("No video on hand?")
    use_demo = st.button("▶️ Run Demo Video") # returns Boolean

if use_demo:
    _filepath = "input.mp4"

# Consider working with dataset from https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022
# for road damage

if uploaded_file is not None:
    # Save the uploaded file to a temporary location so OpenCV can read it
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    _filepath = tfile.name

if _filepath is not None:
    st.info("Processing Video... Please wait.")

    df, out_path = trackObjects(_filepath)
    st.success("Processing Complete!")

    # Display the final interactive video
    st.video(out_path)
    
    # Show a preview of the data in the app
    st.dataframe(df.head())
        
    # Create a download button for the CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Tracking Data (CSV)",
        data=csv, file_name=f'{_filepath}Tracking.csv', mime='text/csv',
    )
