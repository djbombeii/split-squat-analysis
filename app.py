import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import os
import numpy as np
from scipy.signal import find_peaks
import subprocess
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# App Title
st.title("Split Squat Analysis with Visual Overlays and Jump Height")
st.write("Upload a video to analyze alternating split squat jumps.")

# Function to calculate jump height in inches
def calculate_jump_height(foot_positions, frame_height, peaks, person_height_inches=68):
    """
    Calculate jump height based on pixel displacement and estimated scale
    Args:
    foot_positions: Array of foot Y positions
    frame_height: Height of video frame in pixels
    peaks: Array of peak indices
    person_height_inches: Person's height in inches (default 5'8")
    """
    # Get baseline (standing) position by taking median of first few frames
    baseline = np.median(foot_positions[:10])
    
    # Calculate heights at peaks
    peak_heights = foot_positions[peaks]
    
    # Calculate displacement in pixels
    displacements = baseline - peak_heights
    
    # Estimate pixels per inch using person's height as reference
    # Assuming person takes up about 90% of frame height when standing
    pixels_per_inch = (frame_height * 0.9) / person_height_inches
    
    # Convert displacements to inches
    heights_inches = displacements * (1/pixels_per_inch)
    
    return heights_inches

# Video Upload Section
uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov"])

# Optional height input
person_height = st.number_input("Enter your height in inches (default: 68 inches = 5'8\")", 
                              min_value=48, 
                              max_value=84, 
                              value=68)

if uploaded_file:
    # [Previous code remains the same until after peak detection]
    
    # Calculate jump heights for both feet
    left_heights = calculate_jump_height(left_foot_positions, frame_height, left_peaks, person_height)
    right_heights = calculate_jump_height(right_foot_positions, frame_height, right_peaks, person_height)
    
    # Create and display the split squat detection graph
    st.write("### Split Squat Detection Graph")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot individual foot positions
    ax1.plot(-left_foot_positions, label='Left Foot', color='blue', alpha=0.7)
    ax1.plot(-right_foot_positions, label='Right Foot', color='red', alpha=0.7)
    ax1.set_title('Individual Foot Heights')
    ax1.set_ylabel('Height (inverted)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot foot difference and detected alternations
    ax2.plot(foot_difference, label='Foot Height Difference', color='purple', alpha=0.7)
    ax2.plot(left_peaks, foot_difference[left_peaks], "go", label="Left Foot Up")
    ax2.plot(right_peaks, foot_difference[right_peaks], "ro", label="Right Foot Up")
    ax2.set_title('Foot Height Difference\n(Positive = Left Foot Higher)')
    ax2.set_ylabel('Height Difference')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot jump heights
    ax3.plot(left_peaks, left_heights, 'bo-', label='Left Foot Height', alpha=0.7)
    ax3.plot(right_peaks, right_heights, 'ro-', label='Right Foot Height', alpha=0.7)
    ax3.set_title('Jump Heights')
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Height (inches)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

    # Calculate average jump heights
    avg_left_height = np.mean(left_heights)
    avg_right_height = np.mean(right_heights)
    max_left_height = np.max(left_heights)
    max_right_height = np.max(right_heights)

    # Display jump height statistics
    st.write("### Jump Height Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Left Foot:")
        st.write(f"Average Height: {avg_left_height:.1f} inches")
        st.write(f"Max Height: {max_left_height:.1f} inches")
    with col2:
        st.write("Right Foot:")
        st.write(f"Average Height: {avg_right_height:.1f} inches")
        st.write(f"Max Height: {max_right_height:.1f} inches")

    # Update the frame overlay to include jump heights
    for i in range(frame_count):
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        frame = cv2.imread(frame_path)
        
        if frame is not None:
            current_count = jumps_by_frame[i]['count']
            
            def put_text_with_background(img, text, position):
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3
                color = (0, 255, 0)
                
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.rectangle(img, 
                            (position[0] - 10, position[1] - text_height - 10),
                            (position[0] + text_width + 10, position[1] + 10),
                            (0, 0, 0),
                            -1)
                cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

            # Add text with backgrounds showing running counts and heights
            put_text_with_background(frame, f'Split Squats: {current_count}', (50, 50))
            current_rpm = (current_count / (i/fps/60)) if i > 0 else 0
            put_text_with_background(frame, f'Reps/min: {current_rpm:.1f}', (50, 100))
            
            # Find the most recent jump height for this frame
            left_recent = left_heights[left_peaks <= i][-1] if len(left_peaks) > 0 and i >= left_peaks[0] else 0
            right_recent = right_heights[right_peaks <= i][-1] if len(right_peaks) > 0 and i >= right_peaks[0] else 0
            put_text_with_background(frame, f'L Height: {left_recent:.1f}"', (50, 150))
            put_text_with_background(frame, f'R Height: {right_recent:.1f}"', (50, 200))

            cv2.imwrite(frame_path, frame)

    # [Rest of the code remains the same]
