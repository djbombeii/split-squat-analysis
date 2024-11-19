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

# App Title
st.title("Split Squat Analysis with Visual Overlays and Jump Height")
st.write("Upload a video to analyze alternating split squat jumps.")

# Video Upload Section
uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov"])

# Optional height input
person_height = st.number_input("Enter your height in inches (default: 68 inches = 5'8\")", 
                              min_value=48, 
                              max_value=84, 
                              value=68)

if uploaded_file:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    st.success("Video uploaded successfully!")

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Prepare to save frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate video duration in minutes
    video_duration_minutes = total_frames / (fps * 60)

    # Temporary directory for frames
    frames_dir = tempfile.mkdtemp()

    # Variables for jump classification and counting
    left_foot_positions = []
    right_foot_positions = []
    frame_count = 0

    # Process video frame by frame and collect feet positions
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        # Draw pose landmarks and connections
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Track Y-coordinate of both feet
            left_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            right_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            
            left_foot_positions.append(left_foot.y)
            right_foot_positions.append(right_foot.y)

        # Save the frame
        frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()

    # Convert positions to numpy arrays
    left_foot_positions = np.array(left_foot_positions)
    right_foot_positions = np.array(right_foot_positions)
    
    # Calculate foot height difference
    foot_difference = left_foot_positions - right_foot_positions
    
    # Find peaks for alternating pattern
    # Positive peaks (left foot higher)
    left_peaks, left_properties = find_peaks(foot_difference, prominence=0.1)
    # Negative peaks (right foot higher)
    right_peaks, right_properties = find_peaks(-foot_difference, prominence=0.1)
    
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

    # Count alternating patterns
    total_alternations = min(len(left_peaks), len(right_peaks))
    reps_per_minute = total_alternations / video_duration_minutes

    # Display counts in Streamlit
    st.write(f"### Analysis Results:")
    st.write(f"Total Split Squat Jumps: {total_alternations}")
    st.write(f"Video Duration: {video_duration_minutes:.2f} minutes")
    st.write(f"Reps per Minute: {reps_per_minute:.1f}")

    # Create frame-by-frame counters
    jumps_by_frame = {frame: {'count': 0} for frame in range(frame_count)}

    # Update running counter
    current_count = 0
    all_peaks = sorted(list(left_peaks) + list(right_peaks))
    
    for peak_idx in all_peaks:
        current_count = (current_count + 1) // 2  # Increment counter for each complete alternation
        for frame in range(peak_idx, frame_count):
            jumps_by_frame[frame] = {'count': current_count}

    # Process saved frames again to add text overlays
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

    # Use FFmpeg to compile frames into video
    output_video_path = "output_with_overlays.mp4"
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        output_video_path
    ]
    subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Display the processed video
    st.write("### Processed Video with Visual Overlays")
    st.video(output_video_path)

    # Cleanup
    os.remove(video_path)
    for frame_file in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, frame_file))
    os.rmdir(frames_dir)

else:
    st.warning("Please upload a video.")
