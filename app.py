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

def detect_jumps(foot_positions, reference_positions, prominence=0.01, distance=10):
    """
    Detect jumps using both foot movement and reference point validation
    """
    # Find peaks in foot movement
    foot_peaks, foot_properties = find_peaks(-foot_positions, prominence=prominence, distance=distance)
    
    # Find corresponding movements in reference point
    ref_peaks, ref_properties = find_peaks(-reference_positions, prominence=prominence*0.5)
    
    # Validate jumps by checking if there's a corresponding reference movement
    validated_peaks = []
    for foot_peak in foot_peaks:
        # Look for nearby reference peak (within 5 frames)
        nearby_ref_peaks = ref_peaks[np.abs(ref_peaks - foot_peak) <= 5]
        if len(nearby_ref_peaks) > 0:
            validated_peaks.append(foot_peak)
    
    return np.array(validated_peaks), foot_properties

def calculate_flight_time(positions, peaks, fps):
    """
    Calculate flight time for each jump by finding takeoff and landing points
    """
    flight_times = []
    takeoff_indices = []
    landing_indices = []
    
    for peak in peaks:
        # Look before peak for takeoff (when foot leaves ground)
        takeoff_idx = peak
        for i in range(peak, max(peak-20, 0), -1):  # Look up to 20 frames before peak
            if positions[i] - positions[i-1] < -0.01:  # Threshold for takeoff detection
                takeoff_idx = i
                break
                
        # Look after peak for landing
        landing_idx = peak
        for i in range(peak, min(peak+20, len(positions)-1)):  # Look up to 20 frames after peak
            if positions[i] - positions[i-1] > 0.01:  # Threshold for landing detection
                landing_idx = i
                break
        
        # Calculate flight time in seconds
        flight_time = (landing_idx - takeoff_idx) / fps
        
        flight_times.append(flight_time)
        takeoff_indices.append(takeoff_idx)
        landing_indices.append(landing_idx)
    
    return np.array(flight_times), np.array(takeoff_indices), np.array(landing_indices)

# App Title
st.title("Split Squat Analysis with Visual Overlays and Flight Time")
st.write("Upload a video to analyze alternating split squat jumps.")

# Video Upload Section
uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov"])

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

    # Variables for tracking
    left_foot_positions = []
    right_foot_positions = []
    shoulder_positions = []
    hip_positions = []
    frame_count = 0

    # Process video frame by frame and collect positions
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

            # Track multiple reference points
            left_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            right_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Calculate mid-points for more stable reference
            mid_hip_y = (left_hip.y + right_hip.y) / 2
            mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # Store positions
            hip_positions.append(mid_hip_y)
            shoulder_positions.append(mid_shoulder_y)
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
    shoulder_positions = np.array(shoulder_positions)
    hip_positions = np.array(hip_positions)
    
    # Detect jumps for each foot with validation
    left_peaks, left_properties = detect_jumps(left_foot_positions, shoulder_positions, prominence=0.01, distance=15)
    right_peaks, right_properties = detect_jumps(right_foot_positions, shoulder_positions, prominence=0.01, distance=15)

    # Calculate flight times
    left_flight_times, left_takeoffs, left_landings = calculate_flight_time(left_foot_positions, left_peaks, fps)
    right_flight_times, right_takeoffs, right_landings = calculate_flight_time(right_foot_positions, right_peaks, fps)
    
    # Create and display the split squat detection graph
    st.write("### Movement Analysis Graphs")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot foot and reference positions
    ax1.plot(-left_foot_positions, label='Left Foot', color='blue', alpha=0.7)
    ax1.plot(-right_foot_positions, label='Right Foot', color='red', alpha=0.7)
    ax1.plot(-shoulder_positions, label='Shoulder Reference', color='green', alpha=0.5)
    
    # Plot jump phases
    for takeoff, peak, landing in zip(left_takeoffs, left_peaks, left_landings):
        ax1.axvspan(takeoff, landing, alpha=0.2, color='blue')
    for takeoff, peak, landing in zip(right_takeoffs, right_peaks, right_landings):
        ax1.axvspan(takeoff, landing, alpha=0.2, color='red')
    
    ax1.plot(left_peaks, -left_foot_positions[left_peaks], "bx", label="Left Foot Peaks")
    ax1.plot(right_peaks, -right_foot_positions[right_peaks], "rx", label="Right Foot Peaks")
    ax1.set_title('Vertical Movement Tracking\n(Shaded areas show flight phases)')
    ax1.set_ylabel('Height (inverted)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot foot difference for alternation detection
    foot_difference = left_foot_positions - right_foot_positions
    ax2.plot(foot_difference, label='Foot Height Difference', color='purple', alpha=0.7)
    ax2.plot(left_peaks, foot_difference[left_peaks], "bx", label="Left Foot Up")
    ax2.plot(right_peaks, foot_difference[right_peaks], "rx", label="Right Foot Up")
    ax2.set_title('Alternating Pattern Detection')
    ax2.set_ylabel('Height Difference')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot flight times
    if len(left_flight_times) > 0:
        ax3.plot(left_peaks, left_flight_times, 'bo-', label='Left Flight Time', alpha=0.7)
    if len(right_flight_times) > 0:
        ax3.plot(right_peaks, right_flight_times, 'ro-', label='Right Flight Time', alpha=0.7)
    ax3.set_title('Jump Flight Times')
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Flight Time (seconds)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

    # Calculate and display statistics
    total_alternations = min(len(left_peaks), len(right_peaks))
    reps_per_minute = total_alternations / video_duration_minutes

    # Display counts and statistics
    st.write("### Analysis Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Jump Counts:")
        st.write(f"Total Split Squats: {total_alternations}")
        st.write(f"Left Jumps: {len(left_peaks)}")
        st.write(f"Right Jumps: {len(right_peaks)}")
        st.write(f"Reps per Minute: {reps_per_minute:.1f}")
    
    with col2:
        st.write("Flight Times:")
        if len(left_flight_times) > 0:
            st.write(f"Left Max Flight Time: {np.max(left_flight_times):.3f} seconds")
            st.write(f"Left Avg Flight Time: {np.mean(left_flight_times):.3f} seconds")
        if len(right_flight_times) > 0:
            st.write(f"Right Max Flight Time: {np.max(right_flight_times):.3f} seconds")
            st.write(f"Right Avg Flight Time: {np.mean(right_flight_times):.3f} seconds")

    # Create frame-by-frame counters
    jumps_by_frame = {frame: {'count': 0, 'left_flight': 0, 'right_flight': 0} 
                     for frame in range(frame_count)}

    # Update counters and flight times for each frame
    def get_latest_flight_time(frame_idx, peak_indices, flight_times):
        valid_peaks = peak_indices[peak_indices <= frame_idx]
        if len(valid_peaks) > 0:
            return flight_times[len(valid_peaks) - 1]
        return 0

    for i in range(frame_count):
        left_count = len(left_peaks[left_peaks <= i])
        right_count = len(right_peaks[right_peaks <= i])
        total_count = min(left_count, right_count)
        
        jumps_by_frame[i] = {
            'count': total_count,
            'left_flight': get_latest_flight_time(i, left_peaks, left_flight_times),
            'right_flight': get_latest_flight_time(i, right_peaks, right_flight_times)
        }

    # Process frames with overlays
    for i in range(frame_count):
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        frame = cv2.imread(frame_path)
        
        if frame is not None:
            current_data = jumps_by_frame[i]
            
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

            # Add text overlays
            put_text_with_background(frame, f'Split Squats: {current_data["count"]}', (50, 50))
            current_rpm = (current_data["count"] / (i/fps/60)) if i > 0 else 0
            put_text_with_background(frame, f'Reps/min: {current_rpm:.1f}', (50, 100))
            put_text_with_background(frame, f'L Flight: {current_data["left_flight"]:.3f}s', (50, 150))
            put_text_with_background(frame, f'R Flight: {current_data["right_flight"]:.3f}s', (50, 200))

            cv2.imwrite(frame_path, frame)

    # Compile video with FFmpeg
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
    
