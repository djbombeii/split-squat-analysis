# Inside the video processing loop:
    hip_positions = []
    left_foot_positions = []
    right_foot_positions = []
    shoulder_positions = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

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

    # Convert to numpy arrays
    hip_positions = np.array(hip_positions)
    shoulder_positions = np.array(shoulder_positions)
    left_foot_positions = np.array(left_foot_positions)
    right_foot_positions = np.array(right_foot_positions)

    # Detect jumps using multiple reference points
    def detect_jumps(foot_positions, reference_positions, prominence=0.01, distance=10):
        """
        Detect jumps using both foot movement and reference point validation
        """
        # Find peaks in foot movement
        foot_peaks, foot_properties = find_peaks(-foot_positions, prominence=prominence, distance=distance)
        
        # Find corresponding movements in reference point (hip/shoulder)
        ref_peaks, ref_properties = find_peaks(-reference_positions, prominence=prominence*0.5)
        
        # Validate jumps by checking if there's a corresponding reference movement
        validated_peaks = []
        for foot_peak in foot_peaks:
            # Look for nearby reference peak (within 5 frames)
            nearby_ref_peaks = ref_peaks[np.abs(ref_peaks - foot_peak) <= 5]
            if len(nearby_ref_peaks) > 0:
                validated_peaks.append(foot_peak)
        
        return np.array(validated_peaks), foot_properties

    # Detect jumps for each foot with validation
    left_peaks, left_properties = detect_jumps(left_foot_positions, shoulder_positions)
    right_peaks, right_properties = detect_jumps(right_foot_positions, shoulder_positions)

    # Calculate jump heights using shoulder position as reference
    def calculate_improved_jump_height(foot_positions, shoulder_positions, peaks, frame_height, person_height_inches=68):
        """
        Calculate jump height using shoulder position as a stable reference
        """
        # Get baseline positions
        foot_baseline = np.median(foot_positions[:10])
        shoulder_baseline = np.median(shoulder_positions[:10])
        baseline_difference = foot_baseline - shoulder_baseline
        
        # Calculate height displacement relative to shoulder movement
        peak_foot_positions = foot_positions[peaks]
        peak_shoulder_positions = shoulder_positions[peaks]
        
        # Calculate relative displacement
        relative_displacements = (foot_baseline - peak_foot_positions) - (shoulder_baseline - peak_shoulder_positions)
        
        # Convert to inches using person height as reference
        pixels_per_inch = (frame_height * 0.9) / person_height_inches
        heights_inches = relative_displacements * (1/pixels_per_inch)
        
        return heights_inches

    # Calculate jump heights with improved reference
    left_heights = calculate_improved_jump_height(left_foot_positions, shoulder_positions, left_peaks, frame_height, person_height)
    right_heights = calculate_improved_jump_height(right_foot_positions, shoulder_positions, right_peaks, frame_height, person_height)
