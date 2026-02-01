Python 3.14.2 (tags/v3.14.2:df79316, Dec  5 2025, 17:18:21) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> import cv2
... import mediapipe as mp
... import numpy as np
... import csv
... import time
... from datetime import datetime
... import matplotlib.pyplot as plt
... import pandas as pd
... import os
... import threading
... 
... # Initialize pose
... mp_drawing = mp.solutions.drawing_utils
... mp_pose = mp.solutions.pose
... 
... # Initialize pose detector
... pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
... 
... # Voice feedback (disabled if pyttsx3/pywin32 missing)
... voice_enabled = True
... try:
...     import pyttsx3
...     engine = pyttsx3.init()
...     engine.setProperty('rate', 150)
...     engine.setProperty('volume', 0.9)
...     
...     # Get available voices and set to female if available
...     voices = engine.getProperty('voices')
...     if len(voices) > 1:
...         engine.setProperty('voice', voices[1].id)  # Usually female voice
...     
...     def speak(text):
...         if voice_enabled:
...             try:
...                 engine.say(text)
...                 engine.runAndWait()
...             except:
...                 pass
...                 
...     def speak_async(text):
...         if voice_enabled:
...             def speak_thread():
...                 try:
...                     engine.say(text)
...                     engine.runAndWait()
...                 except:
                    pass
            threading.Thread(target=speak_thread, daemon=True).start()
            
except ImportError:
    voice_enabled = False
    print("[⚠] pyttsx3 or pywin32 not found. Voice feedback disabled.")
    def speak(text):
        pass
    def speak_async(text):
        pass

# Exercise-specific motivational messages
exercise_messages = {
    "push-up": {
        1: "Perfect push-up! You're getting stronger!",
        5: "Five push-ups! Your chest is on fire!",
        10: "Ten push-ups! You're a warrior!",
        15: "Fifteen! Your upper body is incredible!",
        20: "Twenty push-ups! Absolutely legendary!"
    },
    "squat": {
        1: "Great squat! Feel those glutes working!",
        5: "Five squats! Your legs are getting powerful!",
        10: "Ten squats! Those quads are burning!",
        15: "Fifteen squats! You're building thunder thighs!",
        20: "Twenty squats! Your lower body is unstoppable!"
    },
    "bicep-curl": {
        1: "Nice curl! Feel those biceps growing!",
        5: "Five curls! Your arms are getting sculpted!",
        10: "Ten curls! Those biceps are pumping!",
        15: "Fifteen curls! Your arms look amazing!",
        20: "Twenty curls! Biceps of steel!"
    },
    "jumping-jack": {
        1: "Great jump! Get that heart pumping!",
        5: "Five jacks! Cardio champion!",
        10: "Ten jacks! You're on fire!",
        15: "Fifteen jacks! Incredible energy!",
        20: "Twenty jacks! Cardio superstar!"
    },
    "shoulder-press": {
        1: "Perfect press! Strong shoulders!",
        5: "Five presses! Your shoulders are powerful!",
        10: "Ten presses! Amazing upper body strength!",
        15: "Fifteen presses! Shoulder muscles of steel!",
        20: "Twenty presses! Incredible definition!"
    }
}

# Exercise list and current selection
exercises = ["push-up", "squat", "bicep-curl", "jumping-jack", "shoulder-press"]
current_exercise_index = 0
exercise = exercises[current_exercise_index]

def get_motivational_message(count, exercise_type):
    if count in exercise_messages[exercise_type]:
        return exercise_messages[exercise_type][count]
    elif count % 5 == 0 and count > 20:
        return f"AMAZING! {count} {exercise_type.replace('-', ' ')}s! You're unstoppable!"
    return None

# Angle calculation
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    angle = np.degrees(np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0]))
    angle = np.abs(angle)
    return 360 - angle if angle > 180 else angle

# Distance calculation for jumping jacks
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Save to CSV
def save_to_csv(data):
    filename = f"workout_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Exercise", "Left Angle", "Right Angle", "Reps", "Duration (s)", "Speed (rep/s)"])
        writer.writerows(data)
    print(f"[✔] Data saved to {filename}")

# Export to Excel
def save_to_excel(data):
    try:
        filename = f"workout_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df = pd.DataFrame(data, columns=["Timestamp", "Exercise", "Left Angle", "Right Angle", "Reps", "Duration (s)", "Speed (rep/s)"])
        df.to_excel(filename, index=False)
        print(f"[✔] Data saved to {filename}")
    except ImportError:
        print("[⚠] pandas/openpyxl not found. Excel export skipped.")

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[❌] Camera error: Unable to open camera at index 0")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("[✔] Camera opened successfully")

# Video recording variables
is_recording = False
video_writer = None
recording_start_time = None
recorded_filename = ""

def start_recording():
    global is_recording, video_writer, recording_start_time, recorded_filename
    if not is_recording:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        recorded_filename = f"workout_{exercise}_{timestamp}.mp4"
        
        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(recorded_filename, fourcc, 20.0, (1280, 720))
        
        is_recording = True
        recording_start_time = time.time()
        speak_async("Recording started! Show off your amazing workout!")
        print(f"[🎥] Recording started: {recorded_filename}")
        return True
    return False

def stop_recording():
    global is_recording, video_writer, recording_start_time, recorded_filename
    if is_recording and video_writer:
        is_recording = False
        video_writer.release()
        video_writer = None
        recording_duration = time.time() - recording_start_time
        speak_async(f"Recording saved! {recording_duration:.1f} seconds of awesome workout captured!")
        print(f"[✔] Recording stopped and saved: {recorded_filename}")
        print(f"[🎬] Duration: {recording_duration:.1f} seconds")
        return True
    return False

# Matplotlib setup
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
x_data, y_data = [], []
line, = ax.plot(x_data, y_data, 'b-', linewidth=2)
ax.set_xlim(0, 30)
ax.set_ylim(0, 180)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle/Distance')
ax.set_title('Live Exercise Metrics vs Time')
ax.grid(True, alpha=0.3)

# Variables
counter = 0
stage = None
data_log = []
start_time = time.time()
prev_time = start_time
rep_times = []
last_count_time = 0
last_encouragement_time = 0
last_guidance_time = 0
min_time_between_reps = 0.5

# Exercise-specific thresholds
exercise_thresholds = {
    "push-up": {"up": 120, "down": 110},
    "squat": {"up": 160, "down": 90},
    "bicep-curl": {"up": 140, "down": 40},
    "jumping-jack": {"apart": 0.3, "together": 0.15},
    "shoulder-press": {"up": 160, "down": 90}
}

def detect_push_up(landmarks, avg_angle):
    """Push-up detection with relaxed position checking"""
    shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

    # Relaxed position detection
    avg_shoulder_y = (shoulder_l[1] + shoulder_r[1]) / 2
    avg_wrist_y = (wrist_l[1] + wrist_r[1]) / 2
    avg_hip_y = (hip_l[1] + hip_r[1]) / 2
    
    hands_down = avg_wrist_y >= avg_shoulder_y - 0.15
    body_forward = abs(avg_shoulder_y - avg_hip_y) < 0.4
    doing_pushup_motion = (avg_angle > 100 and avg_angle < 150)
    
    in_position = hands_down and body_forward or doing_pushup_motion
    
    return in_position, "Get in push-up position!" if not in_position else "Ready for push-ups!"

def detect_squat(landmarks, avg_angle):
    """Squat detection using knee angles"""
    hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    
    # Calculate knee angles
    knee_angle_l = calculate_angle(hip_l, knee_l, ankle_l)
    knee_angle_r = calculate_angle(hip_r, knee_r, ankle_r)
    knee_avg = (knee_angle_l + knee_angle_r) / 2
    
    # Standing position detection (relaxed)
    avg_hip_y = (hip_l[1] + hip_r[1]) / 2
    avg_ankle_y = (ankle_l[1] + ankle_r[1]) / 2
    standing = avg_ankle_y > avg_hip_y  # Feet below hips
    
    in_position = standing
    
    return in_position, knee_avg, "Stand up straight for squats!" if not in_position else "Ready for squats!"

def detect_bicep_curl(landmarks, avg_angle):
    """Bicep curl detection using elbow angles"""
    shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    
    # Arms should be in curl position (relaxed check)
    avg_shoulder_y = (shoulder_l[1] + shoulder_r[1]) / 2
    avg_wrist_y = (wrist_l[1] + wrist_r[1]) / 2
    arms_visible = abs(avg_shoulder_y - avg_wrist_y) < 0.5  # Very relaxed
    
    in_position = arms_visible
    
    return in_position, "Position arms for curls!" if not in_position else "Ready for bicep curls!"

def detect_jumping_jack(landmarks):
    """Jumping jack detection using arm and leg spread"""
    shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    
    # Calculate spreads
    arm_spread = calculate_distance(wrist_l, wrist_r)
    leg_spread = calculate_distance(ankle_l, ankle_r)
    
    # Average spread for detection
    avg_spread = (arm_spread + leg_spread) / 2
    
    # Always in position for jumping jacks
    in_position = True
    
    return in_position, avg_spread, "Ready for jumping jacks!"

def detect_shoulder_press(landmarks, avg_angle):
    """Shoulder press detection using arm elevation"""
    shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    
    # Arms should be visible for shoulder press
    avg_shoulder_y = (shoulder_l[1] + shoulder_r[1]) / 2
    avg_wrist_y = (wrist_l[1] + wrist_r[1]) / 2
    arms_ready = True  # Very relaxed - always ready
    
    in_position = arms_ready
    
    return in_position, "Ready for shoulder press!"

def process_frame():
    global counter, stage, exercise, data_log, start_time, prev_time, rep_times, x_data, y_data
    global last_count_time, last_encouragement_time, last_guidance_time, current_exercise_index

    ret, frame = cap.read()
    if not ret:
        print("❌ Camera error: Unable to capture frame")
        return False

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            current_time = time.time()
            
            # Get basic angles for all exercises
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
            angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r) 
            avg_angle = (angle_l + angle_r) / 2

            # Exercise-specific detection
            in_position = False
            feedback = ""
            metric_value = avg_angle
            
            if exercise == "push-up":
                in_position, feedback = detect_push_up(landmarks, avg_angle)
                thresholds = exercise_thresholds["push-up"]
                
            elif exercise == "squat":
                in_position, metric_value, feedback = detect_squat(landmarks, avg_angle)
                thresholds = exercise_thresholds["squat"]
                avg_angle = metric_value  # Use knee angle for squats
                
            elif exercise == "bicep-curl":
                in_position, feedback = detect_bicep_curl(landmarks, avg_angle)
                thresholds = exercise_thresholds["bicep-curl"]
                
            elif exercise == "jumping-jack":
                in_position, metric_value, feedback = detect_jumping_jack(landmarks)
                thresholds = exercise_thresholds["jumping-jack"]
                
            elif exercise == "shoulder-press":
                in_position, feedback = detect_shoulder_press(landmarks, avg_angle)
                thresholds = exercise_thresholds["shoulder-press"]

            # Exercise counting logic
            if in_position:
                time_since_last_count = current_time - last_count_time
                
                if exercise == "jumping-jack":
                    # Special logic for jumping jacks (spread-based)
                    if metric_value > thresholds["apart"]:  # Arms/legs apart
                        if stage == "together" and time_since_last_count > min_time_between_reps:
                            counter += 1
                            last_count_time = current_time
                            speak_async(str(counter))
                            
                            motivational_msg = get_motivational_message(counter, exercise)
                            if motivational_msg:
                                def delayed_motivation():
                                    time.sleep(0.8)
                                    speak(motivational_msg)
                                threading.Thread(target=delayed_motivation, daemon=True).start()
                            
                            print(f"🎉 Great {exercise}! Count: {counter}")
                            rep_times.append(current_time - prev_time)
                            prev_time = current_time
                            data_log.append([datetime.now().strftime("%H:%M:%S"), exercise, int(angle_l), int(angle_r), counter,
                                            current_time - start_time, 1 / np.mean(rep_times) if rep_times else 0])
                        
                        stage = "apart"
                        feedback = "Arms up! Great spread! 🤸"
                        
                    elif metric_value < thresholds["together"]:  # Arms/legs together
                        stage = "together"
                        feedback = "Good! Arms down! 👏"
                        
                else:
                    # Standard angle-based logic for other exercises
                    if avg_angle > thresholds["up"]:  # Up position
                        if stage == "down" and time_since_last_count > min_time_between_reps:
                            counter += 1
                            last_count_time = current_time
                            speak_async(str(counter))
                            
                            motivational_msg = get_motivational_message(counter, exercise)
                            if motivational_msg:
                                def delayed_motivation():
                                    time.sleep(0.8)
                                    speak(motivational_msg)
                                threading.Thread(target=delayed_motivation, daemon=True).start()
                            
                            print(f"🎉 Great {exercise}! Count: {counter}")
                            rep_times.append(current_time - prev_time)
                            prev_time = current_time
                            data_log.append([datetime.now().strftime("%H:%M:%S"), exercise, int(angle_l), int(angle_r), counter,
                                            current_time - start_time, 1 / np.mean(rep_times) if rep_times else 0])
                        
                        stage = "up"
                        feedback = "Perfect extension! 💪"
                        
                    elif avg_angle < thresholds["down"]:  # Down position
                        stage = "down"
                        feedback = "Great depth! Now extend! 🔥"
                        
                    else:
                        feedback = "Keep the motion going! 👍"
            else:
                stage = None

            # Display information
            cv2.putText(image, f'Left: {int(angle_l)}°', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f'Right: {int(angle_r)}°', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f'Metric: {metric_value:.1f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Feedback display
            feedback_color = (0, 255, 0) if "Perfect" in feedback or "Great" in feedback else (0, 200, 255)
            cv2.putText(image, feedback, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2)

            # Update graph
            graph_time = time.time() - start_time
            x_data.append(graph_time)
            y_data.append(metric_value if exercise == "jumping-jack" else avg_angle)
            
            while len(x_data) > 300:
                x_data.pop(0)
                y_data.pop(0)
            
            line.set_data(x_data, y_data)
            ax.set_xlim(max(0, graph_time - 30), max(30, graph_time))
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.001)

            # Enhanced display with recording indicator
            cv2.rectangle(image, (0, 0), (500, 300), (245, 117, 16), -1)
            cv2.putText(image, f'🏋️ {exercise.upper().replace("-", " ")} TRACKER 🏋️', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Recording indicator
            if is_recording:
                recording_time = time.time() - recording_start_time
                cv2.circle(image, (450, 30), 10, (0, 0, 255), -1)  # Red dot
                cv2.putText(image, f'🎥 REC {recording_time:.1f}s', (350, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.putText(image, str(counter), (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 4)
            cv2.putText(image, exercise.replace('-', ' ').upper(), (150, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            cv2.putText(image, f'Duration: {graph_time:.1f}s', (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, f'Speed: {1 / np.mean(rep_times) if rep_times else 0:.2f} rep/s', (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, f'Stage: {stage if stage else "Ready!"}', (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Exercise selection display
            cv2.putText(image, 'AVAILABLE EXERCISES:', (15, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            for i, ex in enumerate(exercises):
                color = (0, 255, 255) if i == current_exercise_index else (255, 255, 255)
                cv2.putText(image, f'{i+1}. {ex.replace("-", " ")}', (15, 205 + i*15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Recording controls display
            if is_recording:
                cv2.putText(image, '🎥 RECORDING - Press SPACE to stop', (15, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.putText(image, '📹 Press SPACE to start recording', (15, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Position status
            position_text = "✅ READY TO COUNT!" if in_position else "❌ GET IN POSITION"
            position_color = (0, 255, 0) if in_position else (0, 0, 255)
            cv2.putText(image, position_text, (300, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, position_color, 2)

            # Position indicator
            indicator_x, indicator_y = 380, 50
            if in_position:
                if (exercise == "jumping-jack" and metric_value > thresholds["apart"]) or \
                   (exercise != "jumping-jack" and avg_angle > thresholds.get("up", 120)):
                    cv2.circle(image, (indicator_x, indicator_y), 25, (0, 255, 0), -1)
                    cv2.putText(image, 'UP', (indicator_x-15, indicator_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                elif (exercise == "jumping-jack" and metric_value < thresholds["together"]) or \
                     (exercise != "jumping-jack" and avg_angle < thresholds.get("down", 90)):
                    cv2.circle(image, (indicator_x, indicator_y), 25, (0, 0, 255), -1)
                    cv2.putText(image, 'DOWN', (indicator_x-25, indicator_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    cv2.circle(image, (indicator_x, indicator_y), 25, (0, 255, 255), -1)
                    cv2.putText(image, 'MID', (indicator_x-18, indicator_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            else:
                cv2.circle(image, (indicator_x, indicator_y), 25, (128, 128, 128), -1)
                cv2.putText(image, 'WAIT', (indicator_x-25, indicator_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(66, 245, 230), thickness=2, circle_radius=2))
        else:
            cv2.putText(image, "🤖 Looking for you... Please step into view! 📸", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Show image
        cv2.imshow('💪 Multi-Exercise Workout Tracker 💪', image)
        
        # Record frame if recording is active
        if is_recording and video_writer:
            video_writer.write(image)

    except Exception as e:
        print(f"[❌] Error in process_frame: {e}")
        cv2.putText(image, f"Error: {str(e)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow('💪 Multi-Exercise Workout Tracker 💪', image)
    
    return True

# Main loop
print("🎉 WELCOME TO THE MULTI-EXERCISE WORKOUT TRACKER! 🎉")
print("="*60)
print("AVAILABLE EXERCISES:")
for i, ex in enumerate(exercises):
    print(f"  {i+1}. {ex.replace('-', ' ').title()}")
print("="*60)
print("CONTROLS:")
print("🏋️  Press 1-5 to switch exercises")
print("📱 Press 'q' to quit")
print("💾 Press 's' to save workout data")  
print("🔄 Press 'r' to reset current exercise counter")
print("🔊 Press 'v' to toggle voice feedback")
print("🎯 Press 'n' for next exercise")
print("🎯 Press 'p' for previous exercise")
print("🎥 Press SPACE to start/stop recording")
print("📱 Press 'q' to quit")
print("="*60)
print("📹 RECORDING FEATURES:")
print("✨ Records with all overlay graphics and counters")
print("🎬 Perfect for social media sharing (LinkedIn, Instagram)")
print("💾 Saves as MP4 format with timestamp")
print("🔴 Red recording indicator shows when active")
print("="*60)

if voice_enabled:
    speak(f"Welcome to your complete workout tracker! Starting with {exercise.replace('-', ' ')}s. Let's get fit!")

try:
    while True:
        if not process_frame():
            break
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if data_log:
                print("\n[💾] Saving your amazing workout data...")
                save_to_csv(data_log)
                save_to_excel(data_log)
                speak("Fantastic workout! Your data has been saved!")
                print("[✔] Data saved successfully! You're a fitness champion!")
            else:
                print("[⚠] No workout data to save yet - but you're doing great!")
        elif key == ord('r'):
            counter = 0
            stage = None
            data_log.append([datetime.now().strftime("%H:%M:%S"), f"{exercise}_RESET", 0, 0, 0, 0, 0])
            start_time = time.time()
            prev_time = start_time
            rep_times = []
            x_data.clear()
            y_data.clear()
            last_count_time = 0
            last_encouragement_time = 0
            last_guidance_time = 0
            speak(f"Reset complete! Ready for more {exercise.replace('-', ' ')}s!")
            print(f"[🔄] {exercise.replace('-', ' ').title()} counter reset!")
        elif key == ord('v'):
            voice_enabled = not voice_enabled
            status = "enabled" if voice_enabled else "disabled"
            print(f"[🔊] Voice feedback {status}")
            if voice_enabled:
                speak("Voice feedback is back! You're doing amazing!")
        elif key == ord('n'):
            # Next exercise
            current_exercise_index = (current_exercise_index + 1) % len(exercises)
            exercise = exercises[current_exercise_index]
            counter = 0
            stage = None
            x_data.clear()
            y_data.clear()
            last_count_time = 0
            speak(f"Switched to {exercise.replace('-', ' ')}s! Let's work those muscles!")
            print(f"[🎯] Switched to: {exercise.replace('-', ' ').title()}")
        elif key == ord('p'):
            # Previous exercise
            current_exercise_index = (current_exercise_index - 1) % len(exercises)
            exercise = exercises[current_exercise_index]
            counter = 0
            stage = None
            x_data.clear()
            y_data.clear()
            last_count_time = 0
            speak(f"Switched to {exercise.replace('-', ' ')}s! Time to get stronger!")
            print(f"[🎯] Switched to: {exercise.replace('-', ' ').title()}")
        elif key == ord(' '):  # Spacebar for recording
            if is_recording:
                stop_recording()
            else:
                start_recording()
        elif key >= ord('1') and key <= ord('5'):
            # Direct exercise selection
            exercise_num = key - ord('1')
            if exercise_num < len(exercises):
                current_exercise_index = exercise_num
                exercise = exercises[current_exercise_index]
                counter = 0
                stage = None
                x_data.clear()
                y_data.clear()
                last_count_time = 0
                speak(f"Great choice! Let's do some {exercise.replace('-', ' ')}s!")
                print(f"[🎯] Selected: {exercise.replace('-', ' ').title()}")

except KeyboardInterrupt:
    print("\n[⚠] Workout session ended by user")
    speak("Amazing workout session! You should be proud!")

finally:
    # Stop recording if still active
    if is_recording:
        stop_recording()
    
    # Cleanup and workout summary
    print("\n[🧹] Saving your workout progress...")
    cap.release()
    cv2.destroyAllWindows()
    plt.close('all')
    
    # Auto-save data if available
    if data_log:
        print("[💾] Auto-saving your fantastic workout session...")
        save_to_csv(data_log)
        
        # Calculate workout summary
        total_reps = sum(1 for entry in data_log if entry[4] > 0)  # Count entries with reps > 0
        workout_duration = time.time() - start_time
        exercises_done = list(set([entry[1] for entry in data_log if not entry[1].endswith('_RESET')]))
        
        print("\n" + "="*50)
        print("🏆 WORKOUT SUMMARY 🏆")
        print("="*50)
        print(f"💪 Total Reps Completed: {counter}")
        print(f"⏱️ Workout Duration: {workout_duration/60:.1f} minutes")
        print(f"🏋️ Exercises Done: {', '.join([ex.replace('-', ' ').title() for ex in exercises_done])}")
        print(f"🔥 Final Exercise: {exercise.replace('-', ' ').title()}")
        print(f"⚡ Average Speed: {1 / np.mean(rep_times) if rep_times else 0:.2f} reps/second")
        if recorded_filename:
            print(f"🎬 Video Recorded: {recorded_filename}")
            print("📱 Ready for social media sharing!")
        print("="*50)
        print(f"[✔] You completed {counter} {exercise.replace('-', ' ')}s! Session data saved!")
        
        if voice_enabled:
            speak(f"Incredible workout! You completed {counter} {exercise.replace('-', ' ')}s in {workout_duration/60:.1f} minutes! You're getting stronger every day!")
    else:
        print("[👋] Thanks for trying the multi-exercise workout tracker!")
        if voice_enabled:
            speak("Thanks for working out with me! Remember, every step counts toward your fitness goals!")
    
    print("\n🌟 KEEP UP THE AMAZING WORK! 🌟")
    print("💪 Your dedication to fitness is inspiring!")
    print("🚀 Tomorrow is another opportunity to get stronger!")
    print("✨ You're building a healthier, stronger you!")
