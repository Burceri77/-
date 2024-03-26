# -
智能私人教练利用边缘计算和计算机视觉技术,实时分析用户的运动姿势和表现,提供个性化的指导和反馈,帮助用户安全有效地进行体育锻炼。
import cv2
import mediapipe as mp

# Initialize mediapipe Pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize mediapipe drawing class, to draw the keypoints and connections on the image.
mp_drawing = mp.solutions.drawing_utils

def analyze_posture(image):
    """
    Analyze the posture of a person in the given image.
    
    Args:
    image: The image frame from the video stream.
    
    Returns:
    image: The image frame with drawn posture keypoints and connections.
    """
    # Convert the image color from BGR to RGB for mediapipe processing.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image to find the pose.
    results = pose.process(image_rgb)
    
    # Draw the pose annotations on the image.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    return image

def main():
    # Capture video from the webcam (ID 0).
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Analyze posture in the captured frame.
        frame = analyze_posture(frame)
        
        # Display the annotated frame.
        cv2.imshow('Intelligent Personal Trainer', frame)
        
        # Break the loop when 'q' is pressed.
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    # Release the capture after finishing.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
