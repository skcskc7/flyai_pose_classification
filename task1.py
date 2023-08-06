import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0)

pairs = list(mp_pose.POSE_CONNECTIONS)

# initializing the webcam
cap = cv2.VideoCapture(0)

while True:
	ret, image = cap.read()
	if not ret:
		break
	image_height, image_width, _ = image.shape
	
	# predict pose
	rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	results = pose.process(rgb_image)
	if not results.pose_landmarks:
		continue
	
	landmarks = results.pose_landmarks.landmark
	
	# (Task1) Draw pose landmarks
	

	# (Task2) Pose classification


	# quit
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()