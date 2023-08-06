import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0)

pairs = list(mp_pose.POSE_CONNECTIONS)

# init matplot
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.view_init(azim=-90, elev=-90)
		
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
	if not results.pose_world_landmarks:
		continue
	
	world_landmarks = results.pose_world_landmarks.landmark

	# Plot 3D landmarks
	ax.clear()
	ax.set_xlim3d(-1, 1)
	ax.set_ylim3d(-1, 1)
	ax.set_zlim3d(-1, 1)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	
	# draw limbs
	for pair in pairs:
		idx1 = pair[0]
		idx2 = pair[1]  

		x_pair = [world_landmarks[idx1].x, world_landmarks[idx2].x]
		y_pair = [world_landmarks[idx1].y, world_landmarks[idx2].y]
		z_pair = [world_landmarks[idx1].z, world_landmarks[idx2].z]
		
		ax.plot(x_pair, y_pair, zs=z_pair, linewidth=3)
	
	plt.pause(0.0001)
	
	# (Task1) Squat counter 
