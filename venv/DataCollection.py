import cv2
import time
import os


class DataCollector:
	def __init__(self, pose_name: str, output_dir, total_pictures = 10, break_time_sec = 5, camera_id = 0):
		self.pose_name = pose_name
		self.output_dir = output_dir
		self.total_pictures = total_pictures
		self.break_time = break_time_sec 
		self.camera_id = camera_id

		self.cap = None
		self.key = None
		self.frame = None

	def init_camera(self):
		self.cap = cv2.VideoCapture(self.camera_id)
		
		if not self.cap.isOpened():
			print("Error: Could not open camera.\n")
			return

		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
		self.cap.set(cv2.CAP_PROP_FPS, 30)	

		print("Camera succesfully initialized.\n")

	def main(self):#main ne marche ps 
		self.init_camera()
		_, self.frame = self.cap.read()

		
		key = cv2.waitKey(1) & 0xFF


		print("Press 'ESC' to abort early.")

		while self.cap.isOpened():
			cv2.imshow("Camera Feed", cv2.flip(self.frame, 1))

			if key == 27:
				print("Aborted by user")
				self.cap.release()
				return
			
			if key == ord('f'):
				self.take_pictures_series()
		
		self.cap.release()


	def take_pictures_series(self):

			fps = self.cap.get(cv2.CAP_PROP_FPS)
			frame_counter = 0
			picture_counter = 0

			while picture_counter < self.total_pictures:
				# ret, frame = self.cap.read()
				# if not ret:
				# 	continue

				frame_counter += 1

				# flipped = cv2.flip(frame, 1)
				# cv2.imshow("Camera Feed", flipped)
				# key = cv2.waitKey(1) & 0xFF

				# if key == 27:
				# 	print("Aborted by user.")
				# 	break

				if frame_counter % (fps * self.break_time) == 0:
					filename = f"photo_{self.pose_name}_{picture_counter + 1}.bmp"
					filepath = os.path.join(self.output_dir, filename)
					cv2.imwrite(filepath, self.frame)
					print(f"Saved {filepath}")
					picture_counter += 1
	
poses = ["Hello", "Thank you", "I love you"]
for pose in poses:
    name_dir = pose.replace(" ", "_").lower()
    output_dir = f"./data/{name_dir}/"
    
    # Make the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create the DataCollector instance and capture
    data = DataCollector(pose, output_dir, total_pictures=3, break_time_sec=1)
    data.main()

