import cv2
import time
import os


# data folder should be clean
class DataCollector:
    def __init__(
        self,
        pose_name: str,
        output_dir,
        total_pictures=15,
        break_time_sec=3,
        camera_id=0,
    ):
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

    def cleanup(self):
        """Release camera and close windows"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")

    def take_pictures_series(self):
        print("Press 'q' to go to the next pose")
        print("Press 'Esc' to quit")

        self.init_camera()

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_counter = 0
        picture_counter = 0

        while picture_counter < self.total_pictures:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame_counter += 1
            timer = frame_counter % (fps * self.break_time)
            flipped = cv2.flip(frame, 1)

            cv2.putText(
                flipped,
                f"Pose: {self.pose_name} - Picture: {picture_counter}/{self.total_pictures}]",
                (50, 50),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (100, 200, 100),
                2,
            )
            cv2.putText(
                flipped,
                f"Time: {int(timer / fps) + 1}/{self.break_time}sec",
                (50, 100),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (100, 100, 200),
                2,
            )

            cv2.imshow("Camera Feed", flipped)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Aborted by user.")
                break

            if key == 27:
                print("Exiting...")
                exit()

            if timer == 0:
                filename = f"photo_{self.pose_name.lower().replace(' ', '_')}_{picture_counter + 1}.bmp"
                filepath = os.path.join(self.output_dir, filename)
                cv2.imwrite(filepath, frame)
                print(f"Saved {filepath}")
                picture_counter += 1

        print("Pictures series completed")
        self.cleanup()


poses = ["Hello", "Thank you", "I love you", "Yes", "No", "Please", "Albania"]
for pose in poses:
    name_dir = pose.replace(" ", "_").lower()
    output_dir = f"./data/{name_dir}/"

    # Make the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create the DataCollector instance and capture
    data = DataCollector(pose, output_dir)
    data.take_pictures_series()
