import cv2
import mediapipe as mp
import os 

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

DATA_DICTIONARY = {} #data_file, metadata_file
path = "./data/"
for folder in os.listdir(path):
	sub_path = os.path.join(path, folder)
	for bmp_file in os.listdir(sub_path):
		bmp_file_path = os.path.join(sub_path, bmp_file)
		file_name_no_ext = os.path.splitext(os.path.basename(bmp_file))[0]
		txt_file_name = f"{file_name_no_ext}.txt"
		txt_file_path = os.path.join(sub_path, txt_file_name)
		DATA_DICTIONARY[bmp_file_path] = txt_file_path
		try:
			with open(txt_file_path, "w") as f:
				pass
		except FileExistsError:
			# I think I should clean all txt from the folder in case we do the same pose but with a different picture
			pass
		except Exception as e:
			print(f"Error occured while creating {txt_file_name} : {e}")
		
print(DATA_DICTIONARY)
		
with mp_hands.Hands(
	static_image_mode=True,
	max_num_hands=2,
	min_detection_confidence=0.5) as hands:
	for data_file, metadata_file in DATA_DICTIONARY.items():
		print(data_file)

		if not os.path.exists(data_file):
			print(f"File: {data_file} do not exist\n")
			continue

		image = cv2.imread(data_file)
		image = cv2.flip(image, 1)
		if image is None:
			print("Image couldn't be loaded")
			continue

		results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
		with open(metadata_file, "w") as f:
		# print('Handedness:', results.multi_handedness)
			f.write(f"Handedness: {results.multi_handedness}\n")
			if not results.multi_hand_landmarks:
				continue
			image_height, image_width, _ = image.shape
			annotated_image = image.copy()
			for hand_landmarks in results.multi_hand_landmarks:
				# f.write(f"\nHand {idx + 1}:\n")
				f.write(f"hand_landmarks: {hand_landmarks}\n")
				
				index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
				tip_x = index_tip.x * image_width
				tip_y = index_tip.y * image_height
				
				f.write(f"Index finger tip coordinates:\n")
				f.write(f"X: {tip_x}\n")
				f.write(f"Y: {tip_y}\n")
			mp_drawing.draw_landmarks(
				annotated_image,
				hand_landmarks,
				mp_hands.HAND_CONNECTIONS,
				mp_drawing_styles.get_default_hand_landmarks_style(),
				mp_drawing_styles.get_default_hand_connections_style())
