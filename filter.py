#importing necessary files
from PIL import Image, ImageDraw #import pillow
import dlib 
import cv2
import face_recognition
import numpy as np

vid = cv2.VideoCapture(0) #initializing vid capture

#initializing empty lists for data of faces tracked
face_locations = [] 
face_encodings = []

process_this_frame = True



#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor()


while True:	#setting up the loop for continuous webcam capture
	ret, frame = vid.read() #initializing frame object 
	
	small_frame = cv2.resize(frame, None , fx = 0.25, fy = 0.25) #decreasing the size of frame for easier capture
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
	rgb_small_frame = small_frame[:, :, ::-1]

	if process_this_frame: #used to list recognized faces
		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
	
	process_this_frame = not process_this_frame
	#face_landmarks_list = face_recognition.face_landmarks(frame)
	
	#for face_landmarks in face_landmarks_list:
	#	pil_image = Image.fromarray(frame)
	#	d = ImageDraw.Draw(pil_image, 'RGBA')

	#	d.polygon(face_landmarks['left_eye'], fill = (255, 255, 255, 30))
	#	d.polygon(face_landmarks['right_eye'], fill = (255, 255, 255, 30))

	#	pil_image.show()

	for (top, right, bottom, left) in face_locations: #looping the captured face
		top *= 4 #increasing the size of the image since we decreased it
		right *= 4
		bottom *= 4
		left *= 4
		
		face_image = frame[top:bottom, left:right] 
		#face_image = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 1)
		face_image = cv2.GaussianBlur(face_image, (99,99), 30)
		#setting up our simple filter
		frame[top:bottom, left:right] = face_image
		#building up a rectangle on an identified face
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0,0), 2)
		
	cv2.imshow('Video', frame) #output image
	if cv2.waitKey(1) & 0xFF == ord('q'): #to exit the loop
		break


#to avoid data loss
vid.release()
cv2.destroyAllWindows()

 
