#importing necessary files
import dlib 
import cv2
import face_recognition
import numpy as np

vid = cv2.VideoCapture(0) #initializing vid capture
#initializing empty lists for data of faces tracked
face_locations = [] 
face_encodings = []

process_this_frame = True

counter = -1
while True:	#setting up the loop for continuous webcam capture
	ret, frame = vid.read() #initializing frame object 
	
	small_frame = cv2.resize(frame, None , fx = 0.25, fy = 0.25) #decreasing the size of frame for easier capture
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
	rgb_small_frame = small_frame[:, :, ::-1]

	if process_this_frame: #used to list recognized faces
		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
	
	process_this_frame = not process_this_frame
	#setting up the borders
	frame = cv2.copyMakeBorder(frame, 20,20,20,20, cv2.BORDER_CONSTANT, value = [146,74,54])
	frame = cv2.copyMakeBorder(frame, 10,10,10,10,cv2.BORDER_CONSTANT,value = [195,151,106])
	

	for (top, right, bottom, left) in face_locations: #looping the captured face
		top *= 4 #increasing the size of the image since we decreased it
		right *= 4
		bottom *= 4
		left *= 4
		
			
		face_image = frame[top:bottom, left:right] 
		face_image = cv2.GaussianBlur(face_image, (89,89), 50)
		#setting up our simple filter
		frame[top:bottom, left:right] = face_image
		#building up a rectangle on an identified face
		cv2.rectangle(frame, (left, top), (right, bottom), (172,113,74), 2)
		#adding texts
		font = cv2.FONT_HERSHEY_TRIPLEX
		cv2.putText(frame, 'Guess Who?', (50,450), font, 3, (58,16,14), 2, cv2.LINE_AA)
		cv2.putText(frame, '#simplicityisbeauty', (35,500), font, 0.5, (111,111,111), 1, cv2.LINE_AA)	
		cv2.putText(frame, '#111dabest', (570,500), font, 0.5, (111,111,111), 1, cv2.LINE_AA)
	cv2.imshow('Video', frame) #output image
	if cv2.waitKey(1) & 0xFF == ord('q'): #to exit the loop
		break


#to avoid data loss
vid.release()
cv2.destroyAllWindows()

 
