#importing necessary files
from PIL import Image, ImageDraw #import pillow
import dlib 
import cv2
import time
import face_recognition
import numpy as np

vid = cv2.VideoCapture(0) #initializing vid capture
background_capture = cv2.VideoCapture(r'./a.avi')


#initializing empty lists for data of faces tracked
face_locations = [] 
face_encodings = []

process_this_frame = True



#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor()

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
	#face_landmarks_list = face_recognition.face_landmarks(frame)
	
	#for face_landmarks in face_landmarks_list:
	#	pil_image = Image.fromarray(frame)
	#	d = ImageDraw.Draw(pil_image, 'RGBA')

	#	d.polygon(face_landmarks['left_eye'], fill = (255, 255, 255, 30))
	#	d.polygon(face_landmarks['right_eye'], fill = (255, 255, 255, 30))

	#	pil_image.show()
	#counter += 1
	#start_time_extract_figure = time.time()
	#_, frame = vid.read()
	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	#mask = np.zeros(frame.shape[:2], np.uint8)
	#bgdModel = np.zeros((1, 65), np.float64)
	#fgdModel = np.zeros((1,65), np.float64)
	#rect = (200,50,300,400)
	#start_time_grabCut = time.time()
	#cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
	#during_time_grabCut = time.time() - start_time_grabCut
	#print('{}-th t_time: {}'.format(counter, during_time_grabCut))
	#mask2 = np.where((mask == 2) | (mask == 0), (0,), (1,)).astype('uint8')
	#frame = frame * mask2[:, :, np.newaxis]
	#elapsed_time_extract_figure = time.time() -start_time_extract_figure
	#print('{}-th extract_figure_time: {}'.format(counter, elapsed_time_extract_figure))

	#start_time_combination = time.time()
	#ret, background = background_capture.read()
	#background = cv2.resize(background, (1366,768), interpolation=cv2.INTER_AREA)


	#mask_1 = frame > 0
	#mask_2 = frame <= 0
	#combination = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)*mask_1 + background * mask_2
	#elapsed_time_combination = time.time() - start_time_combination
	#print('{}-th combination_time: {}'.format(counter, elapsed_time_combination))

	#cv2.imshow('combination', combination)

	frame = cv2.copyMakeBorder(frame, 20,20,20,20, cv2.BORDER_CONSTANT, value = [146,74,54])
	frame = cv2.copyMakeBorder(frame, 10,10,10,10,cv2.BORDER_CONSTANT,value = [195,151,106])
	

	for (top, right, bottom, left) in face_locations: #looping the captured face
		top *= 4 #increasing the size of the image since we decreased it
		right *= 4
		bottom *= 4
		left *= 4
		
			
		face_image = frame[top:bottom, left:right] 
		#face_image = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 1)
		face_image = cv2.GaussianBlur(face_image, (89,89), 50)
		#setting up our simple filter
		frame[top:bottom, left:right] = face_image
		#building up a rectangle on an identified face
		cv2.rectangle(frame, (left, top), (right, bottom), (172,113,74), 2)
		font = cv2.FONT_HERSHEY_TRIPLEX
		cv2.putText(frame, 'Guess Who?', (50,450), font, 3, (58,16,14), 2, cv2.LINE_AA)	
	cv2.imshow('Video', frame) #output image
	if cv2.waitKey(1) & 0xFF == ord('q'): #to exit the loop
		break


#to avoid data loss
vid.release()
cv2.destroyAllWindows()

 
