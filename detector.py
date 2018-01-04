import cv2
import numpy as np
import playsound
from threading import Thread
import imutils
import time
import dlib
from imutils import face_utils
from imutils.video import VideoStream


def euclidean_dist(ptA, ptB):
	return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
	A = euclidean_dist(eye[1], eye[5])
	B = euclidean_dist(eye[2], eye[4])
	C = euclidean_dist(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def mouth_aspect_ratio(mouth):
	mouth_x = euclidean_dist(mouth[0],mouth[6])
	mouth_y1 = euclidean_dist(mouth[2],mouth[10])
	mouth_y2 = euclidean_dist(mouth[4],mouth[8])
	mou = (mouth_y1 + mouth_y2) / (2.0 * mouth_x)
	print(mou)
	return mou

def warnning():
	path='warning.wav'
	playsound.playsound(path)

EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 14
MOUTH_AR_CONSEC_FRAMES = 10
MOUTH_AR_THRESH = 0.7

COUNTER = 0
MOUTH_COUNTER = 0
ALARM_ON = False


detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(1.0)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=800)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	for (x, y, w, h) in rects:
		rect = dlib.rectangle(int(x), int(y), int(x + w),
			int(y + h))

		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		mouth_outter = shape[48:60]
		mouth_inner = shape[60:68]
		leftEye = shape[42:48]
		rightEye = shape[36:42]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		ear = (leftEAR + rightEAR) / 2.0
		mar = mouth_aspect_ratio(mouth_outter)

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)

		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [mouth_inner], -1, (0, 255, 0), -1)
		cv2.drawContours(frame, [mouth_outter], -1, (0, 255, 0), 1)
		
		if mar > MOUTH_AR_THRESH:
			MOUTH_COUNTER += 1
		
		if ear < EYE_AR_THRESH:
			COUNTER += 1
			if (COUNTER >= EYE_AR_CONSEC_FRAMES) or (MOUTH_COUNTER >= MOUTH_AR_CONSEC_FRAMES):
				cv2.putText(frame, "WARNNING!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				if not ALARM_ON:
				 	ALARM_ON = True
				 	t = Thread(target=warnning)
				 	t.deamon = True
				 	t.start()
		else:
			COUNTER = 0
			MOUTH_COUNTER = 0
			ALARM_ON = False

		cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "MAR: {:.3f}".format(mar), (600, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()