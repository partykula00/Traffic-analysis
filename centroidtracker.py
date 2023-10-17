from scipy.spatial import distance as dist
from collections import OrderedDict
from velocity_calculation import calculate_speed
import cv2
import numpy as np
from csv_writer import Detected

class CentroidTracker():
	def __init__(self, maxDisappeared=5):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.boxes = OrderedDict()
		self.classes = OrderedDict()
		self.confidence = OrderedDict()
		self.max_speed = OrderedDict()
		self.speed = OrderedDict()
		self.first_position = OrderedDict()
		self.frame = OrderedDict()
		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared
		self.detected = Detected()

	def register(self, centroid, rects, classes, confidence, speed, frame):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.boxes[self.nextObjectID] = rects
		self.classes[self.nextObjectID] = classes
		self.confidence[self.nextObjectID] = confidence
		self.speed[self.nextObjectID] = speed
		self.max_speed[self.nextObjectID] = speed
		print(self.max_speed)
		self.first_position[self.nextObjectID] = centroid
		self.frame[self.nextObjectID] = frame
		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		print(self.max_speed)
		self.detected.save([str(objectID),str(self.classes[objectID]),str(self.frame[objectID]),str(self.confidence[objectID]),str(self.max_speed[objectID])])
		del self.objects[objectID]
		del self.disappeared[objectID]
		del self.boxes[objectID]
		del self.classes[objectID]
		del self.speed[objectID]
		del self.max_speed[objectID]
		del self.confidence[objectID]
		del self.frame[objectID]


	def saved (self, objectID):
		if self.speed > self.max_speed:
			self.max_speed = self.speed




	def update(self, rects, classes, confidence, speed, fps,frame, pixel_ratio=0):
		# check to see if the list of input bounding box rectangles
		# is empty
		#print(frame)
		#print(rects)
		#print(classes)
		#print(confidence)
		#print(speed)
		j=0
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			# return early as there are no centroids or tracking info
			# to update
			return self.objects, self.boxes, self.classes, self.speed

		# initialize an array of input centroids for the current frame
		#inputCentroids = np.zeros((len(rects), 2), dtype="int")
		inputCentroids2=[]
		classes2=[]
		rects2=[]
		confidence2=[]
		speed2=[]
		frame2=[]
		# loop over the bounding box rectangles
		for (i, (startY, startX, endY, endX)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			#print(i)
			if cX < 220 or cY < 290:
				# ROI limits
				pass
			else:
				#speed[i] = calculate_speed(inputCentroids[i], frame)
				#print(j)
				inputCentroids2.append((cX, cY))
				classes2.append(classes[i])
				rects2.append(rects[i])
				confidence2.append(confidence[i])
				speed2.append(speed[i])
				frame2.append(frame[i])
				#print("są w środku ramki")
				#print(inputCentroids[j])
				#print(j)

		inputCentroids = np.zeros((len(inputCentroids2), 2), dtype="int")
		inputCentroids = np.array(inputCentroids2, dtype="int")

		rects = rects2
		classes = classes2
		confidence = confidence2
		speed = speed2
		frame = frame2

		#print(inputCentroids)
		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		#print(inputCentroids)
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i], rects[i], classes[i], confidence[i], speed[i], frame[i])

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids

		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			#print(objectCentroids)
			#print(inputCentroids)
			if len(inputCentroids) == 0:
				pass
			else:
				D = dist.cdist(np.array(objectCentroids), inputCentroids)

				#print(D)

				# in order to perform this matching we must (1) find the
				# smallest value in each row and then (2) sort the row
				# indexes based on their minimum values so that the row
				# with the smallest value as at the *front* of the index
				# list
				rows = D.min(axis=1).argsort()

				# next, we perform a similar process on the columns by
				# finding the smallest value in each column and then
				# sorting using the previously computed row index list
				cols = D.argmin(axis=1)[rows]

				# in order to determine if we need to update, register,
				# or deregister an object we need to keep track of which
				# of the rows and column indexes we have already examined
				usedRows = set()
				usedCols = set()

				# loop over the combination of the (row, column) index
				# tuples
				for (row, col) in zip(rows, cols):
					# if we have already examined either the row or
					# column value before, ignore it
					# val
					if row in usedRows or col in usedCols:
						continue

					# otherwise, grab the object ID for the current row,
					# set its new centroid, and reset the disappeared
					# counter
					objectID = objectIDs[row]
					#print(type(self.objects[objectID]), type(inputCentroids[col]))
					if self.frame[objectID] + 8 < frame[col]:
						frames_passed = abs(self.frame[objectID] - frame[col])
						#print(frames_passed)
						self.speed[objectID] = calculate_speed(self.first_position[objectID], inputCentroids[col],fps, frames_passed)
						#Tu se zmieniłem
						if self.max_speed[objectID] is None:
							self.max_speed[objectID] = self.speed[objectID]

						if self.speed[objectID] > self.max_speed[objectID]:
							self.max_speed[objectID] = self.speed[objectID]

						self.first_position[objectID] = inputCentroids[col]
						self.frame[objectID] = frame[col]

					self.objects[objectID] = inputCentroids[col]
					self.boxes[objectID] = rects[col]
					if self.confidence[objectID] < confidence[col] + 10:
						self.classes[objectID] = classes[col]
						self.confidence[objectID] = confidence[col]
						#detect = Detected(objectID, self.classes[objectID], self.)

					#self.classes[objectID] = classes[col]
					self.disappeared[objectID] = 0

					# indicate that we have examined each of the row and
					# column indexes, respectively
					usedRows.add(row)
					usedCols.add(col)

				# compute both the row and column index we have NOT yet
				# examined
				unusedRows = set(range(0, D.shape[0])).difference(usedRows)
				unusedCols = set(range(0, D.shape[1])).difference(usedCols)

				# in the event that the number of object centroids is
				# equal or greater than the number of input centroids
				# we need to check and see if some of these objects have
				# potentially disappeared
				if D.shape[0] >= D.shape[1]:
					# loop over the unused row indexes
					for row in unusedRows:
						# grab the object ID for the corresponding row
						# index and increment the disappeared counter
						objectID = objectIDs[row]
						self.disappeared[objectID] += 1

						# check to see if the number of consecutive
						# frames the object has been marked "disappeared"
						# for warrants deregistering the object
						if self.disappeared[objectID] > self.maxDisappeared:
							self.deregister(objectID)

				# otherwise, if the number of input centroids is greater
				# than the number of existing object centroids we need to
				# register each new input centroid as a trackable object
				else:
					for col in unusedCols:
						self.register(inputCentroids[col], rects[col], classes[col], confidence[col], speed[col], frame[col])

		# return the set of trackable objects
		if not self.objects:
			#self.objects=[]
			print("brakuje centroidów")
		if not self.boxes:
			#self.boxes=[]
			print("brakuje boxes")
		if not self.speed:
			#self.speed=[]
			print("brakuje predkosci")
		if not self.classes:
			#self.classes=[]
			print("brakuje klas")

		return self.objects, self.boxes, self.classes, self.speed

# test_list = [[272, 307, 204, 218], [155, 190, 205, 220], [696, 718, 270, 283], [776, 811, 302, 330], [236, 308, 350, 393]]
# test_list2 = [[240, 302, 198, 118], [151, 184, 205, 219], [224, 295, 351, 396], [460, 504, 324, 356], [696, 717, 269, 282]]
# test_class = [3, 3, 8, 8, 3]
# test_class2 = [3, 3, 3, 3, 8]
# speed1 = [0,0,0,0,0]
# speed2 = [0,0,0,0,0]
# test_confidence1 = [60, 65, 55, 47, 89]
# test_confidence2 = [70, 56, 55, 57, 78]
#
# fps = 2
# tracking = CentroidTracker()
#
# height=650
# width=650
# blank_image = np.zeros((height,width,3), np.uint8)
#
# tracking.update(test_list, test_class, test_confidence1,speed1, fps)
# for (objectID, centroid) in tracking.objects.items():
# 	print(centroid)
# 	cv2.circle(blank_image, (centroid[0], centroid[1]), 4, (0,255,0), -1)
#
#
#
# tracking.update(test_list2, test_class2, test_confidence2,speed2, fps)
# for (objectID, centroid) in tracking.objects.items():
# 	print(centroid)
# 	cv2.circle(blank_image, (centroid[0], centroid[1]), 4, (0,0,255), -1)
#
# cv2.imshow("image", blank_image)
# cv2.waitKey()
# # print(tracking.speed)
# #wizualziacja
#
# height=800
# width=800
# blank_image = np.zeros((height,width,3), np.uint8)
#

#print(tracking.objects)
#print(tracking.boxes)
#print(tracking.classes)
#print(tracking.classes[0])
