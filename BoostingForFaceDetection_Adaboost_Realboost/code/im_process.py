from sklearn.feature_extraction import image as IMG
import numpy as np
import cv2
from utils import integrate_images

#extract patches from the image for all scales of the image
#return the INTEGREATED images and the coordinates of the patches
def image2patches(scales, image, patch_w = 16, patch_h = 16,toNormalize = True):
	all_patches = np.zeros((0, patch_h, patch_w))
	all_x1y1x2y2 = []
	for s in scales:
		simage = cv2.resize(image, None, fx = s, fy = s, interpolation = cv2.INTER_CUBIC)
		height, width = simage.shape
		print('Image shape is: %d X %d' % (height, width))
		patches = IMG.extract_patches_2d(simage, (patch_w, patch_h)) # move along the row first

		total_patch = patches.shape[0]
		row_patch = (height - patch_h + 1)
		col_patch = (width - patch_w + 1)
		assert(total_patch == row_patch * col_patch)
		scale_xyxy = []
		for pid in range(total_patch):
			y1 = pid / col_patch
			x1 = pid % col_patch
			y2 = y1 + patch_h - 1
			x2 = x1 + patch_w - 1
			scale_xyxy.append([int(x1 / s), int(y1 / s), int(x2 / s), int(y2 / s)])
		all_patches = np.concatenate((all_patches, patches), axis = 0)
		all_x1y1x2y2 += scale_xyxy
	if(toNormalize):
		return integrate_images(normalize(all_patches)), all_x1y1x2y2
	else:
		return all_patches, all_x1y1x2y2


#return a vector of prediction (0/1) after nms, same length as scores
#input: [x1, y1, x2, y2, score], threshold used for nms
#output: [x1, y1, x2, y2, score] after nms
def nms(xyxys, overlap_thresh):
	######################
	######## TODO ########
	######################
	boxes = xyxys
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
											   np.where(overlap > overlap_thresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")
	#return

def normalize(images):
	standard = np.std(images)
	images = (images - np.min(images)) / (np.max(images) - np.min(images))
	return images

def main():
	original_img = cv2.imread('Testing_Images/Face_1.jpg', cv2.IMREAD_GRAYSCALE)
	scales = 1 / np.linspace(1, 10, 46)
	patches, patch_xyxy = image2patches(scales, original_img)
	print(patches.shape)
	print(len(patch_xyxy))
if __name__ == '__main__':
	main()
