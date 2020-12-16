import cv2 as cv


def show_video(vis_mask):
	rgb = cv.cvtColor(vis_mask, cv.COLOR_HSV2BGR)
	cv.imshow('preprocessed data example', rgb)
	# if cv.waitKey(1) & 0xFF == ord('q'):
	# 	break