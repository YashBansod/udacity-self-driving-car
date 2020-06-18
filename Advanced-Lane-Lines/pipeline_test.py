#!/usr/bin/env python

# ******************************************    Libraries to be imported    ****************************************** #
from __future__ import print_function
# noinspection PyPackageRequirements
import numpy as np
import matplotlib.image as mpimg
import cv2
from glob import glob
from moviepy.editor import VideoFileClip


# ******************************************    Func Declaration Start      ****************************************** #
# Lets create a helper function that handles the entire process of computing calibration parameters for a camera
def calibrate_camera_distortion(image_paths, nx=9, ny=6):
    obj = np.zeros((nx * ny, 3), np.float32)
    obj[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    img = None
    for img_path in image_paths:
        img = mpimg.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret is True:
            imgpoints.append(corners)
            objpoints.append(obj)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


# ******************************************    Func Declaration Start      ****************************************** #
# Lets write a simple pipeline to convert the orignal RGB image to a binary image that accentuates the lane lines
def binary_img_pipe(image, gray_thresh=250, b_thresh=160):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    b_channel = lab[:, :, 2]
    b_channel = (((b_channel - b_channel.min()) / b_channel.ptp()) * 255).astype(np.uint8)
    combined_binary = np.zeros_like(gray)
    combined_binary[(gray > gray_thresh) | (b_channel > b_thresh)] = 255
    return combined_binary


# ******************************************    Func Declaration Start      ****************************************** #
# perspective transform
def perspective_transform(image, reverse=False):
    imshape = image.shape
    src = np.float32([[(200, 680), (575, 455), (705, 455), (1100, 680)]])
    dst = np.float32([[350, 720], [350, 0], [950, 0], [950, 720]])
    #     src = np.float32([[(200,680),(505, 500),(790, 500), (1100,680)]])
    #     dst = np.float32([[200,680],[200,40],[1100,40],[1100,680]])
    # reverse - to get back the original image from birds-eye view, swap the points for perspective transform
    if reverse:
        mat = cv2.getPerspectiveTransform(dst, src)
    else:
        mat = cv2.getPerspectiveTransform(src, dst)
    img_size = (imshape[1], imshape[0])
    return cv2.warpPerspective(image, mat, img_size, flags=cv2.INTER_LINEAR)


# ******************************************    Func Declaration Start      ****************************************** #
def hist(image):
    return np.sum(image[image.shape[0] // 2:, :], axis=0)


# ******************************************    Func Declaration Start      ****************************************** #
def find_lane_pixels(binary_warped, prev_left_fit, prev_right_fit):
    # Take a histogram of the bottom half of the image
    histogram = hist(binary_warped)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # HYPERPARAMETERS
    # number of sliding windows
    nwindows = 9
    # width of the windows +/- margin
    margin = 80
    # minimum number of pixels found to recenter window
    minpix = 30
    # height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    # empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        # four  boundaries of the window
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    # concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # fit a second order polynomial
    try:
        if len(leftx) < 1000:
            raise TypeError
        left_fit = np.polyfit(lefty, leftx, 2)
        prev_left_fit.append(left_fit)
    except TypeError:
        pass

    try:
        if len(rightx) < 500:
            raise TypeError
        right_fit = np.polyfit(righty, rightx, 2)
        prev_right_fit.append(right_fit)
    except TypeError:
        pass

    left_fit = np.empty((len(prev_left_fit), 3))
    right_fit = np.empty((len(prev_right_fit), 3))

    for i in range(len(prev_left_fit)):
        left_fit[i] = np.array(prev_left_fit[i])

    for i in range(len(prev_right_fit)):
        right_fit[i] = np.array(prev_right_fit[i])

    left_fit = left_fit.mean(axis=0)
    right_fit = right_fit.mean(axis=0)

    # generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    # calc both polynomials using ploty, left_fit and right_fit
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # create an image to draw on and an image to show the selection window
    out_img = np.zeros([binary_warped.shape[0], binary_warped.shape[1], 3], np.uint8)
    window_img = np.zeros_like(out_img)
    # color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    # new margin : mid of the left and right polynomial : polynomial for center of the lane
    marginx = (right_fitx - left_fitx) / 2
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + marginx, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - marginx, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    # draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    return result, left_fitx, right_fitx, ploty


# ******************************************    Func Declaration Start      ****************************************** #
# measure radius of curvature
def measure_curvature(image, leftx, rightx):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 25 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 550  # meters per pixel in x dimension

    ploty = np.linspace(0, 719, num=720)

    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    y_eval = np.max(ploty)

    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / (
        np.absolute(2 * left_fit_cr[0]))
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / (
        np.absolute(2 * right_fit_cr[0]))

    z = np.mean([left_curverad, right_curverad])

    cv2.putText(image, 'Radius of Curvature: {0:.3f}(m)'.format(z),
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


# ******************************************    Func Declaration Start      ****************************************** #
# calculate the vehicle position wrt center of the lane
def dst_from_center(image, left_fitx, right_fitx):
    # find the x coordinate corresponding to the lane center
    lane_center_x = (left_fitx[-1] + right_fitx[-1]) / 2

    xm_per_pix = 3.7 / 550

    # calculate the offset i.e deviation of the lane center coordinate from the image center
    # this will give the deviation of the vehicle from the center of the lane
    dist = (image.shape[1] / 2 - lane_center_x) * xm_per_pix

    # dist is the offset: if the deviation is positive - the vehicle is left from the center of the lane
    # if the distance is negative - the vehicle is right from the center of the lane
    if dist <= 0:
        pos = 'left'
    else:
        pos = 'right'

    cv2.putText(image, "Position: {0:.3f}(m) ".format(abs(dist)) + pos + " of center.",
                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


# ******************************************    Class Declaration Start     ****************************************** #
class LaneDetection(object):

    def __init__(self, mtx, dist):
        self.mtx, self.dist = mtx, dist
        self.left_fit, self.right_fit = [], []

    def lane_detection(self, image):
        # distortion correction
        undist_img = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

        # calculate gradient
        binary_img = binary_img_pipe(undist_img)

        # perspective transform
        warped_img = perspective_transform(binary_img, False)

        # detect lane pixels
        result, left_fitx, right_fitx, ploty = find_lane_pixels(warped_img, self.left_fit, self.right_fit)

        if len(self.left_fit) > 15:
            self.left_fit.pop(0)

        if len(self.right_fit) > 15:
            self.right_fit.pop(0)

        # reverse perspective
        check = perspective_transform(result, True)

        # wrap to the original image
        output = cv2.addWeighted(undist_img, 1, check, 1, 0)

        # measure radius of curvature and distance from the center
        measure_curvature(output, left_fitx, right_fitx)
        dst_from_center(output, left_fitx, right_fitx)

        return output

# ******************************************    Class Declaration End       ****************************************** #


# ******************************************        Main Program Start      ****************************************** #
def main():
    """
    The main of the program.
    Description of the Main Here.
    """
    # Now lets read the image paths from the file system.
    image_paths = glob('camera_cal/calibration*.jpg')

    # Now lets use the images pointed by the elements of the list `image_paths` to compute camera calibration parameters
    num_corners_x, num_corners_y = 9, 6
    ret, mtx, dist, rvecs, tvecs = calibrate_camera_distortion(image_paths, nx=num_corners_x, ny=num_corners_y)

    white_output = 'output_videos/challenge_video.mp4'
    clip1 = VideoFileClip("test_videos/challenge_video.mp4")
    ld = LaneDetection(mtx, dist)
    white_clip = clip1.fl_image(ld.lane_detection)
    white_clip.write_videofile(white_output, audio=False)


# ******************************************        Main Program End        ****************************************** #
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nProcess interrupted by user. Bye!')


"""
Author: Yash Bansod
"""