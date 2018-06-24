import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
import sys
import os

plt.ion()
# prepare object points
nx = 9
ny = 5

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Make a list of calibration images
fname = './camera_cal/calibration1.jpg'
img = cv2.imread(fname)

plt.imshow(img)
plt.show()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# If found, draw corners
if ret == True:
    # Draw and display the corners
    plt.figure()
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    imgpoints = [corners]
    # print(corners)
    plt.imshow(img)
    plt.show()


def undisort_img(img):
    if os.path.exists("./camera_cal/wide_dist_pickle.p") is False:

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('./camera_cal/calibration*.jpg')

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            # print(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                # print(objp)
                imgpoints.append(corners)
                # print(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                # write_name = 'corners_found'+str(idx)+'.jpg'
                # cv2.imwrite(write_name, img)
                cv2.imshow('img', img)
                cv2.waitKey(500)

        cv2.destroyAllWindows()

        # Test undistortion on an image
        # img = cv2.imread('./camera_cal/calibration1.jpg')
        img_size = (img.shape[1], img.shape[0])

        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        dst = cv2.undistort(img, mtx, dist, None, mtx)
        # cv2.imwrite('./camera_cal/test_undist.jpg', dst)

        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump(dist_pickle, open("./camera_cal/wide_dist_pickle.p", "wb"))

        return dst
    else:
        with open("./camera_cal/wide_dist_pickle.p", 'rb') as f:
            dist_pickle = pickle.load(f)
        mtx = dist_pickle['mtx']
        dist = dist_pickle['dist']

        dst = cv2.undistort(img, mtx, dist, None, mtx)

        return dst


def warper(img, update=False):
    width = img.shape[0]
    height = img.shape[1]
    if os.path.exists("./camera_cal/perspective_cor.p") is False or update is True:
        print('Updated the perspective transfer')
        src = np.float32([[220, height], [580, height - 260],
                          [width - 575, height - 260], [width - 170, height]])
        dst = np.float32([[440, height], [440, 0],
                          [width - 330, 0], [width - 330, height]])
        dist_pickle = {}
        dist_pickle["src"] = src
        dist_pickle["dst"] = dst
        pickle.dump(dist_pickle, open("./camera_cal/perspective_cor.p", "wb"))
    else:
        with open("./camera_cal/perspective_cor.p", 'rb') as f:
            dist_pickle = pickle.load(f)
            src = dist_pickle['src']
            dst = dist_pickle['dst']

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped, src, dst, Minv


def draw_poly(img, poly):
    img_draw = np.copy(img)
    cv2.polylines(img_draw, np.int32([poly]), 1, (255, 0, 255), thickness=2)
    return img_draw


def select_white_yellow(image):
    # To convert the color image to HLS color so to filter white and yellow colored lanes
    cov_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # white color mask
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(cov_image, lower, upper)
    # yellow color mask
    lower = np.uint8([10, 0, 100])
    upper = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(cov_image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    output = cv2.bitwise_and(image, image, mask=mask)
    return output


def pipeline(img, s_thresh=(100, 255), mag_thresh=(30, 100), grad_thresh=(0.7, 1.3), sobel_kernel=5):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    img = undisort_img(img)
    img = select_white_yellow(img)

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    noneblur_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(noneblur_gray, (5, 5), 0)
    # l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)  # Take the derivative in x
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    abs_sobely = np.absolute(sobely)
    abs_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    gradient = np.arctan2(abs_sobely, abs_sobelx)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    scaled_sobely = np.uint8(255 * abs_sobely / np.max(abs_sobely))

    # Threshold x gradient
    magbinary = np.zeros_like(scaled_sobel)
    magbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    gradbinary = np.zeros_like(gradient)
    gradbinary[(gradient >= grad_thresh[0]) & (gradient <= grad_thresh[1])] = 1
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(s_binary == 1) | (gradbinary == 1) & (magbinary == 1)] = 1
    color_binary = np.dstack((gradbinary, magbinary, s_binary)) * 255

    return combined_binary, color_binary


# window settings
window_width = 60
window_height = 80  # Break image into 9 vertical layers since image height is 720
margin = 150  # How much to slide left and right for searching


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def find_window_centroids(image, window_width, window_height, margin, fit=None):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template
    if fit == None:
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(1 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
        r_sum = np.sum(image[int(1 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)

        # Add what we found for the first layer
        window_centroids.append((l_center, r_center))
        pre_l_center = l_center
        pre_r_center = r_center
    else:
        left_fit_prev, right_fit_prev = fit
        left_fitx = left_fit_prev[0] * (image.shape[0] - window_height) ** 2 + left_fit_prev[1] * (
            image.shape[0] - window_height) + left_fit_prev[2]
        right_fitx = right_fit_prev[0] * (image.shape[0] - window_height) ** 2 + right_fit_prev[1] * (
            image.shape[0] - window_height) + right_fit_prev[2]
        image_masked = np.copy(image)
        image_masked[:, :np.max([int(left_fitx) - margin, 0])] = 0
        image_masked[:, np.min([int(left_fitx) + margin, int(image.shape[1] / 2)]):int(image.shape[1] / 2)] = 0
        image_masked[:, int(image.shape[1] / 2):np.min([int(right_fitx) - margin, image.shape[1]])] = 0
        image_masked[:, np.max([int(right_fitx) + margin, int(image.shape[1] / 2)]):image.shape[1]] = 0
        l_sum = np.sum(image_masked[int(1 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
        r_sum = np.sum(image_masked[int(1 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)
        l_center = l_center * 0.5 + left_fitx * 0.5
        r_center = r_center * 0.5 + right_fitx * 0.5
        # Add what we found for the first layer
        window_centroids.append((l_center, r_center))
        pre_l_center = l_center
        pre_r_center = r_center
    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(image.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset

        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # print('r--> %s, l--> %s' % (
        # np.max((conv_signal[l_min_index:l_max_index])), np.max((conv_signal[r_min_index:r_max_index]))))
        # print(np.max((conv_signal[r_min_index:r_max_index])))
        if (np.max((conv_signal[l_min_index:l_max_index]))) < 500:
            l_center = pre_l_center
            if np.max((conv_signal[r_min_index:r_max_index])) > 500:

                l_center = pre_l_center + r_center - pre_r_center
            else:
                l_center = pre_l_center
                r_center = pre_r_center
        else:
            if np.max((conv_signal[r_min_index:r_max_index])) < 500:
                r_center = pre_r_center + l_center - pre_l_center

        # #Let's check the change direction of the both right and left lanes, they should be the same
        # if abs(r_center - l_center) > image.shape[1]/2 or abs(r_center - l_center) < image.shape[1]/3:
        #     # Check which lane is more reliable
        #     if np.max(conv_signal[r_min_index:r_max_index]) > np.max(conv_signal[l_min_index:l_max_index]):
        #         l_center = pre_l_center + (r_center - pre_r_center)
        #     else:
        #         r_center = pre_r_center + (l_center - pre_l_center)

        if abs(l_center - pre_l_center) > 1.5 * window_width:
            l_center = pre_l_center
        else:
            pre_l_center = l_center

        if abs(r_center - pre_r_center) > 1.5 * window_width:
            r_center = pre_r_center
        else:
            pre_r_center = r_center
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids


def poly_fit(img, window_centroids, scalex=1, scaley=1):
    lefty = []
    righty = []
    leftx = []
    rightx = []
    for level in range(0, len(window_centroids)):
        # Window_mask is a function to draw window areas
        lefty.append(img.shape[0] - (level + 0.5) * window_height * scaley)
        righty.append(img.shape[0] - (level + 0.5) * window_height * scaley)
        leftx.append(window_centroids[level][0] * scalex)
        rightx.append(window_centroids[level][1] * scalex)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit


if os.path.exists("./camera_cal/wide_dist_pickle.p") is False:

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        # print(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            # print(objp)
            imgpoints.append(corners)
            # print(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            # write_name = 'corners_found'+str(idx)+'.jpg'
            # cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Test undistortion on an image
    img = cv2.imread('./camera_cal/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('./camera_cal/test_undist.jpg', dst)

    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("./camera_cal/wide_dist_pickle.p", "wb"))

    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.show()

else:
    images = glob.glob('./test_images/*.jpg')
    print(images)
    with open("./camera_cal/wide_dist_pickle.p", 'rb') as f:
        dist_pickle = pickle.load(f)
    mtx = dist_pickle['mtx']
    dist = dist_pickle['dist']
    undist_images = []
    warpped_images = []
    threshold_images = []
    filenames = []
    for filename in images:
        img = cv2.imread(filename)
        path, file_output = os.path.split(filename)
        file_output = os.path.splitext(file_output)[0]
        img_size = (img.shape[1], img.shape[0])
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        undist_images.append(dst)
        writeName = './output_images/%s_undist.jpg' % file_output
        filenames.append(file_output)
        print(writeName)
        cv2.imwrite(writeName, dst)

    for img, file_output in zip(undist_images, filenames):
        thred_image, color_image = pipeline(img)
        threshold_images.append(thred_image)
        writeName = './output_images/%s_thred.jpg' % file_output
        thred_color = np.dstack((thred_image, thred_image, thred_image)) * 255
        cv2.imwrite(writeName, thred_color)
        writeName = './output_images/%s_colored.jpg' % file_output
        cv2.imwrite(writeName, color_image)

        warp_image, src, dst, Minv = warper(img)
        # cv2.polylines(warp_image,)
        warpped_images.append(warp_image)
        warp_draw = draw_poly(warp_image, dst)
        img_draw = draw_poly(img, src)
        writeName = './output_images/%s_warp.jpg' % file_output
        cv2.imwrite(writeName, warp_draw)
        writeName = './output_images/%s_warp_org.jpg' % file_output
        cv2.imwrite(writeName, img_draw)

    for img, file_output in zip(threshold_images, filenames):
        warp_image, src, dst, Minv = warper(img)
        print(warp_image.shape)
        warpped_images.append(warp_image)
        warp_final = np.dstack((warp_image, warp_image, warp_image)) * 255
        writeName = './output_images/%s_warp_final.jpg' % file_output
        cv2.imwrite(writeName, warp_final)

        warp_image = np.uint8(warp_image)
        warp_image = cv2.GaussianBlur(warp_image, (5, 5), 0)
        print(file_output)
        window_centroids = find_window_centroids(warp_image, window_width, window_height, margin)

        # If we found any window centers
        if len(window_centroids) > 0:

            # Points used to draw all the left and right windows
            l_points = np.zeros_like(warp_image)
            r_points = np.zeros_like(warp_image)

            # Go through each level and draw the windows
            for level in range(0, len(window_centroids)):
                # Window_mask is a function to draw window areas
                l_mask = window_mask(window_width, window_height, warp_image, window_centroids[level][0], level)
                r_mask = window_mask(window_width, window_height, warp_image, window_centroids[level][1], level)
                # Add graphic points from window mask here to total pixels found
                l_points[(l_points == 255) | ((l_mask == 1))] = 255
                r_points[(r_points == 255) | ((r_mask == 1))] = 255

            # Draw the results
            template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
            zero_channel = np.zeros_like(template)  # create a zero color channel
            template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
            warpage = np.dstack(
                (warp_image, warp_image, warp_image)) * 255  # making the original road pixels 3 color channels
            output = cv2.addWeighted(warpage, 1, template, 0.5,
                                     0.0)  # overlay the orignal road image with window results

        # If no window centers found, just display orginal road image
        else:
            output = np.array(cv2.merge((warp_image, warp_image, warp_image)), np.uint8)

        writeName = './output_images/%s_lane_find.jpg' % file_output
        cv2.imwrite(writeName, output)

        left_fit, right_fit = poly_fit(warp_image, window_centroids)
        print(left_fit)
        print(right_fit)
        ploty = np.linspace(0, warp_image.shape[0] - 1, warp_image.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        plt.figure()
        plt.imshow(output)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        writeName = './output_images/%s_polyfit.jpg' % file_output
        plt.savefig(writeName)

        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        left_fit_cr, right_fit_cr = poly_fit(warp_image, window_centroids, scalex=xm_per_pix, scaley=ym_per_pix)
        y_eval = warp_image.shape[0] / 2
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                             (1 + (
                                 2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[
                                     1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        print(left_curverad, 'm', right_curverad, 'm')

        warp_zero = np.zeros_like(warp_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
        # Combine the result with the original image
        print(newwarp.shape)

        img = cv2.imread('./test_images/%s.jpg' % file_output)
        print(img.shape)
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        writeName = './output_images/%s_lane_final.jpg' % file_output
        cv2.imwrite(writeName, result)


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None


def poly_fit_eval(poly_fit, ploty):
    plotx = poly_fit[0] * ploty ** 2 + poly_fit[1] * ploty + poly_fit[2]
    return plotx


def video_process(img):
    # print(img.shape)

    global left_fit_prev
    global right_fit_prev
    global col_R_prev
    global col_L_prev
    global init_set
    global mask_poly_L
    global mask_poly_R

    img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # The image for processing is in BGR color
    dst = cv2.undistort(img_BGR, mtx, dist, None, mtx)
    thred_image, color_image = pipeline(dst)
    warp_image, src, dst, Minv = warper(thred_image)

    warp_image = np.uint8(warp_image)
    warp_image = cv2.GaussianBlur(warp_image, (5, 5), 0)
    # print(file_output)
    if init_set == 0:
        window_centroids = find_window_centroids(warp_image, window_width, window_height, margin)
    else:
        window_centroids = find_window_centroids(warp_image, window_width, window_height, margin,
                                                 fit=(left_fit_prev, right_fit_prev))
    # If we found any window centers
    # if len(window_centroids) > 0:
    #
    #     # Points used to draw all the left and right windows
    #     l_points = np.zeros_like(warp_image)
    #     r_points = np.zeros_like(warp_image)
    #
    #     # Go through each level and draw the windows
    #     for level in range(0, len(window_centroids)):
    #         # Window_mask is a function to draw window areas
    #         l_mask = window_mask(window_width, window_height, warp_image, window_centroids[level][0], level)
    #         r_mask = window_mask(window_width, window_height, warp_image, window_centroids[level][1], level)
    #         # Add graphic points from window mask here to total pixels found
    #         l_points[(l_points == 255) | ((l_mask == 1))] = 255
    #         r_points[(r_points == 255) | ((r_mask == 1))] = 255
    #
    #     # Draw the results
    #     template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
    #     zero_channel = np.zeros_like(template)  # create a zero color channel
    #     template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
    #     warpage = np.dstack(
    #         (warp_image, warp_image, warp_image)) * 255  # making the original road pixels 3 color channels
    #     output = cv2.addWeighted(warpage, 1, template, 0.5,
    #                              0.0)  # overlay the orignal road image with window results
    #
    # # If no window centers found, just display orginal road image
    # else:
    #     output = np.array(cv2.merge((warp_image, warp_image, warp_image)), np.uint8)
    # return output
    # writeName = './output_images/%s_lane_find.jpg' % file_output
    # cv2.imwrite(writeName, output)

    left_fit, right_fit = poly_fit(warp_image, window_centroids)
    left_fit_prev = left_fit
    right_fit_prev = right_fit
    init_set = 1
    # print(left_fit)
    # print(right_fit)
    ploty = np.linspace(0, warp_image.shape[0] - 1, warp_image.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # plt.figure()
    # plt.imshow(output)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # writeName = './output_images/%s_polyfit.jpg' % file_output
    # plt.savefig(writeName)

    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    left_fit_cr, right_fit_cr = poly_fit(warp_image, window_centroids, scalex=xm_per_pix, scaley=ym_per_pix)
    y_eval = warp_image.shape[0] / 2
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = (
                         (1 + (
                             2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')

    warp_zero = np.zeros_like(warp_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    # print(newwarp.shape)
    ## Compute intercepts
    img_size = newwarp.shape
    left_bot = poly_fit_eval(left_fit, img_size[0])
    right_bot = poly_fit_eval(right_fit, img_size[0])
    val_center = (left_bot + right_bot) / 2.0
    ## Compute lane offset
    dist_offset = val_center - img_size[1] / 2
    dist_offset = np.round(dist_offset * 3.7 / 700, 2)
    str_offset = 'Lane deviation: ' + str(dist_offset) + ' m.'

    # img = cv2.imread('./test_images/%s.jpg' % file_output)
    # print(img.shape)
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    # writeName = './output_images/%s_lane_final.jpg' % file_output
    # cv2.imwrite(writeName, result)

    str_curv = 'Curvature: Right = ' + str(np.round(right_curverad, 2)) + 'm, Left = ' + str(
        np.round(left_curverad, 2)) + 'm'
    # Change color if distance is more than 30 cm
    font = cv2.FONT_HERSHEY_COMPLEX
    if dist_offset < 30:
        cv2.putText(result, str_curv, (100, 60), font, 1, (0, 255, 0), 2)
        cv2.putText(result, str_offset, (100, 90), font, 1, (0, 255, 0), 2)
    else:
        cv2.putText(result, str_curv, (100, 60), font, 1, (255, 0, 0), 2)
        cv2.putText(result, str_offset, (100, 90), font, 1, (255, 0, 0), 2)
    return result


show_stat = 0

init_set = 0

from moviepy.editor import VideoFileClip
from IPython.display import HTML
import imageio

# imageio.plugins.ffmpeg.download()

video1 = 'project_video.mp4'
video2 = 'challenge_video.mp4'
video3 = 'harder_challenge_video.mp4'

videos_out_folder1 = './output_video/project_video_out'
video_output = '%s/project_video_out.mp4' % videos_out_folder1
if not os.path.exists(videos_out_folder1):
    os.makedirs(videos_out_folder1)
clip1 = VideoFileClip(video1)
clipout = clip1.fl_image(video_process)  # NOTE: this function expects color images!!
clipout.write_videofile(video_output, audio=False)

init_set = 0
videos_out_folder2 = './output_video/challenge_video_out'
video_output = '%s/challenge_video_out.mp4' % videos_out_folder2
if not os.path.exists(videos_out_folder2):
    os.makedirs(videos_out_folder2)
clip1 = VideoFileClip(video2)
clipout = clip1.fl_image(video_process)  # NOTE: this function expects color images!!
clipout.subclip(0, 5).write_videofile(video_output, audio=False)

init_set = 0
videos_out_folder3 = './output_video/harder_challenge_video_out'
video_output = '%s/harder_challenge_video_out.mp4' % videos_out_folder3
if not os.path.exists(videos_out_folder3):
    os.makedirs(videos_out_folder3)
clip1 = VideoFileClip(video3)
clipout = clip1.fl_image(video_process)  # NOTE: this function expects color images!!
clipout.subclip(0, 5).write_videofile(video_output, audio=False)
