## Advanced Lane Finding Project Writeup

### This project is to apply the image processing techniques to detect the lane's information in real-time

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Distorted"
[image2]: ./camera_cal/test_undist.jpg "Undistorted"
[image3]: ./test_images/straight_lines1.jpg "Tested Straight Line"
[image4]: ./output_images/straight_lines1_undist.jpg "Undistorted Straight Line Image"
[image5]: ./output_images/straight_lines1_colored.jpg "Lane Detection using three thresholds"
[image6]: ./output_images/straight_lines1_thred.jpg "Binary after the threshoding"
[image7]: ./output_images/straight_lines1_warp_org.jpg "Before Perspetive Transformation"
[image8]: ./output_images/straight_lines1_warp.jpg "After Perspetive Transformation"
[image9]: ./output_images/straight_lines1_lane_find.jpg "Straight Lane Detection"
[image10]: ./output_images/test6_lane_find.jpg "Straight Lane Detection"
[image11]: ./output_images/ScreenShot_OutputVideo.png "Straight Lane Detection"


[video1]: ./output_video/project_video_out/project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### This document is the Writeup / README that includes all the rubric points and how I addressed the points one by one. 


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is in lines # 318 through # 357 of the file called `CameraCalibration.py`). Also, the function 'undistor_img()' are used in the video's image process pipeline. First to detect if the camera has been calibrated or not by checking the calibrated file exsists or not. If not exsists, then, to start the calibration. If the calibration file exsists, then, load the calibration data, and start correction on the input image.

First, I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates in the format of list, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

The calibration images I used are in the folder camera_cal. And the function I used to calibrate the camera are cv2.findChessboardCorners(). The grid size of the chessboard are nx = 9 and ny = 5 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
Distorted Image
![alt text][image1] 
Corrected Undistorted Image
![alt text][image2]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

The following are the original and corrected undistorted straight line images. As seen in the iamges, the distorted corner parts of the original image have been corrected in the undistorted image. And therefore, the image is ready for the following lane extraction and perspective transformation.

Original Image:
![alt text][image3]
Corrected Undistorted Image:
![alt text][image4]  

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color, gradient magnitude, and gradient direction thresholds to generate a binary image (in the function 'pipline() thresholding steps at lines # 154 through # 191 in 'CameraCalibration.py'). Here's an example of my output for this step.

First, based on the experience in the first project, I got that it is useful to apply the color filter to select the white and yellow color from the image in HLS color domain. The white and yellow filter is from #137 to #151 in the function select_white_yellow() function.

After that, I applied the threshold on the S-channel in the image, combined with gradient and direction threshold. The following image shows the filtered image in those three channels.
![alt text][image5]

As seen in the image, the red channel (color filter channel) picking out the lane through the color filter and threshold. The Green and Blue channels (gradient and direction channels) pick out the shape of the lane. 

The following is the binary output of the lane detection image:
![alt text][image6]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 105 through 128 in the file `CameraCalibration.py` The `warper()` function takes as inputs an image (`img`). Inside the warper function, I first check if the perspective transfer matrixs 'src' and 'dst' exsist. If so, will do the perspective transformation, if not, will use the default value to do the transformation.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[220, height], [580, height - 260],
                  [width - 575, height - 260], [width - 170, height]])
dst = np.float32([[440, height], [440, 0],
                  [width - 330, 0], [width - 330, height]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 220, 720      | 440, 0        | 
| 1110, 720     | 440, 720      |
| 580, 460      | 950, 720      |
| 705, 460      | 950, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
Before Perspective Transformation
![alt text][image7]
After Perspective Transformation
![alt text][image8]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The function to find the right/left lane's center position is find_window_centroids() from line 209 to line 300 in 'CameraCalibration.py'.

To identiify lane-line pixels, the first step is to determine a good start point. It is because the starting point would affect the following lane pixel detections very much. Therefore, I would determine the start point in the first center point based on two situations:
1. If there is no previous frame, I would use 3/4 of the image, and integrate along the y-axis, in order to have stable histogram for the x-axis.
2. If there is a previous frame, I would use the previous frame's fitted curve with margin as the mask to find the starting point.

Once the start point (the center of right and left lane at the bottom of the image) has been decided. The following centers will be determined by using the convolution. 
Also, I would check if the detected right/left centers makes any sense. 
1. If the maximum convolution value is less than 500, I would check the convolution value of the other lane. If the other lane's convolution value is higher than 500, I will trust the other lane's change on the lane's center, and change the current lane's center based on the other lane. The logic behind this is two lane's should be parallel in the image. 
2. Also, if the center changes more than 1.5 times of sliding window's width, I would just ignore the change of the lane centers. 

The following are results of the lane detection on straight lane and curved lane
Straigth Lane
![alt text][image9]
Curved Lane
![alt text][image10]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #303 through #317 in my code in `CameraCalibration.py`. Once the centers of right/left lanes are determined along the y-axis. I used np.polyfit() function to fit the lane with 2nd order polynomial curve, in order to calculate the curvature of the lane, and the position of the vehicle respect to center.
The scaling factor I used are:
```python
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
```    
The formula to calculate curvature is:
```python
left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute( 2 * left_fit_cr[0])
right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
```
To calculate the position of car respect to center, assuming the camera is positioned at the center of the car, the position of the car respect to center is the same as the offset between the image's center and the center of the right/left lanes. I calculated the curvature and car position offset from line # 625 to line # 657 in 'CameraCalibration.py'

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #634 through #675 in my code in `CameraCalibration.py` in the function 'video_process()'.  I used the inverted warping matrix to convert the perspective and ditortion back to the original image, and therefore casting the lane detetion info and image onto the original video image. Here is an example of my result on video output after the lane detection and plotting back down to the road:

![alt text][image11]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [video1] [link to my video result](./output_video/project_video_out/project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. The first step of the project is well defined camera calibration and undistortion. Once the calibration image are in good quality and undistorttion transformation matrix is well defined. We shouldn't see issues in this step. 
2. The perspective transformation is also straight forward, as long as the camera's position is not moving, or the angle of the camera is not tilting during the driving, the perspetive transformation should be the same. 
3. The critical steps in this project are: Lane detection by the thresholding in different domains, such as color domain, shape domain, and other possible features, like the lane to lane distance, curvature range, and so on. The land detection proecess need really tough robustness testing since the road conditions change a lot. On the other side, as we consider more and more situation for the lane detection, the complexity of the program would increase exponentially. The approach I used in this project are based on the thresholding on color, s-channel, gradient, and direction for the lane detection. It provided reasonable well detection on the regular road condition. But I still have issue on the chanllenge videos. I will look into the reasons for the issue.
 4. For the lane center detections, the starting point location is very critical, and it will be tricky on the dashed lane, since there is no binary pixels for the lane center detections. Therefore, both lanes need to be reference to each other. And if the detection is not trustable (convolution value is lower than 500 in my code), it is better to use the previously detected center than use the unreliable center detections.
 5. In order to make the lane detection more robust, I need to look into the failure modes of the chanllnge video causeing the failure on the lane detections and lane center calcualtion. I believe by add more features in the lane finding, as well as adding the flexbility of the lane's width adjustment, would be able to make the chanllenge video work. 
