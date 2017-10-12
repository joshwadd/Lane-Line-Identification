# Lane Line Identification
***
This project uses traditional computer vision techniques implemented in Python and OpenCV to identify lane lines in the road. The image data is collected from a front facing video camera mounted on a car driving in freeway traffic. This project was done as part of the Udacity self driving car nanodegree program.


[![png](image_output/output1.png)](https://youtu.be/gcRVc0u5Qr0)
[Video Link](https://youtu.be/gcRVc0u5Qr0)

The image processing pipeline used involves the following techniques

1. Guassian blur to remove high frequency information.
2. Canny edge detection.
3. Region of interest mask.
4. Hough transform line detection.


***
## Dataset

Detecting lane lines in the road is an important feature extraction step used for subsequent tasks in an autonomous vehicle such as pose estimation, navigation and guidance. The vision systems on an autonomous vehicle is a primary candidate to detect such road markings. This is in principle due to the development of road markings being designed to be easily detectable and identifiable for the human visual system. However detecting road lane lines can still be a difficult task in computer vision due to the vast range of varying road and weather conditions that can be encountered when driving, and the varying design and colors of the lane lines themselves. 

For this task I develop a system that is capable of detecting lane lines in video data taken from a front facing mount on a car. As the system is required to be fast and running in real time for a final hardware implementation, the pipeline will be designed to work on individual frames and take no account of any temporal or optical flow information present in the video data. Examples of the types of image we want to be able to detect lane lines on are shown below


