# **Finding Lane Lines on the Road** 

First project of the Term 1 of Udacity's Self Driving Car Nanodegree program.

  

**Google Drive:** https://drive.google.com/drive/folders/13bZe7EeKvSxU1CN9yvI5eXsU6RyCN5GT?usp=sharing

Please download the following files from the above mentioned google drive link:

- **test_images:** Contains the test images for the project

- **test_videos:** Contains the test videos for the project  

- **test_images_output:** Contains the output of test images for the project

- **test_videos_output:** Contains the output of test videos for the project  

   

**Repository:** https://github.com/YashBansod/udacity-self-driving-car/Lane-Lines/



- [**_Overview_**](#overview)
- [**_The Pipeline_**](#the-pipeline)
- [**_Present Limitations and Possible Improvements_**](#present-limitations-and-possible-improvements)
- [**_Run Instructions_**](#run-instructions)
- [**_Project Development Done Using_**](#project-development-done-using)

## Overview
This Github repository was created for sharing the application implemented for the First project of the Term 1 of 
[Udacity's Self Driving Car Nanodegree program](https://in.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)

The original project repository containing the template code used in this project is 
[udacity/CarND-LaneLines-P1](https://github.com/udacity/CarND-LaneLines-P1)

**Sample results:**
<div class="table-wrapper">
<table class="alt">
    <tr>
      <th>
        <p align="center">
             <div><span class="image fit"><img src="./images/solidWhiteCurve.jpg"></span></div>
             <br>solidWhiteCurve
        </p>
      </th>
      <th>
        <p align="center">
             <div><span class="image fit"><img src="./images/solidYellowCurve.jpg"></span></div>
             <br>solidYellowCurve
        </p>
      </th>
      <th>
        <p align="center">
             <div><span class="image fit"><img src="./images/whiteCarLaneSwitch.jpg"></span></div>
             <br>whiteCarLaneSwitch
        </p>
      </th>
    <tr>
</table>
</div>



More sample results of the project can be visualized at:  
- **test_images_output ** 

- **test_videos_output**
  
  

## The Pipeline

The following processing pipeline was used in this implementation:  
1. **Image Input Verification**  
This block checks if the value of pixels in the image is in 0-255 range. If not, it modifies the values to that range.
This is optional block in the pipeline. User can choose to remove this block if it is not required in the pipeline.  

2. **Region of Interest (ROI) Selection**  
This block selects a trapezoidal region in fron of the car. The lane detection only happens in this region.  

3. **Selection of White and Yellow Pixels**  
The indices of white and yellow pixels in the ROI image is saved.  

4. **Creating a Mask from the Color Selection**  
The color selected image is dialated and a Mask is created from it. This mask marks the postions in the image where the pixels are either yellow or white.  

5. **Running Canny Edge Detection on ROI Selected Output**  
Canny Edge Detection is done over the ROI Selected Output (Output from step 2).  

6. **Compute Relevant Edges by applying Color Selection Mask**  
Relevant Edges are extracted from the Canny Edge Detection Output by processing the Color Selection Mask (Output of Step 4) over it.  

7. **Hough Transform is applied over the Relevant Edges**  
Hough Transform is applied over the Relevant Edges (Output of Step 6). Hought Transfrom reverts back a list containing the extreme coordinates of the various lines detected in the image. The wrapper function written in the program also computes unique lines from this output and draws them onto a blank image.  

8. **Detected Lanes are Overlayed on the Original Input**  
The lane lines detected (output of Step 7) are overlayed on the original image for visualization.  


## Present Limitations and Possible Improvements
- The current lane line modelling assumes the lanes as straight lines. This is obviously not correct. A more accurate approach would be to use clothoids / splines for modelling the lanes.  
  
- This lane detection can be unreliable if there are extreme fluctuations in the environment lighting, shadows etc. This is primaraly because of the hardcoded parameters in the algorithm. Adaptive selection of parameters based on the ambience might help solve this issue. But, personally I believe that, a properly designed Machine Learning based solution for Lane Detection will easily beat all other alternatives.  
  
- Although there are various blocks in the pipeline that operate on the same input, no effort was made in writing a multi-processed / multi-threaded scheduling of the pipeline.  

- Many operations like ROI selection can be made much faster by using Look Up Tables (LUTs) rather than computing the mask everytime.

_and the list goes on ..._

## Run Instructions
- Open the Terminal
- Clone this repository onto your local system.
- Enable a python interpreter in your environment (python version >=3.5) (It is recommended to use a python virtual interpreter).
- Install the requirements of this project from the [requirements.txt](./requirements.txt) by typing `pip install -r requirements.txt`.
- Change your present working directory to the inside of the cloned repository.
- Enable the jupyter notebook environment by typing `jupyter notebook` or `jupyter-notebook`.
- Open the [P1.ipynb](./P1.ipynb) file in Jupyter Notebook and run it like any other jupyter notebook.


## Project Development Done Using
Ubuntu 16.04  
PyCharm 2018.2  
Python 3.6  
