# Traffic Sign Recognition

## Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./plots/Train_data_before.png "Train data before"
[image2]: ./plots/Train_data_after.png "Train data after"
[image3]: ./plots/learning_curve.png "Learning curve"
[image4]: ./web_img/00.jpg "Traffic Sign 1"
[image5]: ./web_img/12.jpg "Traffic Sign 2"
[image6]: ./web_img/21.jpg "Traffic Sign 3"
[image7]: ./web_img/22.jpg "Traffic Sign 4"
[image8]: ./web_img/38.jpg "Traffic Sign 5"
[image10]: ./examples/dangerous_curve_left.jpg  "Traffic Sign 6"
examples\dengerous_curve_left.jpg
[image9]: ./examples/train_and_preprocessed_samples.jpg "Train_and_preprocessed"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.
The implementation and the project writeup can be found [project code](https://github.com/GosiaP/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) to my project code

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.
Below a statistic of the traffic signs data set calculated numpy library:

* The size of training set is 34799 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is 32x32 (RGB)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the data set.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across the different labels.

![Before][image1]

Below my observation:
* Looking at this chart it was very clear for me that some of types of traffic signs have more samples as others - even 10 times more. It means I have to extend the data set by creation of some faked data - as suggested in the lecture.
* Also the samples are sorted by class in data set what is a clear hint for me to that I have to shuffle them during of training.
* You can find some samples of signs in chapter "Exploratory visualization of the data set" in my [project](https://github.com/GosiaP/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) .

### Design and Test a Model Architecture

#### Question 1
_Describe how you preprocessed the image data. What techniques were chosen and why did you choose them_

Preprocessing of the image is done in following steps:
* conversion of the image to YUV color space as suggested in the [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). I didn't even tried to convert the image to grayscale. I learned in the previous project that this transformation doesn't bring good effect in image processing. Finally I decided to use only Y channel as in some preliminary experiments full color images seem to confuse the classifier (as also reported in the paper).
* intensity normalization to achieve mean of image around 0.
* improve contrast of the image using of adaptive histogram equalization [CLAHE](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization)

Here a sample of final image preprocessing (they don't matched 1:1 as they are created randomly):

![Preproc][image9]


#### Question 2
_Describe how you set up the training, validation and testing data for your model. Optional: If you generated additional data, how did you generate the data? Why did you generate the data? What are the differences in the new dataset (with generated data) from the original data set?_

As the samples of signs are not distributed in sign classes uniformly I decided to add some faked images to achieve more uniformly number of sign per class.
I achieved it by calculation of number of operations required for particular image class:
* num_images - number if images in the calss the image belongs to
* num_to_generate = max_uniques_count - num_images, where max_uniques_count is the largest number of images in one class
* num_operations = num_to_generate // num_images

The _num_operations_ was the number of faked images I had to create.
I applied following image transformation (choosed randomly) to create faked images:
* rotation left/right about some angle
* scaling up/down
* horizontal motion blur of the image

The result of the transformation are shown in my project in chapter in chapter "Exploratory visualization of the data set".
Distribution of images per classes was not perfect but clearly better. See bulk chart below.

![After][image2]

#### Question 3
_Describe what your final model architecture looks like._

As suggested in the lecture I used LeNet-5 architecture. I added a dropout to prevent over fitting.


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 (Y channel image)  			    	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 14x14x6    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16.			        |
| Fully Connected       | Input 400, output 120.                        |
| RELU					|												|
| Droput	      	    | keep probability 0.7				            |
| Fully connected		| input 120, output  84.       			        |
| RELU					|												|
| Droput	      	    | keep probability 0.7				            |
| Softmax				|          									    |

#### Question 4
_How did you train your model?_

To train the model, I used:
* Adam Optimizer
* learning rate = 0.001
* dropout rate of 0.3
* batch size of 128
* number of epochs = 20

#### Question 5
_Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem._

To train the model, I used a LeNet-5 architecture as suggested in the lecture. First I didn't used a droput and I started with only 10 epochs. I realized that the model tended to overfit. So added two dropout with rate 0.3 and increase the numbers of epochs to 20. As I installed Tensor Flow with GPU and I have pretty good GPU, the training of data set took me 15 minutes.

Training curves can be seen below.

![Learning][image3]

My final model results were:
* training set accuracy of 99%
* validation set accuracy of 97%
* test set accuracy of 95%

### Test a Model on New Images

#### Question 1
_Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify._

Here are five German traffic signs that I found on the web:

| Sign class            |     Web Image           |
|:---------------------:|:-----------------------:|
| Speed limit (20km/h)  | ![alt text][image4]     |
| Priority road         | ![alt text][image5]	  |
| Double curve			| ![alt text][image6]     |
| Bumpy road      		| ![alt text][image7]	  |
| Keep right			| ![alt text][image8]     |

I converted these images to the size 32x32 before I started their processing.
I don't have any feeling which the images from the set I choose can be difficult to train. So, I simply started prediction and analyzed the result of it.

#### Question 2
_Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set._

Here are the results of the prediction:

| Sign class            |     Web Image           |Sign class            |     Predicted Web Image  | Probability |
|:---------------------:|:-----------------------:|:---------------------:|:-----------------------:|:-----------:|
| Speed limit (20km/h)  | ![alt text][image4]     |Speed limit (20km/h)  | ![alt text][image4]     | 95%          |
| Priority road         | ![alt text][image5]	  |Speed limit (20km/h)  | ![alt text][image5]     |100%          |
| Double curve			| ![alt text][image6]     |Dangerous curve to the left | ![alt text][image10] |98%        |
| Bumpy road      		| ![alt text][image7]	  |Speed limit (20km/h)  | ![alt text][image7]     |100%          |
| Keep right			| ![alt text][image8]     |Speed limit (20km/h)  | ![alt text][image8]     |100%          |

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80% which is compareable to the accuracy of test set.
Loading of the test web images is provided in cell 34, prediction in cell 35. Calculation of Top 5 Softmax Probabilities is done in cell 36.

#### Question 3
_Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability._

Here the results of prediction.

1.Probabilities for image 'Speed limit (20km/h)' (0):

|     Prediction	            	| Probability        |
|:---------------------------------:|:-----------------: |
| Speed limit (20km/h)			    | 95%         	     |
| Speed limit (60km/h)				| 4%     		     |
| Speed limit (30km/h)			   	| 0%			     |
| Speed limit (80km/h)				| 0%	      	     |
| End of speed limit (80km/h)     	| 0%			     |

2.Probabilities for image 'Priority road':

|     Prediction	            	| Probability        |
|:---------------------------------:|:-----------------: |
| Priority road         		    | 100%         	     |
| No passing        				| 0%     		     |
| Roundabout mandatory   		   	| 0%			     |
| No vehicles               		| 0%	      	     |
| Ahead only                    	| 0%			     |

3.Probabilities for image 'Double curve':

|     Prediction	            	| Probability        |
|:---------------------------------:|:-----------------: |
| Dangerous curve to the left       | 98%         	     |
| Slippery road             		| 2%     		     |
| Double curve              	   	| 0%			     |
| Bicycles crossing             	| 0%	      	     |
| Road narrows on the right     	| 0%			     |

4.Probabilities for image 'Bumpy road':

|     Prediction	            	| Probability        |
|:---------------------------------:|:-----------------: |
| Bumpy road		                | 100%         	     |
| Bicycles crossing              	| 40%     		     |
| Road work                     	| 0%			     |
| Turn left ahead           		| 0%	      	     |
| Wild animals crossing         	| 0%			     |

5.Probabilities for image 'Keep right':

|     Prediction	            	| Probability        |
|:---------------------------------:|:-----------------: |
| Keep right                	    | 100%         	     |
| Speed limit (80km/h)          	| 0%     		     |
| Wild animals crossing          	| 0%			     |
| Speed limit (50km/h)      		| 0%	      	     |
| Speed limit (30km/h)          	| 0%			     |

Prediction of images 1, 2, 4, 5 was correct. Prediction of image 3 - Double curve - was not correct.  The Double curve was predicted as Dangerous curve to the left - it is interesting but somehow explainable. Both of traffic signs have same form, similar color palette and the symbol of dangerous curve left is part of symbol for double curve.


