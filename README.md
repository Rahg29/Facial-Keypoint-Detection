[//]: # (Image References)

[image1]: ./images/key_pts_example.png "Facial Keypoint Detection"

# Facial Keypoint Detection

## Project Overview

In this project, you’ll combine your knowledge of computer vision techniques and deep learning architectures to build a facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. Your completed code should be able to look at any image, detect faces, and predict the locations of facial keypoints on each face; examples of these keypoints are displayed below.

![Facial Keypoint Detection][image1]

The project will be broken up into a few main parts in four Python notebooks, **only Notebooks 2 and 3 (and the `models.py` file) will be graded**:

__Notebook 1__ : Loading and Visualizing the Facial Keypoint Data

__Notebook 2__ : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

__Notebook 3__ : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

__Notebook 4__ : Fun Filters and Keypoint Uses




#### Specify the CNN architecture
| Criteria       		|     Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  Define a CNN in `models.py`. |  Convolutional neural network with 5 convolutional layer and 3 fully-connected layer. This network takes a grayscale squared image. |


### Notebook 2

#### Define the data transform for training and test data
| Criteria       		|     Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  Define a `data_transform` and apply it to a DataLoader. |  The composed transform should include: rescaling/cropping, normalization, and turning input images into torch Tensors. The transform turns any input image into a normalized, square, grayscale image and then a Tensor for the model to take it as input. |

#### Define the loss and optimization functions
| Criteria       		|     Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  MSE loss function and Adam optimizer is used for training the model. |  The loss and optimization functions should be appropriate for keypoint detection, which is a regression problem. |


#### Train the CNN

| Criteria       		|     Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Train the model.  |  Train the CNN after defining its loss and optimization functions. Visualize the loss over time/epochs by printing it out occasionally and/or plotting the loss over time. Save the best trained model. |


### Notebook 3

#### Detect faces in a given image
| Criteria       		|     Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Use a haar cascade face detector to detect faces in a given image. | The submission successfully employs OpenCV's face detection to detect all faces in a selected image. |

#### Transform each detected face into an input Tensor
| Criteria       		|     Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Turn each detected image of a face into an appropriate input Tensor. | Transform the face into a normalized, square, grayscale image and then a Tensor for the model to take in as input (similar to what the `data_transform` did in Notebook 2). |

#### Predict and display the keypoints
| Criteria       		|     Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Predict and display the keypoints on each detected face. | After face detection with a Haar cascade and face pre-processing, apply the trained model to each detected face, and display the predicted keypoints on each face in the image. |

LICENSE: This project is licensed under the terms of the MIT license.
