![app](https://user-images.githubusercontent.com/83170837/229351563-1b4237b9-e33b-4e69-a4f8-719c83170d92.png)

## Description
This project is a dog breed recognition system that can recognize 114 different dog breeds with an accuracy of 89% using the Inceptionv3 model trained on GPU. The project consists of a backend server built with Flask and a frontend user interface built with React and JavaScript.

## Tools Used
- Python
- Python libraries (Numpy, Pandas, Scikit-learn, OpenCV, Pillow, TensorFlow, Keras, Flask)
- Jupyter notebook, Visual Studio Code
- Javascript, React, HTML, CSS
- Docker
- Kubernetes
- Google Cloud

## Dataset
The dataset used for training and testing the model is the [Stanford Dogs dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset), which contains over 20 000 images of 120 different dog breeds. The breeds were shortened to 114. In addition to this, more than 20 000 additional images were scraped from various sources to create a more diverse and comprehensive dataset.

## Exploratory data analysis

##### Figure 1: Number of images in each dog breed class (only Stanford Dog Dataset)
![old_num_images](https://user-images.githubusercontent.com/83170837/229351670-b55587dc-d287-435f-9299-db7ad0f5f810.png)

The first plot shows the number of images in each class for the old dataset. The x-axis displays the different dog breeds included in the dataset, while the y-axis shows the number of images associated with each breed. The plot is presented as a bar chart, with the height of each bar representing the number of images associated with the corresponding breed. It is worth noting that some of the classes have relatively few images, which could impact the performance of machine learning models trained on this dataset.
The old dataset was used as a starting point for the new dataset. To improve the quality of the data, some pre-processing steps were taken. Firstly, incorrect images were manually removed from the dataset to ensure that all images were of good quality and classes were assigned correctly. Additionally, five classes of dogs that were very similar were removed to avoid redundancy in the dataset. After these pre-processing steps, new data was scraped and added to the dataset, resulting in a new dataset with a significantly larger number of images. These additional images are expected to improve the performance of machine learning models trained on the dataset, due to increased diversity and size of the dataset.
##### Figure 2: Number of images in each dog breed class (New dataset)
![num_images_combined](https://user-images.githubusercontent.com/83170837/229351700-b2c13826-4bc4-4b5f-b6b2-f9f2bb513122.png)
The new dataset features a significant increase in the number of images per class compared to the old dataset. Specifically, data for the majority of classes has at least doubled, while some classes have even more images than before.

##### Figure 3: Few examples of images
![sample_images](https://user-images.githubusercontent.com/83170837/229351753-3301ae67-1929-4a83-8292-484349131dc0.png)
This plot showcases the diversity of our dataset and gives an idea of the types of images we will be working with in our project. We randomly selected ten dog breeds from our dataset, and for each breed, we displayed one sample image. The plot consists of two rows and five columns of images, with the breeds displayed in random order. The images displayed have different resolutions and properties, as the number of images in each breed directory varies between 229 and 511.

##### Figure 4: Distribution of Image Sizes
![distribution_of_image_sizes](https://user-images.githubusercontent.com/83170837/229351802-91d30ac9-98db-433e-8534-784d10aa6581.png)

The above plot shows the distribution of image sizes in the combined dog breed dataset, which includes both the training and testing sets. The x-axis displays the width of the images in pixels, while the y-axis displays the height of the images in pixels. Each blue dot in the plot represents an individual image in the dataset. As can be seen, the majority of images fall in the range of 0x0 to 2500x2500 pixels. However, there are some outliers in the dataset, such as the image with a height of around 7000 pixels and a width of 10000 pixels.

## Manual Oversampling for Balancing Dataset

A quick analysis of the dataset showed that the distribution of images across the classes was imbalanced, with some classes having significantly fewer images than others. Imbalanced datasets can lead to biased model training, where the model performs better on the majority classes and poorly on the minority classes. To address this issue, manual oversampling was used to balance the dataset. The result is here:
![num_images_balanced_dataset](https://user-images.githubusercontent.com/83170837/229351828-755b5993-47fd-450e-add1-16bf8a068eba.png)


## InceptionV3 Image Preprocessing, Data Augmentation and Model Performance
The InceptionV3 model was chosen for the dog breed recognition project because it is a powerful neural network that is good at recognizing images. It can learn different features of the pictures and has already been trained on a lot of pictures, which helps it recognize dog breeds more accurately.
#### Image Preprocessing and Image Augmentation
In the dog breed recognition project, the InceptionV3 image preprocessing function was used to preprocess the images before feeding them to the model. This function normalizes the pixel values and performs other image transformations that help the model learn better features from the images.
##### Figure 5: Sample Augmented and Preprocessed Images (InceptionV3)
![preprocessed_and_augmented_sample_images-Inceptionv3](https://user-images.githubusercontent.com/83170837/229351957-533ad6a0-13f7-4938-bc25-f51838c23d0e.png)
The images in that plot have been augmented and preprocessed using the InceptionV3 preprocessing function. The images have been rotated, sheared, zoomed, and flipped both horizontally and vertically to increase the size of the training dataset and improve model generalization. The images have also been preprocessed using the InceptionV3 model's preprocessing function, which normalizes pixel values and performs other transformations. Each image is labeled with the corresponding dog breed class name, which is hot encoded by default in the data generator.

The above plot shows some example images from the dataset after applying InceptionV3 preprocessing and data augmentation techniques. The images have been augmented by rotating, shearing, zooming, and flipping them both horizontally and vertically. The data augmentation process helps to increase the size of the training dataset and improve the model's ability to generalize to new data. The images have also been preprocessed using the InceptionV3 model's preprocessing function, which normalizes pixel values and performs other transformations.
#### Model Accuracy
The accuracy of the model on the dog breed recognition task is 89%. Additionally, the top 2 categorical accuracy of the model is 96%, which means that the correct dog breed is included in the top 2 predicted breeds in 96% of the test images. This is a strong performance considering the large number of dog breeds in the dataset.
##### Figure 6: InceptionV3 Model Accuracy
![incv3_performance](https://user-images.githubusercontent.com/83170837/229351989-b41957fd-d413-4480-b310-5f707485df67.png)
##### Figure 7: Top 2 categorical accuracy vs iterations (Tensorboard)
![incv3_top2_categorical_accuracy](https://user-images.githubusercontent.com/83170837/229351998-49dd9a4e-4248-4624-9972-51d31d298275.png)

Now let's move on with another model.
## Resnet101 Image Preprocessing, Data Augmentation and Model Performance
ResNet101 was chosen for the dog breed recognition project because it has shown strong performance in image classification tasks and has a large capacity to handle complex and deep neural network architectures.
#### Image Preprocessing and Image Augmentation
When it comes to preprocessing and data augmentation for Resnet101 model, the only one thing that has changed is preprocess function.

##### Figure 8: Sample Augmented and Preprocessed Images (ResNet101)
![preprocessed_and_augmented_sample_images-Resnet](https://user-images.githubusercontent.com/83170837/229352029-270a9107-236d-4d40-8f5e-c832d6eefec3.png)
##### Model Accuracy
The accuracy of the Resnet model is 81% which is lower than InceptionV3 model. Additionally, the top 2 categorical accuracy of the model is 92% which is also lower than previous model.
##### Figure 9: Resnet101 Model Accuracy
![resnet_performance](https://user-images.githubusercontent.com/83170837/229352052-d39851be-4898-4138-a812-d466ff83c12f.png)
##### Figure 10: Top 2 categorical accuracy vs iterations (Tensorboard)
![resnet_top2_categorical_accuracy](https://user-images.githubusercontent.com/83170837/229352064-0e1e8470-37ab-4d0d-a450-5e0f9172a39a.png)

Last but not least EfficientNetB6
## EfficientNetB6 Image Preprocessing, Data Augmentation and Model Performance
EfficientNet was chosen because it was trained on a huge amount of images, and shown highest performance. The Resnet101 model performed at 81% accuracy (input image size = 224,224) and InceptionV3 model performed at 89% accuracy (input image size = 299,299) so there is a correlation. The bigger the input image size the better are the results, so I have chosen B6 architecture of that model, because it demands bigger image sizes then previous models (528,528).

#### Image Preprocessing and Image Augmentation
The only thing that has changed in preprocessing in data augmentation with comparison to previous models is preprocess function.

##### Figure 11: Sample Augmented and Preprocessed Images (EfficientNetB6)
![preprocessed_and_augmented_sample_images-Effnet](https://user-images.githubusercontent.com/83170837/229352074-90a47725-8e3e-463a-9067-f32c90c0f611.png)

##### Model Accuracy
The accuracy of the EfficientnetB6 model is 92% which is the highest i achieved so far. In addition to the top 2 categorical accuracy of the model is 97% which is also the highest.

##### Figure 12: EfficientNetB6 Model Accuracy
![efficientnetb6_performance](https://user-images.githubusercontent.com/83170837/229352091-fb990a79-ccf3-40e5-b511-6f649016ce9c.png)
##### Figure 13: Top 2 categorical accuracy vs iterations (Tensorboard)
![effnetb6_top2_categorical_accuracy](https://user-images.githubusercontent.com/83170837/229352104-e26046d9-70f7-432a-a968-c2d6c97069c3.png)


## Deployment
I have chosen EfficientNetB6 model to be the one to make a predictions due to best results during training. The project has been deployed using Docker and Kubernetes on the Google Cloud Platform. The deployment process involves creating Docker images for the backend and frontend, pushing them to the Google Container Registry, and deploying them on Kubernetes clusters (Deployment with Kubernetes on Google Cloud has been done only for learning purposes).
![kubernetes](https://user-images.githubusercontent.com/83170837/229352116-2c3cb85e-2b64-43af-8578-461fa5e68a4c.png)

## Process of creating that project

- Data Collection (Collect the data from Stanford Dog Breed dataset and additional data scraped from Google Images).
- Data Preprocessing (Preprocess the data by cleaning, resizing, balancing, and labeling the images).
- Model Selection (Select a suitable deep learning model for image classification, and evaluate its performance on the dataset).
- Model Tuning (Fine-tune the selected model using techniques such as transfer learning to improve its accuracy).
- Model Testing (Test manually the model to see how accurate is that on example images).
- Backend Development (Build a Flask backend to host the trained model and expose it through APIs).
- Frontend Development (Develop a React-based frontend to provide a user interface for interacting with the system).
- Containerization (Containerize the application using Docker to ensure consistent behavior across different environments).
- Deployment (Deploy the application on Kubernetes to achieve scalability and high availability).
- Testing (Test the application using manual tests).
- Monitoring (Monitor the application's performance and usage to identify issues and optimize its behavior).

## Running the Application
Make sure You have installed Git and Docker.
To run the application locally, follow these steps:
```sh
git clone https://github.com/pmajews/dog_breed_recognition.git
```
Go to the cloned directory in your CMD and run this command:
```sh
docker-compose up
```
Click that link if upper command were executed successfully: [http://localhost:5173/](http://localhost:5173/)
