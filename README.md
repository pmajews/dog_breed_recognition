# Dog Breed Recognition Project
![](./app_image.png)

## Description
This project is a dog breed recognition system that can recognize 115 different dog breeds with an accuracy of 89% using the Inceptionv3 model trained on GPU. The project consists of a backend server built with Flask and a frontend user interface built with React and JavaScript.

## Dataset
The dataset used for training and testing the model is the [Stanford Dogs dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset), which contains over 20 000 images of 120 different dog breeds. The breeds were shortened to 115. In addition to this, more than 20 000 additional images were scraped from various sources to create a more diverse and comprehensive dataset.

## Tools Used
- Python
- Python libraries (Numpy, Pandas, Scikit-learn, OpenCV, Pillow, TensorFlow, Keras, Flask)
- Jupyter notebook, Visual Studio Code
- Javascript, React, HTML, CSS
- Docker
- Kubernetes
- Google Cloud

## Deployment
The project has been deployed using Docker and Kubernetes on the Google Cloud Platform. The deployment process involves creating Docker images for the backend and frontend, pushing them to the Google Container Registry, and deploying them on Kubernetes clusters (Deployment with Kubernetes on Google Cloud has been done only for learning purposes).
![](./kubernetes.png)

## Process of creating that project

- Data Collection (Collect the data from Stanford Dog Breed dataset and additional data scraped from Google Images).
- Data Preprocessing (Preprocess the data by cleaning, resizing, and labeling the images).
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
