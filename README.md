Photo to Monet with CycleGAN 
================================================

Overview
--------

This repository details the process of developing and deploying a CycleGAN model that transforms photos into Monet-style paintings. The project is designed to showcase best practices in machine learning model lifecycle management, from development through deployment, using modern tools like Docker, Kubernetes, and GitHub Actions.
![/path/to/image.jpeg](https://github.com/chuyangzh/Photo2Monet/blob/main/notebook_asset/monet_painting.jpeg?raw=true)

Project Setup
-------------

### Prerequisites

-   Docker installed on your machine.
-   Access to a GPU for model training and inference tasks.

### Downloading Necessary Resources

-   **Docker Image**: Download and use the Kaggle Python GPU image for a compatible setup:


    `docker pull gcr.io/kaggle-images/python`

-   **Trained Model**: The trained model is not included in the GitHub repository due to file size constraints. Download the trained model from [Chuyang Zhang's Model](https://www.kaggle.com/models/chuyangzhang/monet2photo_cyclegan_v4).
-   **Notebook**: The notebook used for training with in-depth illustration on CycleGAN's components under the hood can be found in this link [Chuyang Zhang's Notebook](https://www.kaggle.com/code/chuyangzhang/i-m-something-of-a-painter-myself-cyclegan).

### Running the Notebook for Model Development and Training

-   **Clone the Repository**: Clone this repository to your local machine to get started with the project.

-   **Set Up the Docker Environment**:

    -   **Start the Kaggle Docker Image**:


        `docker run --name photo2monet -v $(pwd):/home/jupyter -w /home/jupyter -p 8888:8888 --gpus all -it gcr.io/kaggle-gpu-images/python`

    -   **Build Custom Docker Image**: For training on newer GPU architectures (sm_86 and up, such as RTX 3090), use the Dockerfile provided in the repository to build a custom image that incorporates the necessary support:


        `docker build -t custom_photo2monet_image .`

        This command builds a Docker image using the Dockerfile that adjusts the environment to support the newer GPU architectures.
        ![/path/to/image.png](https://github.com/chuyangzh/Photo2Monet/blob/main/notebook_asset/sm_86_support.png?raw=true)

-   **Access the Jupyter Notebook**: Access the Jupyter Notebook through `localhost:8888` to interact with the project directly.
        ![/path/to/image.png](https://github.com/chuyangzh/Photo2Monet/blob/main/notebook_asset/jupyter_lab_container.png?raw=true)

 
  
  ### Setting Up the Web Application


To host the CycleGAN model for inference through a web application:

-   **Requirements**

    -   Ensure the `Docker` environment is correctly set up as detailed above.
    -   Download the `model.pth` file from my Kaggle page or train your own model. Rename the model file to `serving_model_v0.pth` and place it in the `webapp/` directory.

-   **Running the Web Application**

    -   **Build the Docker Image for the Web Application**: Use the Dockerfile within the `webapp/` folder to build the image:
    
    
        `docker build -t webapp-photo2monet ./webapp`
    
    -   **Start the Web Application**: Run the Docker container for the web application:
    
        `docker run --name webapp-photo2monet -p 5000:5000 webapp-photo2monet`
    
    -   **Access the Web Application**: Open a web browser and go to `http://localhost:5000` to interact with the web application and perform image transformations.

      ![/path/to/image.png](https://github.com/chuyangzh/Photo2Monet/blob/main/notebook_asset/demo.gif?raw=true)



Future Development
------------------

-   **CI/CD Pipeline**: Implementation of a CI/CD pipeline using GitHub Actions to automate the build and push of Docker images.
-   **Testing in Local Kubernetes Environment**: Configuring and testing the deployment within a local Kubernetes environment to ensure scalability and reliability.
-   **Deployment on Cloud Services**: Future deployment plans include utilizing free tier services from cloud providers such as Azure to demonstrate cloud deployment scenarios.
