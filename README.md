\# 🐟 Multiclass Fish Image Classification 🐟



This repository contains the complete code and documentation for a deep learning project focused on classifying fish species from images. The project involves building and evaluating a custom Convolutional Neural Network (CNN) and a transfer learning model, culminating in a user-friendly web application for real-time predictions.



---



\## 📋 Table of Contents

\* \[Project Summary](#project-summary)

\* \[Problem Statement](#problem-statement)

\* \[Dataset](#dataset)

\* \[Project Workflow](#-project-workflow)

\* \[How to Run the Project](#-how-to-run-the-project)

\* \[Model Details](#-model-details)

\* \[Jupyter Notebook Code (`fish4.ipynb`)](#-jupyter-notebook-code-fish4ipynb)

\* \[Streamlit App Code (`fishapp.py`)](#-streamlit-app-code-fishappy)

\* \[Requirements](#-requirements)

\* \[Project Deliverables](#-project-deliverables)

\* \[Future Work](#-future-work)



---



\## 📝 Project Summary

This capstone project delivers a robust, end-to-end system for identifying fish species from images using deep learning. The primary goal is to address the need for accurate and automated fish classification in fields like sustainable fisheries, ecological monitoring, and the seafood industry.



The solution involves two main technical approaches:

1\.  \*\*Custom CNN\*\*: A Convolutional Neural Network built from scratch using TensorFlow/Keras to establish a baseline performance for fish-specific feature learning.

2\.  \*\*Transfer Learning\*\*: A more advanced model that fine-tunes a pre-trained VGG16 network on the fish dataset. This method leverages powerful, general-purpose features learned from the large ImageNet dataset to achieve higher accuracy and faster convergence.



The project pipeline covers everything from data exploration and preprocessing to model training, hyperparameter tuning, and final evaluation. The best-performing model is saved and deployed as an interactive web application using Streamlit, allowing users to upload a fish image and receive an instant species prediction.



---



\## 🎯 Problem Statement

Design a scalable deep learning solution to accurately classify images of different fish species in a multiclass setting. The project involves:

\* Building a custom CNN model from scratch to serve as a baseline.

\* Leveraging transfer learning with pre-trained architectures to maximize accuracy.

\* Applying rigorous data preprocessing and augmentation to handle image variability.

\* Systematically evaluating model performance and selecting the highest-accuracy model.

\* Persisting the best model for future use in `.h5` or `.keras` format.

\* Deploying the trained model via a Streamlit web application that allows users to upload fish photos and get immediate predicted class labels.



The final solution should be user-friendly, deployment-ready, and able to generalize well to new, unseen images, supporting use cases in ecological research, seafood supply monitoring, and educational outreach.



---



\## 📦 Dataset

This project uses a dataset of fish images categorized into multiple species.



\*\*‼️ IMPORTANT\*\*: The dataset is provided as a Zip file. You must download it and make it available to the project environment.



\* \*\*Download Link\*\*:

&nbsp;   ```

&nbsp;   \[[Click here to download](https://drive.google.com/drive/folders/1iKdOs4slf3XvNWkeSfsszhPRggfJ2qEd)]

&nbsp;   ```



\* \*\*Required Structure\*\*:

For the code rename the Dataset.zip file into Fish.zip(don't unzip the file,since we will do that in ipynb file itself)    

(for your clarity u casn try this step to double check you have downloaded data correctly or not)After unzipping the file, you must have a primary data folder that contains three subdirectories: `train`, `val`, and `test`. Each of these subdirectories must contain folders named after the fish species, with the corresponding images inside.



&nbsp;   ```

&nbsp;   Dataset/

&nbsp;   └── data/

&nbsp;       ├── train/

&nbsp;       │   ├── animal\_fish/

&nbsp;       │   │   ├── image1.jpg

&nbsp;       │   │   └── ...

&nbsp;       │   ├── fish\_sea\_food\_sea\_bass/

&nbsp;       │   └── ...

&nbsp;       ├── val/

&nbsp;       │   ├── animal\_fish/

&nbsp;       │   └── ...

&nbsp;       └── test/

&nbsp;           ├── animal\_fish/

&nbsp;           └── ...

&nbsp;   ```



---



\## ⚙️ Project Workflow

1\.  \*\*Data Preprocessing \& Augmentation\*\*: Images are loaded and prepared for training. This includes rescaling pixel values to a `\[0, 1]` range and applying data augmentation (rotation, zoom, flips) to the training set to improve model robustness.

2\.  \*\*Model Training\*\*:

&nbsp;   \* A custom CNN is built and trained from the ground up.

&nbsp;   \* A transfer learning model using VGG16 is fine-tuned on the fish dataset.

3\.  \*\*Hyperparameter Tuning\*\*: KerasTuner is used with Random Search to find the optimal hyperparameters (e.g., learning rate, number of layers, dropout rate) for both models to maximize validation accuracy.

4\.  \*\*Model Evaluation\*\*: The models are evaluated on the test set using metrics like accuracy, precision, recall, and F1-score. Confusion matrices and classification reports are generated to analyze performance in detail.

5\.  \*\*Model Saving\*\*: The best-performing model (VGG16) is saved in the `.keras` format for deployment.

6\.  \*\*Deployment\*\*: A web application is built using Streamlit, allowing users to upload a fish image and get a real-time classification from the saved model.



---



\## 🚀 How to Run the Project



\### 1. Training the Model (Google Colab)

1\.  \*\*Open Google Colab\*\*: Go to \[colab.research.google.com](https://colab.research.google.com).

2\.  \*\*Upload Notebook\*\*: Upload the `FishImageClassification.ipynb` file.

3\.  \*\*Set Runtime\*\*: Go to `Runtime > Change runtime type` and select \*\*GPU\*\* as the hardware accelerator. This is crucial for training speed.

4\.  \*\*Upload Data\*\*: In the Colab file explorer on the left, upload your dataset Zip file.

5\.  \*\*Run the Notebook\*\*: Execute the cells in the notebook sequentially. The notebook will handle unzipping the data, training the models, performing hyperparameter tuning, and saving the final trained model (`model2\_transferlearning.keras`) to your Google Drive.



\### 2. Running the Web App (Locally)

1\.  \*\*Download Files\*\*:

&nbsp;   \* Download the `fishapp.py` file.

&nbsp;   \* Download the trained model file `model2\_transferlearning.keras` from your Google Drive and place it in the \*\*same folder\*\* as `fishapp.py`.

2\.  \*\*Install Libraries\*\*: Open your terminal or command prompt and install the required Python libraries:

&nbsp;   ```bash

&nbsp;   pip install -r requirements.txt

&nbsp;   ```

3\.  \*\*Run Streamlit\*\*: Navigate to the folder containing your files in the terminal and run the following command:

&nbsp;   ```bash

&nbsp;   streamlit run fishapp.py

&nbsp;   ```

4\.  \*\*Use the App\*\*: Your web browser will open a new tab with the running application. You can now upload a fish image to get a prediction.



---



\## 🧠 Model Details

Two primary models were developed and compared:



1\.  \*\*Custom CNN\*\*: A sequential model with several convolution, batch normalization, and dropout layers. It served as a good baseline but achieved lower accuracy compared to the transfer learning approach.

2\.  \*\*VGG16 Transfer Learning (Final Model)\*\*: This model uses the VGG16 architecture pre-trained on ImageNet, with its top classification layers removed. A new custom head was added, consisting of a Global Average Pooling layer, a Dense layer, Batch Normalization, Dropout, and a final softmax output layer. This model significantly outperformed the custom CNN and was selected for the final deployment.



---



\##  notebooks/`FishImageClassification.ipynb`

```python

\#

\# This is a placeholder for the full content of fish4.ipynb.

\# The actual file contains over 100 cells of code and markdown.

\# Please refer to the uploaded fish4.ipynb file for the complete, executable code.

\# The code covers the following key sections:

\#

\# 1.  \*\*Data Loading and Preprocessing\*\*:

\#     - Importing libraries (TensorFlow, Keras, Pandas, etc.).

\#     - Unzipping the dataset.

\#     - Setting up ImageDataGenerator for training, validation, and test sets with data augmentation.

\#

\# 2.  \*\*Exploratory Data Analysis (EDA)\*\*:

\#     - Visualizing sample images.

\#     - Plotting class distributions (Bar plots, Pie charts).

\#     - Analyzing image dimensions and aspect ratios (Histograms).

\#     - Checking for corrupt files and empty directories.

\#

\# 3.  \*\*Hypothesis Testing\*\*:

\#     - Performing ANOVA and t-tests to check for statistical differences in image properties between classes.

\#

\# 4.  \*\*Model 1: Custom CNN Implementation\*\*:

\#     - Building a sequential CNN model from scratch.

\#     - Compiling and training the model.

\#     - Evaluating performance with classification reports and confusion matrices.

\#     - Hyperparameter tuning with KerasTuner.

\#

\# 5.  \*\*Model 2: VGG16 Transfer Learning Implementation\*\*:

\#     - Loading the VGG16 base model with pre-trained weights.

\#     - Freezing the base layers and adding a custom classification head.

\#     - Compiling and training the transfer learning model.

\#     - Evaluating its superior performance.

\#     - Hyperparameter tuning with KerasTuner.

\#

\# 6.  \*\*Model Explainability\*\*:

\#     - Using Grad-CAM to visualize which parts of an image the model focuses on for its predictions.

\#

\# 7.  \*\*Saving and Loading Models\*\*:

\#     - Saving the final tuned models to Google Drive.

\#     - Loading the saved model for a sanity check prediction on unseen data.

\#

\# To run this, please open the provided fish4.ipynb file in Google Colab.

\# 🐟 Fish Species Classifier Web App 🚀



This repository contains the source code for an interactive web application that classifies fish species from an uploaded image. The app is built with \*\*Streamlit\*\* and powered by a \*\*TensorFlow/Keras\*\* deep learning model.



---



\## ✨ Features



\* \*\*🖼️ Image Upload\*\*: Easily upload fish images in `jpg`, `jpeg`, or `png` format.

\* \*\*🧠 Real-Time Prediction\*\*: Get an instant species classification from the trained VGG16 model.

\* \*\*📊 Confidence Score\*\*: See the model's confidence level for its top prediction.

\* \*\*📈 Probability Chart\*\*: Visualize the top 3 most likely species and their probabilities in a clean bar chart.

\* \*\*📜 Detailed Probabilities\*\*: An expandable section to view the model's probability scores for all possible classes.



---



\## 📸 Demo



!\[Demo GIF of the Fish Classifier App](https://i.imgur.com/YOUR\_DEMO\_GIF.gif)

\*(\*\*Note\*\*: You can replace the link above with a GIF or screenshot of your running application.)\*



---



\## 🛠️ Tech Stack



\* \*\*Framework\*\*: Streamlit

\* \*\*Deep Learning\*\*: TensorFlow / Keras

\* \*\*Data Handling\*\*: Pandas, NumPy

\* \*\*Image Processing\*\*: Pillow (PIL)



---



\## ⚙️ Setup and Installation



Follow these steps to run the application on your local machine.



\### 1. Prerequisites

\* Python 3.8+

\* `pip` package manager



\### 2. Clone the Repository

Clone this repository to your local machine:

```bash

git clone \[THIS\_REPOSITORY\_LINK]

cd \[THIS\_REPOSITORY\_FOLDER]

Install Dependencies

Install all the required libraries using the requirements.txt file.

pip install -r requirements.txt

\##Download the Model File

You need the trained model file to run the app.



Download the model2\_transferlearning.keras file.



Place it in the same root directory as your fishapp.py file

&nbsp;Run the Application

Once the setup is complete, run the following command in your terminal:





python -m streamlit run fishapp.py


