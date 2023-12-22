# NoCap🎓 - Note Capture

### App Description
Many students  To address this issue, our team aims to develop a mobile application named "" that can automatically convert slides or board photos into editable notes. Using machine learning techniques, the app will recognize and extract text from images, formatting and organizing them into easily accessible and editable notes for students.

The "NoCap🎓" mobile application is designed for students who face the problem of not being able to take notes or understand the lecture content when the lecturer changes the slide or board too fast.

## Machine Learning

### Model Building
 Here's the documentation for the provided code, incorporating insights from feedback and addressing noted issues:
**Purpose:**
- This code builds a convolutional neural network (CNN) to perform optical character recognition (OCR), specifically character classification.
- It trains the model on a dataset of character images and then uses it to predict characters in new images.

**Key Sections:**

1. **Import Libraries:**
   - `os`: Operating system operations (accessing files)
   - `cv2`: OpenCV for image processing
   - `numpy`: Numerical operations
   - `matplotlib.pyplot`: Creating visualizations
   - `sklearn.preprocessing`: Data preprocessing
   - `keras`: Building and training the CNN model
   - `tensorflow_io`: Handling file system plugins (note potential issues)

2. **Load and Preprocess Data:**
   - Loads images and labels from specified directories.
   - Preprocesses images by resizing, normalizing pixel values, and encoding labels.

3. **Build CNN Model:**
   - Defines a sequential CNN architecture with convolutional, pooling, and dense layers.
   - Uses ReLU activation for non-linearity.
   - Employs softmax activation in the final layer for multi-class classification.

4. **Train Model:**
   - Compiles the model with 'adam' optimizer, 'sparse_categorical_crossentropy' loss, and 'accuracy' metric.
   - Trains the model on provided training data with a validation set.

5. **Evaluate Model:**
   - Evaluates the trained model on a separate testing set.
   - Provides loss and accuracy metrics.

6. **Make Predictions:**
   - Loads a test image.
   - Preprocesses it similarly to training data.
   - Uses the model to predict the character in the image.
   - Displays the image and prediction.

7. **Segmentation and Prediction for Single Words and Sentences:**
   - Demonstrates how to segment multi-character words and sentences into individual characters.
   - Uses the model to predict each character and reconstruct the text.

8. **Saving and Loading Model:**
   - Shows how to save the trained model using TensorFlow's `model.save()`.
   - Demonstrates loading the saved model using `tf.keras.models.load_model()`.

**Additional Notes:**

- The code includes sections for loading and saving model weights, but those sections are not fully commented.
- It also includes code for loading and saving class names using Pickle, but those sections are not essential for model functionality.
- Consider addressing the warnings related to TensorFlow I/O plugins if they persist.

## **Key Technical Aspects:**

**1. Convolutional Neural Network Architecture:**

- **Layer Structure:**
    - Input layer: Accepts 28x28x1 grayscale images (normalized pixel values).
    - Convolutional layers:
      - 2D convolutional layers with 32 and 64 filters, respectively.
      - Kernel size of 3x3 for both layers.
      - ReLU activation for non-linearity.
    - Max pooling layers:
      - Reduce spatial dimensions with 2x2 pooling.
    - Flatten layer:
      - Converts pooled feature maps into a 1D vector for dense layers.
    - Dense layers:
      - Fully connected layers with 128 and 46 (number of classes) neurons.
      - ReLU activation for hidden layer, softmax activation for output layer.

- **Training Process:**
    - Optimizer: Adam optimizer with default learning rate.
    - Loss function: Sparse categorical crossentropy for multi-class classification.
    - Metric: Accuracy to track model performance.

**2. Data Preprocessing:**

- **Image Resizing:**
    - Reshapes images to a consistent size of 28x28 pixels for model input.
- **Pixel Normalization:**
    - Scales pixel values to the range of 0 to 1 for efficient model training.
- **Label Encoding:**
    - Converts character labels into numerical indices for compatibility with categorical crossentropy loss.

**3. Model Evaluation:**

- **Metrics:**
    - Loss: Sparse categorical crossentropy measures prediction error.
    - Accuracy: Percentage of correctly classified characters in the test set.

**4. Prediction Process:**

- **Image Preprocessing:**
    - Applies the same resizing and normalization techniques used during training.
- **Character Segmentation (for words/sentences):**
    - Splits multi-character images into individual character images for prediction.
- **Prediction:**
    - Passes preprocessed images through the model to obtain character probabilities.
    - Predicted class is the character with the highest probability.

**5. Model Saving and Loading:**

- **Saving:**
    - Serializes the entire model architecture and weights using `model.save()`.
- **Loading:**
    - Restores the model structure and weights using `tf.keras.models.load_model()`.
### Flask Integration Details
As Cloud Computing requested, the ML team would build a a Flask web application that serves a simple OCR (Optical Character Recognition) model. It uses a pre-trained deep-learning model that ML build before to recognize text in an image.

#### Library Used
* `flask`: A web framework for Python.
* `cv2` (OpenCV): Computer vision library for image and video processing.
* `numpy`: A library for numerical operations in Python.
* `matplotlib`: Used for creating visualizations (not essential for the main OCR functionality).
* `pickle`: Used for serializing and deserializing Python objects.
* `tensorflow` (tf): Deep learning library for training and deploying machine learning models.

#### Flask Application Setup
An instance of the Flask web application is created with the name `app`.

#### Model and Class Names Loading
* A pre-trained OCR model (loaded_model) is loaded using TensorFlow's Keras API from the file `OCRmodel.h5`.
* The class names used in the model are loaded from a pickled file (`class_names.pkl`).

#### Flask Routes
**1. `/predict` Route**
- Accepts POST and GET requests.
- Expects an image file in the request.
- Reads the image file, converts it to a NumPy array (nparr), and decodes it using OpenCV.
- Applies image preprocessing steps, including converting to grayscale and thresholding.
- Extracts text regions using contour detection.
- Processes each segmented region, normalizes, and reshapes it to match the input size of the loaded model.
- Makes predictions using the loaded model for each segmented region.
- Assembles the predicted labels into a single string (predicted_text) and returns it as a JSON response.
  
**2. `/` Route**
- Displays 'Hello World' when accessing the root URL.

#### Running the Application
-  The `app.run(debug=True)` line runs the Flask application in debug mode.

#### How to Use
-  Send a POST or GET request to the '/predict' endpoint with an image file attached.
-  The server processes the image, recognizes text regions, predicts the content of each region using the loaded OCR model, and returns the predicted text as a JSON response.

#### Important Notes:
-  Ensure that the required libraries are installed (`flask`, `cv2`, `numpy`, `matplotlib`, `pickle`, `tensorflow`).
-  The paths for the loaded model (`OCRmodel.h5`) and class names (`class_names.pkl`) should be accurate and accessible.
-  The preprocessing steps, model architecture, and class names should be consistent with the model used during training.


## Mobile Development
### Main Feature
The app provides the following features for the user:
* Firebase Google Authorization and authentication
  - we included firebase login using google sign in method to enhance experience and simplify user authorization 
* Add new topic
  - this is the main key of our app, using trained ML that we create on scratch, and deploy it as web service (API) to recognize text on submited picture
* Take picture using camera
  - this is an addtional feature to sumbit picture of our APP
* local databases
    - using RoomDb library as part of android jetpack to saving topic or data user in our APP

### Depedencies
* [Firebase](https://firebase.google.com/?hl=id)
* [Material Dialog](https://github.com/afollestad/material-dialogs?tab=readme-ov-file)
* [RoomDb](https://developer.android.com/training/data-storage/room?hl=id)
* [Fast Android Networking](https://github.com/amitshekhariitbhu/Fast-Android-Networking)

## API Deployment
we got 1 API that installed to the app, and 1 "CC Undeployed" bacouse the CC team run out the time. but you can see it at `CC' repo with Undeployed tag in it
this one we would explain how we install our emergence API, which run a OCR model v1, its a bad at reading text
- Create `main.py` File:

    Place your main application logic in the   `main.py` file. This file handles HTTP requests, defines routes, and implements the core functionality of your web app.

- Create `app.yaml` File:

    Configure your application settings in the `app.yaml` file. This includes specifying the runtime, environment variables, and other configuration details relevant to running your application on Google App Engine.

- Create `requirements.txt` File:

    List all Python packages and their versions that your application depends on in the `requirements.txt` file. This file is used to install the necessary dependencies during the deployment process.

- Deploy Using Gogole Cloud deploy:

    Ensure you have the Google Cloud SDK installed and configured on your local machine. Run the gcloud deploy command to upload your application code and configuration to your specified Google Cloud project. This will make your web application accessible on the web.

  ## Cloud Computing
  Undeployed
  
This one is the CC team's hard work, but, as we already know, this project got undeployed from our app. However, it's here anyway

- Create VM Instance:

    Begin the deployment process by creating a virtual machine instance. Specify the machine type, operating system, and other relevant configurations. VM instances serve as the computing resources where your model will be deployed, enabling it to serve predictions.

- Open SSH to Install Requirements and Deploy the Model:

    After successfully creating the VM instance, establish an SSH connection to access the instance's command line remotely. Once connected, install the necessary requirements for your machine learning model. This step may involve installing libraries, dependencies, and any tools essential for serving the model.

- Deploy Model in VM Instance:

    Utilize the prepared VM to deploy your machine learning model. This involves transferring the trained model files, the inference script, and any additional resources required for making predictions onto the instance. Ensure that the model is properly configured to run within the environment provided by the VM.

- Create API for Cloud Storage:

    If your goal is to create an API (Application Programming Interface) for interacting with your model, consider setting up an API for accessing model files stored in cloud storage, such as Google Cloud Storage. This API can facilitate the retrieval of input data, sending requests to the model, and receiving predictions.
