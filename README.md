**Speech Emotion Recognition (SER) using LSTM**

**Overview:**
This repository presents a Python implementation of a Speech Emotion Recognition (SER) system utilizing Long Short-Term Memory (LSTM) networks. The project aims to predict human emotions from audio files, specifically leveraging the TESS Toronto Emotional Speech Set dataset. SER finds applications in improving user experience, enhancing Voice User Interfaces (VUIs), and various other domains such as customer services, recommender systems, and healthcare applications.

**Objective:**
The main objectives of this project are as follows:
1. Load the TESS dataset, extract audio features (MFCC), and split the data into training and testing sets.
2. Train two models (MLP and LSTM) as emotion classifiers and evaluate their performance.
3. Calculate the accuracy of the models to assess their effectiveness in emotion recognition.

**Pipeline:**
1. **Loading the Dataset**: Load the TESS dataset, extract audio features such as MFCC, and split the data into training and testing sets.
2. **Training the Model**: Train the LSTM model on the preprocessed data.
3. **Testing the Model**: Measure the accuracy of the trained model on a separate validation set.

## Dataset
The TESS Toronto Emotional Speech Set dataset is used for training and evaluating the emotion recognition model. It contains a collection of short audio clips, each labeled with one of seven emotions: angry, disgusted, fearful, happy, neutral, sad, and surprised.

You can download the TESS dataset from [this link](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess/download?datasetVersionNumber=1).


**Dependencies:**
Ensure that you have the following dependencies installed:
- pandas
- numpy
- seaborn
- matplotlib
- librosa
- scikit-learn
- keras (with TensorFlow backend)

Install the dependencies using pip:
```bash
pip install pandas numpy seaborn matplotlib librosa scikit-learn keras
```

**Usage:**
1. **Load the Dataset**: Update the `dataset_folder` variable with the path to the TESS dataset folder. Run the provided code to load the dataset.
2. **Exploratory Data Analysis**: Visualize the distribution of classes and analyze audio samples using waveplots and spectrograms.
3. **Feature Extraction**: Extract Mel-frequency cepstral coefficients (MFCC) features from the audio files.
4. **Model Training**: Define an LSTM model architecture using Keras, compile the model, and train it using the extracted features.
5. **Plot the Results**: Visualize the training and validation accuracy/loss over epochs.

**Directory Structure:**
- **data/**: Contains the TESS dataset or any other relevant data.
- **notebooks/**: Jupyter notebooks for data exploration, preprocessing, model training, and evaluation.
- **src/**: Source code for data preprocessing, model architecture, training, and evaluation.
- **models/**: Trained models saved here for later use.
- **utils/**: Utility functions for data loading, feature extraction, and evaluation.

**License:**
This project is licensed under the MIT License. Refer to the [LICENSE](LICENSE) file for details.

**Feedback and Contributions:**
Feedback, bug reports, and contributions are welcome. Feel free to create an issue or pull request on the GitHub repository.
