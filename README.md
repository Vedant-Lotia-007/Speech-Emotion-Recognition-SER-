# Speech Emotion Recognition (SER)

Speech Emotion Recognition (SER) is a Python application that recognizes human emotions from speech data. This project is an implementation during the undergraduate coursework, utilizing the librosa and sklearn libraries and tested on the RAVDESS dataset.

## Overview
SER is complex due to the subjective nature of emotions and the challenge of annotating audio data accurately. This implementation seeks to mitigate these challenges by employing a dataset that has been rated by multiple individuals to ensure the reliability of the emotion annotations.

The project uses the [RAVDESS dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio), which contains 7356 files rated by 247 individuals 10 times for emotional validity, intensity, and genuineness.

## Prerequisites
- Python 3.x
- librosa
- sklearn
- numpy
- soundfile

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/Speech-Emotion-Recognition-SER-.git
cd Speech-Emotion-Recognition-SER-
```

## Usage 
1. Ensure the dataset is downloaded and extracted to the appropriate directory.
2. Run the script:
   ```bash
   python Mini-Project.py
   ```

## Features
- Utilizes Mel Frequency Cepstral Coefficient (MFCC), chroma, and mel features extracted from audio data for training the model.
- Employs a Multi-Layer Perceptron Classifier for emotion recognition.
- Trains and tests the model using data from the RAVDESS dataset.
- Allows for customizable paths to easily change the dataset being used for training and testing.

## Results
The script outputs the accuracy of the model after training and testing. You can test the model with additional data by modifying the script to point to a different dataset or audio files.

## Acknowledgements
Thanks to the creators of the RAVDESS dataset for providing publicly accessible, high-quality emotional speech audio data.

## Contact
For any questions, feel free to contact at vedantlotia007@gmail.com or raise an issue on this GitHub repository.
