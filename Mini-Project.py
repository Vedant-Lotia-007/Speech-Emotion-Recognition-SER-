"""
Created on Tue Jun  5 17:43:36 2021

@author: Vedant
"""

import librosa
import soundfile
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Extract features (mfcc, chroma, mel) from a sound file

# mfcc: Mel Frequency Cepstral Coefficient, represents the short-term power spectrum of a sound
# chroma: Pertains to the 12 different pitch classes
# mel: Mel Spectrogram Frequency

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

# Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgusting',
  '08':'surprised'
}
# Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust', 'sad', 'angry', 'surprised']

# Load the data and extract features for each sound file
def load_data(flag,path,test_size):
    x,y=[],[]
    
    for file in glob.glob(path):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
            
        
    if(flag==0):        
        return train_test_split(np.array(x), y, test_size=test_size, random_state=9)
    else:
        return x,y;
    
# Training    
x_train,x_test,y_train,y_test=load_data(0,"C:\\Users\\HP\\Desktop\\Mini Project\\RAVDESS Speech Audios\\Actor_*\\*.wav",0.2)



# Get the shape of the training and testing datasets
print("Trained and tested data respectively: ",(x_train.shape[0], x_test.shape[0]))
print("-----------------------------------------------------------------------------------")

# Get the number of features extracted
print(f'Total Features extracted: {x_train.shape[1]}')

# Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

# Train the model
model.fit(x_train,y_train)

# Predict for the test set
y_pred=model.predict(x_test)

# Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))

# Testing
# m,n=load_data(1,"C:\\Users\\HP\\Desktop\\Mini Project\\03-01-06-02-02-01-01.wav",0.0)
# m,n=load_data(1,"C:\\Users\\HP\\Desktop\\Mini Project\\03-01-03-01-01-01-13.wav",0.0)
m,n=load_data(1,"C:\\Users\\HP\\Desktop\\Mini Project\\03-01-03-02-01-02-24.wav",0.0)
user_pred= model.predict(m)
accuracy=accuracy_score(y_true=n, y_pred=user_pred)
print("-----------------------------------------------------------------------------------")
print("Emotion Recognised :",user_pred[0])
print("Accuracy: {:.2f}%".format(accuracy*100))
