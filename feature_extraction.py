import librosa
import numpy as np

max_pad_len = 174

def extract_features(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccsscaled = np.pad(mfccs, pad_width=((0,0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print(e)
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccsscaled