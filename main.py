import pandas as pd
import feature_extraction
import numpy as np
from WavFileHelper import WavFileHelper
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint

# Set the path to the Metadata CSV file
metadatapath = 'C:/Users/dju10/AudioNN/UrbanSound8K/metadata/UrbanSound8K.csv'
#Set the path to the actual WAV files
wavfiledatapath = 'C:/Users/dju10/AudioNN/UrbanSound8K/audio/'
#Set the path to where you want to save the best model from each fold
modelsavepath = 'C:/Users/dju10/AudioNN/saved_models/'

metadata = pd.read_csv(metadatapath)

features = []
num_folds = 10
num_rows = 40
num_columns = 174
num_channels = 1
num_epochs = 72
num_batch_size = 256
verbosity = 1

wavfilehelper = WavFileHelper()

audiodata = []
for index, row in metadata.iterrows():
    file_name = wavfiledatapath+'fold'+str(row["fold"])+'/'+str(row["slice_file_name"])
    class_label = row["class"]
    data = feature_extraction.extract_features(file_name)
    features.append([data, class_label])

#Convert into a Pandas DataFrame
featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

print("Finished feature extraction from ", len(featuresdf), ' files')

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

X = X.reshape(X.shape[0], num_rows, num_columns, num_channels)

num_labels = yy.shape[1]

kfold = KFold(n_splits=num_folds, shuffle=False)

acc_per_fold = []
loss_per_fold = []

fold_no = 1
for train, test in kfold.split(X, yy):
    
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(num_labels, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    def get_model_name(k):
        return 'model_'+str(k)+'.h5'
    
    checkpoint = ModelCheckpoint(modelsavepath+get_model_name(fold_no), 
							monitor='val_accuracy', verbose=1, 
							save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    history = model.fit(X[train], yy[train],
              validation_split=0.1,
              batch_size=num_batch_size,
              epochs=num_epochs,
              callbacks=callbacks_list,
              verbose=verbosity)
    
    model.load_weights(modelsavepath + "model_"+str(fold_no)+".h5")
    
    scores = model.evaluate(X[test], yy[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
     
    fold_no = fold_no + 1
     
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')