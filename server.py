# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask
import pandas as pd
import numpy as np
import librosa
from keras.models import load_model
from flask import render_template,request
from werkzeug.utils import secure_filename
import math
import traceback

model = load_model('my_model_cnn.h5')

genre_list = ['Blues','Classical','Country','Disco','Hiphop','Jazz','Metal','Pop','Reggae','Rock'] 


def predict(model, X):#Passed model in the prediction..
    
    
    # add a dimension to input data for sample - model.predict() expects a 4d array
    X = X[np.newaxis, ...]# (1, 130, 13, 1)We are adding extra dimension to make a 4D array.

    # perform prediction
    prediction = model.predict(X)

    predicted_index = np.argmax(prediction, axis=1)
    #Out of prediction array we want to extract the max probable value.Gives 1D value b/w 0 to 9.
    
    print("Predicted label: {}".format( genre_list[predicted_index[0]]))
    return predicted_index[0]


def save_mfcc(f,num_mfcc=13,n_fft=2048,hop_length=512,num_segments=10):
    SAMPLE_RATE = 22050
    TRACK_DURATION = 30 #seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION#6 lacs approx.. 30*22050
    #dictionary for data storage
    data={
          "mfcc":[],#mfccs for each segment
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

  

    # load audio file
    file_path = f
    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
     # process all segments of audio file
    for d in range(num_segments):

        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment
        print(start,finish)
        # extract mfcc
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T
          # store only mfcc feature with expected number of vectors
        if len(mfcc) == num_mfcc_vectors_per_segment:
            # We append the no. of mfccs in list if condition valid
            data["mfcc"].append(mfcc)
#             data["labels"].append(i-1)

#             print("{}, segment:{}".format(file_path, d+1))
    return data



def predict_model(filename):
	data = save_mfcc(filename)
	prediction_list= []
	for i in range(9):
		gp_test = data['mfcc'][i]
		gp_test = gp_test[..., np.newaxis]
		prediction_list.append(predict(model, gp_test))
		
	print(prediction_list)
	from collections import Counter
	data = Counter(prediction_list)
	data.most_common(1)[0][0]
	 
	return genre_list[data.most_common(1)[0][0]]


# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def home_page():
	return render_template('index.html')


#Handling error 404 and displaying relevant web page
@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html'),404
 
#Handling error 500 and displaying relevant web page
@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html'),500


@app.route('/upload', methods = ['GET', 'POST'])
# ‘/’ URL is bound with hello_world() function.
def upload():
	print(request.method)
	if request.method == 'POST': # check if the method is post
		
		print(request.__dir__)
		f = request.files['file'] # get the file from the files object
		print(f)
		f.save(secure_filename(f.filename)) # this will secure the file
		print(f.filename)
		output = predict_model(f.filename)
		print(output)
		# return 'file uploaded successfully. Predicted genre is ' +  output  # Display this message after uploading
	return render_template('upload.html', variable=output)

	

# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application
	# on the local development server.
	app.run()
