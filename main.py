import numpy as np
import glob
import os
import csv
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import random
import csv
from pickle import dump
from pickle import load
import statistics as st
import sys
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Best model voting system 
def get_test_accuracy(test_data, config):
	print('Getting test accuracy of {}'.format('compare'))	
	X_test = test_data['compare']
	y_test = test_data['y_clf']
	model_dir = './'
	predictions_ind = np.zeros((X_test.shape[0], config.n_folds))
	predictions_val = np.zeros((X_test.shape[0], config.n_folds))
	X_test_org = X_test
	for fold in range(config.n_folds):
		X_test = X_test_org 
		model = tf.keras.models.load_model(os.path.join(model_dir, 'fold_{}.h5'.format(fold+1)))
		sc = load(open(os.path.join(model_dir, 'compare/scaler_{}.pkl'.format(fold+1)), 'rb'))
		pca = load(open(os.path.join(model_dir, 'compare/pca_{}.pkl'.format(fold+1)), 'rb'))
		X_test = sc.transform(X_test)
		X_test = pca.transform(X_test)
		pred = model.predict(X_test)
		pred_ind = np.argmax(pred, axis=-1)
		pred_val = np.max(pred, axis=-1)
		predictions_ind[:, fold] = pred_ind
		predictions_val[:, fold] = pred_val

	y_predicted = np.array([])
	for pred_ind, pred_val in zip(predictions_ind, predictions_val):
		ind = np.argmax(pred_val, axis=-1)
		y_predicted = np.append(y_predicted, pred_ind[ind])
	accuracy = accuracy_score(np.argmax(y_test, axis=-1), y_predicted)	
	
	return accuracy


# Majority voting system
"""
def get_test_accuracy(test_data, config):
	print('Getting test accuracy of {}'.format('compare'))	
	X_test = test_data['compare']
	y_test = test_data['y_clf']
	model_dir = './'
	predictions = np.zeros((X_test.shape[0], config.n_folds))
	X_test_org = X_test
	for fold in range(config.n_folds):
		X_test = X_test_org 
		model = tf.keras.models.load_model(os.path.join(model_dir, 'fold_{}.h5'.format(fold+1)))
		sc = load(open(os.path.join(model_dir, 'compare/scaler_{}.pkl'.format(fold+1)), 'rb'))
		pca = load(open(os.path.join(model_dir, 'compare/pca_{}.pkl'.format(fold+1)), 'rb'))
		X_test = sc.transform(X_test)
		X_test = pca.transform(X_test)
		pred = model.predict(X_test)
		pred = np.argmax(pred, axis=-1)
		predictions[:, fold] = pred

	y_predicted = np.array([])
	for row in predictions:
		mode_val = st.mode(row)
		y_predicted = np.append(y_predicted, mode_val)
	accuracy = accuracy_score(np.argmax(y_test, axis=-1), y_predicted)	
	
	return accuracy
"""

def train_a_fold(X_train, y_train, X_val, y_val, fold, config):
	print('Training fold {} of {}'.format(fold, 'compare'))	
	model = create_compare_model(config.compare_features_size)
	sc = StandardScaler()
	sc.fit(X_train)
	X_train = sc.transform(X_train)
	X_val = sc.transform(X_val)
	pca = PCA(n_components=config.compare_features_size)
	pca.fit(X_train)
	X_train = pca.transform(X_train)
	X_val = pca.transform(X_val)
	model_dir = './'

	dump(sc, open(os.path.join(model_dir, 'compare/scaler_{}.pkl'.format(fold)), 'wb'))
	dump(pca, open(os.path.join(model_dir, 'compare/pca_{}.pkl'.format(fold)), 'wb'))
 	
	model.compile(loss=tf.keras.losses.categorical_crossentropy, 
		optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate, 
			epsilon=config.epsilon), metrics=['categorical_accuracy'])
				
	checkpointer = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'fold_{}.h5'.format(fold)), 
			monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, 
			mode='auto', save_freq='epoch')
		
	hist = model.fit(X_train, y_train,
			batch_size=config.batch_size,
			epochs=config.n_epochs,
			verbose=config.verbose,
			callbacks=[checkpointer],
			validation_data=(X_val, y_val),
			sample_weight=None)
			
	model = tf.keras.models.load_model(os.path.join(model_dir, 'fold_{}.h5'.format(fold)))
	train_score = model.evaluate(X_train, y_train, verbose=0)[1]
	val_score = model.evaluate(X_val, y_val, verbose=0)[1]

	return train_score, val_score
							
def train_n_folds(train_data, config):
	X_train_val = train_data['compare']
	y_train_val = train_data['y_clf']
	fold = 0
	train_accuracies = []
	val_accuracies = []
	test_preds = []
	for train_index, val_index in KFold(config.n_folds).split(X_train_val):
		fold += 1
		X_train, X_val = X_train_val[train_index], X_train_val[val_index]
		y_train, y_val = y_train_val[train_index], y_train_val[val_index]
		train_accuracy, val_accuracy = train_a_fold(X_train, y_train, X_val, y_val, fold, config)
		train_accuracies.append(train_accuracy)
		val_accuracies.append(val_accuracy)	
	
	return (np.mean(train_accuracies), np.mean(val_accuracies))	
							
class EasyDict(dict):
	def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
	def __getattr__(self, name): return self[name]
	def __setattr__(self, name, value): self[name] = value
	def __delattr__(self, name): del self[name] 

def create_compare_model(features_size):
	model = tf.keras.Sequential()
	model.add(layers.Input(shape=(features_size,)))
	model.add(layers.Dense(24, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01)))
	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(0.2))
	model.add(layers.Dense(2, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01)))
	
	return model

def get_classification_values(metadata_filename):
	values = []
	with open(metadata_filename, 'r') as f:
		content = f.readlines()[1:]
		for idx, line in enumerate(content):
			token = line.split('; ')[-2].strip('\n')
			if token!='NA':  values.append(int(token))
			else:   values.append(30) # NA fill value
	
	return values

def get_compare_features(compare_filename):
	compare_features = []
	with open(compare_filename, 'r') as file:
		content = csv.reader(file)
		for row in content:
			compare_features = row
	compare_features_floats = [float(item) for item in compare_features[1:-1]]
	return compare_features_floats

def create_config():
	config = EasyDict({
		'n_folds': 5,
		'compare_features_size': 21,
		'n_epochs': 2000,
		'batch_size': 16, 
		'learning_rate': 0.01, 
		'epsilon': 1e-07,
		'verbose': 0})	
	return config	
	
config = create_config()
dataset_dir = './'

cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare/train/cc/*.csv')))
X_cc = np.array([get_compare_features(f) for f in cc_files])
y_cc = np.zeros((X_cc.shape[0], 2))
y_cc[:,0] = 1

cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare/train/cd/*.csv')))
X_cd = np.array([get_compare_features(f) for f in cd_files])
y_cd = np.zeros((X_cd.shape[0], 2))
y_cd[:,1] = 1

X_train = np.concatenate((X_cc, X_cd), axis=0).astype(np.float32)
y_train = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
filenames_train = np.concatenate((cc_files, cd_files), axis=0)

p = np.random.permutation(108)
X_train = X_train[p]
y_train = y_train[p] 
filenames_train = filenames_train[p]

cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare/test/cc/*.csv')))
X_cc = np.array([get_compare_features(f) for f in cc_files])
y_cc = np.zeros((X_cc.shape[0], 2))
y_cc[:,0] = 1

cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare/test/cd/*.csv')))
X_cd = np.array([get_compare_features(f) for f in cd_files])
y_cd = np.zeros((X_cd.shape[0], 2))
y_cd[:,1] = 1

X_test = np.concatenate((X_cc, X_cd), axis=0).astype(np.float32)
y_test = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
filenames_test = np.concatenate((cc_files, cd_files), axis=0)

p = np.random.permutation(48)
X_test = X_test[p]
y_test = y_test[p] 
filenames_test = filenames_test[p]

train_data = {'compare': X_train, 'y_clf': y_train}
test_data = {'compare': X_test, 'y_clf': y_test}


# Evaluate results
train_accuracies = []
val_accuracies = []
test_accuracies = []
for i in range(20):
	print('Iteration {}'.format(i+1))	
	train_accuracy, val_accuracy = train_n_folds(train_data, config)
	train_accuracies.append(train_accuracy)
	val_accuracies.append(val_accuracy)
	test_accuracy = get_test_accuracy(test_data, config)
	test_accuracies.append(test_accuracy)


f = open('train_accuracies.csv', 'w', newline='', encoding='utf-8')
writer = csv.writer(f)
writer.writerow(train_accuracies)
f.close()
#
f = open('val_accuracies.csv', 'w', newline='', encoding='utf-8')
writer = csv.writer(f)
writer.writerow(val_accuracies)
f.close()
#
f = open('test_accuracies.csv', 'w', newline='', encoding='utf-8')
writer = csv.writer(f)
writer.writerow(test_accuracies)
f.close()


