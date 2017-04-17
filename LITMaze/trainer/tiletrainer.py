
import cv2
import numpy as np
import itertools as it

from trainingdata import TrainingData
from tilemodel import TileModel


class TileTrainer():


	model_file = '../tiles_svm.dat'


	def __init__(self):
		self.td = TrainingData()
		self.tm = TileModel()


	def create_sheet(self, images):
		cols = int(np.ceil(len(images) / float(np.sqrt(len(images)))));
		samples = iter(images)
		s0 = samples.next()
		pad = np.zeros_like(s0)
		samples = it.chain([s0], samples)
		args = [iter(samples)] * cols
		rows = it.izip_longest(fillvalue=pad, *args)
		return np.vstack(map(np.hstack, rows))


	def evaluate_model(self, samples, descriptors, labels):
		resp = self.tm.predict(descriptors)
		err = (labels != resp).mean()
		print 'Accuracy: %.5f %%' % ((1 - err)*100)

		res = []
		for img, flag in zip(samples, resp == labels):
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
			if not flag: img[...,:2] = 0
			res.append(img)
		
		return self.create_sheet(res)


	def train(self, num_samples=500, auto=False):

		self.td.create_sets(num_samples)

		print 'Loading samples ... '
		samples, labels = self.td.get_training_data()
		print len(samples), 'samples loaded'

		rand = np.random.RandomState(10)
		shuffle = rand.permutation(len(samples))
		samples, labels = samples[shuffle], labels[shuffle]

		print 'Calculating HOG descriptors ... '
		hog = self.tm.get_hog();
		hog_descriptors = []
		for s in samples:
			hog_descriptors.append(hog.compute(s))
		hog_descriptors = np.squeeze(hog_descriptors)

		print 'Splitting data into training (90%) and test sets (10%)... '
		train_n = int(0.9*len(hog_descriptors))
		samples_train, samples_test = np.split(samples, [train_n])
		hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
		labels_train, labels_test = np.split(labels, [train_n])

		print 'Training SVM model,',
		if auto:
			print 'auto-adjusting C and gamma values ...'
			self.tm.trainAuto(hog_descriptors_train, labels_train)
		else:
			print 'C = %.2f and gamma = %.4f ...' % (self.tm.C, self.tm.gamma)
			self.tm.train(hog_descriptors_train, labels_train)

		print 'Saving SVM model to "%s"...' % self.model_file
		self.tm.save(self.model_file)

		print 'Evaluating model ... '
		res = self.evaluate_model(samples_test, hog_descriptors_test, labels_test)
		return res



if __name__ == '__main__':

	tt = TileTrainer()

	res = tt.train(2000)
	img = cv2.resize(res, (600,600), interpolation=cv2.INTER_AREA)
	
	cv2.imshow('Result', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



