import cv2
# http://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/
class TileModel():

	def __init__(self, C=80, gamma=0.50625):
		self.C = C
		self.gamma = gamma
		self.model = cv2.ml.SVM_create()
		self.model.setGamma(gamma)
		self.model.setC(C)
		self.model.setKernel(cv2.ml.SVM_RBF)
		self.model.setType(cv2.ml.SVM_C_SVC)
		self.sample_dim = (60,60)
		self.hog = self.get_hog()

	def get_hog(self): 
		win_size = self.sample_dim
		cell_size = (20,20)
		block_size = (40,40)
		block_stride = (10,10)
		nbins = 9
		deriv_aperture = 1
		win_sigma = -1.
		histogramNorm_type = 0
		L2hys_threshold = 0.2
		gamma_correction = 1
		nlevels = 64
		signed_gradient = True

		hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins,
								deriv_aperture, win_sigma, histogramNorm_type, L2hys_threshold,
								gamma_correction, nlevels, signed_gradient)
		return hog

	def load(self, fn):
		self.model = cv2.ml.SVM_load(fn)	
	
	def save(self, fn):
		self.model.save(fn)

	def train(self, samples, labels):
		self.model.train(samples, cv2.ml.ROW_SAMPLE, labels)

	def trainAuto(self, samples, labels):
		self.model.trainAuto(samples, cv2.ml.ROW_SAMPLE, labels)

	def predict(self, tiles):
		return self.model.predict(tiles)[1].ravel()

