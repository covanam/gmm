import scipy.io
from gmm import GMM
import numpy as np

# load data and split train/test
data = scipy.io.loadmat('dataset/cardio.mat')
train_data = data['X'][0:1500]
test_data = data['X'][1500:]
test_label = data['y'][1500:].reshape(-1).astype(np.bool)

# ...
gmm = GMM(k=1)
gmm.fit(train_data)

# testing
prob = gmm.prob(test_data)
prediction = prob < gmm.thresh  # 1: abnormal, 0: normal

precision = np.sum(prediction == test_label) / prediction.shape[0]

prediction_on_abnormal = prediction[test_label]
recall = np.sum(prediction_on_abnormal) / prediction_on_abnormal.shape[0]

print('precision = ', precision)
print('recall = ', recall)
