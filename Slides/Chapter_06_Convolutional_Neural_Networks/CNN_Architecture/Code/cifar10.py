# example of loading the cifar10 dataset
import matplotlib.pyplot as plt
from keras.datasets import cifar10
# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# define subplot
fig, ax = plt.subplots(3, 3)
# plot first few images
for i in range(9):
	# plot raw pixel data
	ax[i//3, i%3].imshow(trainX[i])
# show the figure
fig.tight_layout()
fig.set_figheight(8)
fig.set_figwidth(8)
plt.show()