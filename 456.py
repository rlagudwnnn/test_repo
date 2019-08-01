from tensorflow.python.keras.models import load_model

model = load_model('ronono2.h5')
model.summary()

import matplotlib.pyplot as plt
#%matplotlib inline

test_num = plt.imread('./000047.bmp')
test_num = test_num[:, :, 0]
test_num = (test_num > 80) * test_num
test_num = test_num.astype('float32') / 255
plt.imshow(test_num, cmap='Greys', interpolation='nearest')
plt.show()
test_num = test_num.reshape((1, 28, 28, 1))
print('The Answer is ', model.predict_classes(test_num))

