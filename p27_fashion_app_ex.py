from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
type = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model_save_path = './checkpoint/fashion.ckpt'
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
                                        
model.load_weights(model_save_path)
name='u=2864086092,4085421878&fm=26&gp=0'
image_path=r'C:/Users/Shiwei Pan/Desktop/mine/python/fashion/'+name+'.jpg'
img = Image.open(image_path)

image = plt.imread(image_path)

img=img.resize((28,28),Image.ANTIALIAS)
img_arr = np.array(img.convert('L'))
img_arr = 255 - img_arr

img_arr=img_arr/255.0
plt.imshow(img_arr)
x_predict = img_arr[tf.newaxis,...,tf.newaxis]

result = model.predict(x_predict)
print(result)
pred=np.argmax(result)
print('\n')
print(pred)
print(type[int(pred)])
