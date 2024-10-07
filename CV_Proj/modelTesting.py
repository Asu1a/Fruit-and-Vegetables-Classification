import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

#Load Model
mdl = tf.keras.models.load_model('/Users/shoko_o/Desktop/CV_Proj/trained_model_relu.keras')

#Visualization and Performaing Predictions on Single Image
image_path = '/Users/shoko_o/Desktop/CV_Proj/image.png'
img = cv2.imread(image_path)

#Testing model
image = tf.keras.preprocessing.image.load_img(image_path, target_size = (64,64))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr]) # Converting single image to batch

prediction = mdl.predict(input_arr)

test_set = tf.keras.utils.image_dataset_from_directory(
    '/Users/shoko_o/Desktop/CV_Proj/archive/test',
    labels = 'inferred',
    label_mode = 'categorical',
    class_names = None,
    color_mode = 'rgb',
    batch_size = 32,
    image_size = (64,64),
    shuffle = True,
    seed = None,
    validation_split = None,
    subset = None,
    interpolation = 'bilinear',
    follow_links = False,
    crop_to_aspect_ratio = False
)

result_index = np.where(prediction[0] == max(prediction[0]))

#Display image
plt.imshow(img)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.show()

#Single prediction
print("It's a {}".format(test_set.class_names[result_index[0][0]]))
