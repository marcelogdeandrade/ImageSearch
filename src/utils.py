from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
import numpy as np
import os
import matplotlib.pyplot as plt


def term_to_id(term):
    result = {}
    with open('id_to_term.txt') as f:
        for line in f:
            line_splitted = line.strip().split(":")
            _id = line_splitted[0]
            terms = line_splitted[1].replace("'", "").replace('"',"").split(",")
            terms = list(filter(None, terms))
            terms = [x.strip() for x in terms]
            for i in terms:
                result[i] = _id
    return int(result[term])

def mnetv2_input_from_image(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

def get_imgs_paths(dir_path):
    images = []
    for root, directories, filenames in os.walk(dir_path):
        for filename in filenames: 
            images.append(os.path.join(root,filename))
    return images

def get_imgs_probs(model, images, _id):
    probs = []
    for i in images:
        img = image.load_img(i, target_size=(224, 224))
        proc_img = mnetv2_input_from_image(img)
        preds = model.predict(proc_img)
        prob = preds[0][_id]
        probs.append(prob)
    return probs

def get_top_probs(probs, num):
    return np.array(probs).argsort()[-num:][::-1]

def show_imgs(images, idxs):
    for i in idxs:
        img = image.load_img(images[i])
        plt.imshow(img)
        plt.show()

