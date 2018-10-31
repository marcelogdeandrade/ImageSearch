from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
import numpy as np
import os
import os.path
import matplotlib.pyplot as plt
import pickle
import progressbar

FILENAME = "probs.pickle"

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
    for root, _, filenames in os.walk(dir_path):
        for filename in filenames: 
            images.append(os.path.join(root,filename))
    return images

def get_imgs_probs(model, images):
    probs = {}
    print("Calculando probabilidades das imagens:")
    for i in progressbar.progressbar(range(len(images))):
        img = image.load_img(images[i], target_size=(224, 224))
        proc_img = mnetv2_input_from_image(img)
        preds = model.predict(proc_img)
        prob = preds[0]
        probs[images[i]] = prob
    return probs

def save_probs(probs):
    with open(FILENAME, 'wb') as handle:
        pickle.dump(probs, handle, protocol=pickle.HIGHEST_PROTOCOL)

def open_probs():
    with open(FILENAME, 'rb') as handle:
        probs = pickle.load(handle)
        return probs

def get_probs_id(probs, _id):
    probs_id = {}
    for key, value in probs.items():
        probs_id[key] = value[_id]
    return probs_id

def get_top_probs(probs, num):
    return sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:num]

def show_imgs(images, top_imgs):
    for i in top_imgs:
        print("Confiança da imagem {0}: {1:.2f}%".format(i[0], i[1]*100))
        img = image.load_img(i[0])
        plt.imshow(img)
        plt.show()

def check_file_exists():
    return os.path.isfile(FILENAME) 

