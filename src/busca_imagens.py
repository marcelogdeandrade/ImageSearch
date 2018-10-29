from keras.applications.mobilenet import MobileNet

import argparse
import utils as utils

IMAGES_FOLDER = "../images"

# Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--term", help="Pass the term of the image you are looking",
                    type=str)
args = parser.parse_args()

term = args.term
model = MobileNet(weights='imagenet')

images = utils.get_imgs_paths(IMAGES_FOLDER)
_id = utils.term_to_id(term)
probs = utils.get_imgs_probs(model, images, _id)
idxs = utils.get_top_probs(probs, 3)
utils.show_imgs(images, idxs)
