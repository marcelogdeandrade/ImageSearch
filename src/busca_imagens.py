from keras.applications.mobilenet import MobileNet

import argparse
import utils as utils

IMAGES_FOLDER = "../images"

# Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--term", 
                    help="Pass the term of the image you are looking",
                    type=str)
parser.add_argument("--build-index",
                    action='store_true',
                    help="Recreate file with image probabilities",
                    default=False)
args = parser.parse_args()

term = args.term
model = MobileNet(weights='imagenet')

images = utils.get_imgs_paths(IMAGES_FOLDER)
_id = utils.term_to_id(term)

if utils.check_file_exists() and not args.build_index:
    probs = utils.open_probs()
else:
    probs = utils.get_imgs_probs(model, images)
    utils.save_probs(probs)

probs_id = utils.get_probs_id(probs, _id)
top_imgs = utils.get_top_probs(probs_id, 3)
utils.show_imgs(images, top_imgs)
