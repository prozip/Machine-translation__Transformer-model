import tensorflow as tf
import tensorflow_text
import argparse, os

from translator import EXPORT_DIR

if __name__ == '__main__':

    if not os.path.isdir(EXPORT_DIR):
        print('Model not found, Export fist')
        quit()

    # argument parser
    my_parser = argparse.ArgumentParser(
        description='predict model')
    my_parser.add_argument('Input',
                           metavar='input_string',
                           type=str,
                           help='input sentense')
    args = my_parser.parse_args()
    sentence = args.Input

    # reload model and predict
    reloaded = tf.saved_model.load('translator')
    print(reloaded(sentence).numpy())