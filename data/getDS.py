import tensorflow_datasets as tfds
import tensorflow as tf


def do_map(ds):
    splited = tf.strings.split(ds, sep='\t')
    inp = tf.squeeze(tf.slice(splited, [0], [1]))
    targ = tf.squeeze(tf.slice(splited, [1], [1]))
    return (inp, targ)


def load(type, path):
    if (type == 'local'):
        dataset = tf.data.TextLineDataset(path)
        examples = dataset.map(lambda ds: do_map(ds))
        return examples
    else:
        examples, metadata = tfds.load(path, with_info=True,
                                       as_supervised=True)
        train_examples, val_examples = examples['train'], examples['validation']
        return train_examples
