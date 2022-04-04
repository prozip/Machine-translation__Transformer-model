import tensorflow_datasets as tfds

def load(path):
    examples, metadata = tfds.load(path, with_info=True,
                               as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']
    return train_examples