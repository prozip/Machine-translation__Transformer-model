import os, time
import tensorflow as tf
import argparse

from data.getDS import load
from preprocess.build_tokenizers import build
from preprocess.my_tokenizer import MyTokenizer
from model.transformer import Transformer
from hyper_para import HyperParameter
from model.optimizer import getOptimizer

CONVERTER_SAVED_PATH = 'converter_saved'
MAX_TOKENS = 128

BUFFER_SIZE = 20000
BATCH_SIZE = 64

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


def preprocess(train_examples, name):
    if os.path.isdir(CONVERTER_SAVED_PATH + '/' + name):
        print("use existed converter model")
    else:
        print("build new converter model")
        build(train_examples, name)


def filter_max_tokens(inp, targ):
    num_tokens = tf.maximum(tf.shape(inp)[1], tf.shape(targ)[1])
    return num_tokens < MAX_TOKENS


def make_batches(ds, tokenizers):
    def tokenize_pairs(inp, targ):
        inp = tokenizers.pt.tokenize(inp)
        # Convert from ragged to dense, padding with zeros.
        inp = inp.to_tensor()

        targ = tokenizers.en.tokenize(targ)
        # Convert from ragged to dense, padding with zeros.
        targ = targ.to_tensor()
        return inp, targ
    return (
        ds
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
        .filter(filter_max_tokens)
        .prefetch(tf.data.AUTOTUNE))


# Optimizer
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Machine traslation with Transformer')
    parser.add_argument("-t", "--type", choices=['local','tfds'], required=True,
                        help="choose dataset type")
    parser.add_argument('path',help="dataset path")
    parser.add_argument("-e", "--epoch", type=int, required=True,
                        help="training epoch")
    args = parser.parse_args()


    # get dataset
    # DATASETS_PATH = 'ted_hrlr_translate/pt_to_en'

    if args.type == 'local':
        train_examples = load("local", args.path)
    else:
        train_examples = load("tfds", args.path)
        
    name = os.path.basename(args.path).split('.')[0]

    preprocess(train_examples, name)
    myTokenizer = MyTokenizer(CONVERTER_SAVED_PATH + '/' + name)
    tokenizers = myTokenizer.tokenizers

    train_batches = make_batches(train_examples, tokenizers)

    # Hyperparameter
    hpara = HyperParameter()
    num_layers, d_model, dff, num_heads, dropout_rate = hpara.para

    optimizer = getOptimizer(d_model)

    # model
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
        target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
        rate=dropout_rate)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    checkpoint_path = './checkpoints/' + name +'/train'
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    # Training
    EPOCHS = args.epoch
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]


    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = transformer([inp, tar_inp],
                                        training=True)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # input -> portuguese, target -> english
        for (batch, (inp, tar)) in enumerate(train_batches):
            train_step(inp, tar)

            if batch % 100 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        # save every 5 epoch
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

        #always save
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')


        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
