import tensorflow as tf
import tensorflow_text
import argparse


from preprocess.my_tokenizer import MyTokenizer
from hyper_para import HyperParameter
from model.transformer import Transformer
from model.optimizer import getOptimizer
from train import CONVERTER_SAVED_PATH, MAX_TOKENS

EXPORT_DIR = 'translator_model'

class Translator(tf.Module):
    def __init__(self, tokenizers, transformer):
        self.tokenizers = tokenizers
        self.transformer = transformer

    def __call__(self, sentence, max_length=MAX_TOKENS):
        # input sentence is portuguese, hence adding the start and end token
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

        encoder_input = sentence

        # As the output language is english, initialize the output with the
        # english start token.
        start_end = self.tokenizers.en.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a python list) so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(
            dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions, _ = self.transformer(
                [encoder_input, output], training=False)

            # select the last token from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.argmax(predictions, axis=-1)

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output_array = output_array.write(i+1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())
        # output.shape (1, tokens)
        text = tokenizers.en.detokenize(output)[0]  # shape: ()

        tokens = tokenizers.en.lookup(output)[0]

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop. So recalculate them outside
        # the loop.
        _, attention_weights = self.transformer(
            [encoder_input, output[:, :-1]], training=False)

        return text, tokens, attention_weights


class ExportTranslator(tf.Module):
    def __init__(self, translator):
        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        (result,
         tokens,
         attention_weights) = self.translator(sentence, max_length=MAX_TOKENS)

        return result


def print_translation(sentence, tokens):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')


if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser(
        description='test the Translator class')
    parser.add_argument("-m", "--model", required=True,
                        help="choose model")
    parser.add_argument('Input',
                           metavar='input_string',
                           type=str,
                           help='input sentense')
    args = parser.parse_args()
    sentence = args.Input

    # init
    myTokenizer = MyTokenizer(CONVERTER_SAVED_PATH + '/' + args.model)
    tokenizers = myTokenizer.tokenizers

    # Hyperparameter
    hpara = HyperParameter()
    num_layers, d_model, dff, num_heads, dropout_rate = hpara.para

    optimizer = getOptimizer(d_model)

    # reload model
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
        target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
        rate=dropout_rate)

    checkpoint_path = './checkpoints/' + args.model +'/train'
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    translator = Translator(tokenizers, transformer)


    if sentence == 'export':
        translator = ExportTranslator(translator)
        tf.saved_model.save(translator, export_dir=EXPORT_DIR)
        print('model saved to translator/')
    else:
        translated_text, translated_tokens, attention_weights = translator(
            tf.constant(sentence))
        print_translation(sentence, translated_text)
    
