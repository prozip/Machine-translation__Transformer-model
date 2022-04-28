from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import tensorflow_text as text
import tensorflow as tf
import pathlib
import re, os

reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
END = tf.argmax(tf.constant(reserved_tokens) == "[END]")
INPUT_PATH = 'pt_vocab.txt'
TARGET_PATH = 'en_vocab.txt'

def add_start_end(ragged):
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count, 1], START)
    ends = tf.fill([count, 1], END)
    return tf.concat([starts, ragged, ends], axis=1)


def cleanup_text(reserved_tokens, token_txt):
    # Drop the reserved tokens, except for "[UNK]".
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
    bad_token_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

    # Join them into strings.
    result = tf.strings.reduce_join(result, separator=' ', axis=-1)

    return result


def write_vocab_file(filepath, vocab):
    with open(filepath, 'w', encoding="utf-8") as f:
        for token in vocab:
            print(token, file=f)


class CustomTokenizer(tf.Module):
    def __init__(self, reserved_tokens, vocab_path):
        self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text('utf-8').splitlines()
        self.vocab = tf.Variable(vocab)

        # Create the signatures for export:

        # Include a tokenize signature for a batch of strings.
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        enc = enc.merge_dims(-2, -1)
        enc = add_start_end(enc)
        return enc

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return cleanup_text(self._reserved_tokens, words)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)


def gen_vocab(train_examples):

    train_en = train_examples.map(lambda pt, en: en)
    train_pt = train_examples.map(lambda pt, en: pt)

    bert_tokenizer_params = dict(lower_case=False)

    bert_vocab_args = dict(
        # The target vocabulary size
        vocab_size=8000,
        # Reserved tokens that must be included in the vocabulary
        reserved_tokens=reserved_tokens,
        # Arguments for `text.BertTokenizer`
        bert_tokenizer_params=bert_tokenizer_params,
        # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
        learn_params={},
    )

    pt_vocab = bert_vocab.bert_vocab_from_dataset(
        train_pt.batch(1000).prefetch(2),
        **bert_vocab_args
    )
    write_vocab_file(INPUT_PATH, pt_vocab)

    en_vocab = bert_vocab.bert_vocab_from_dataset(
        train_en.batch(1000).prefetch(2),
        **bert_vocab_args
    )
    write_vocab_file(TARGET_PATH, en_vocab)

def build(train_examples, name):

    # generate vocab and save to txt file
    if (os.path.exists(TARGET_PATH) & os.path.exists(INPUT_PATH)):
        print('Found vocab file')
    else:
        print('Generating vocab')
        gen_vocab(train_examples)

    tokenizers = tf.Module()
    tokenizers.pt = CustomTokenizer(reserved_tokens, INPUT_PATH)
    tokenizers.en = CustomTokenizer(reserved_tokens, TARGET_PATH)

    model_name = 'converter_saved/' + name
    tf.saved_model.save(tokenizers, model_name)

