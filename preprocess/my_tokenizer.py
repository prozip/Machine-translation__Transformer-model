import tensorflow as tf

class MyTokenizer:
    def __init__(self, model_path):
        self.tokenizers = tf.saved_model.load(model_path)
    def tokenize(self, en_examples):
        return self.tokenizers.en.tokenize(en_examples)
    def detokenize(self, encoded):
        round_trip = self.tokenizers.en.detokenize(encoded)
        return round_trip
    def lookup(self, encoded):
        tokens = self.tokenizers.en.lookup(encoded)
        return tokens
