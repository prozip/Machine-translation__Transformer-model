class HyperParameter:
    def __init__(self):
        num_layers = 4
        d_model = 128
        dff = 512
        num_heads = 8
        dropout_rate = 0.1
        self.para = [num_layers, d_model, dff, num_heads, dropout_rate]