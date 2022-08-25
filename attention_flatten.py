class attention_flatten(Layer): # Based on the source code of Keras flatten
    def __init__(self, keep_dim, **kwargs):
        self.keep_dim = keep_dim
        super(attention_flatten, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise Exception('The shape of the input to "Flatten" is not fully defined (got  + str(input_shape[1:]) + . Make sure to pass a complete "input_shape" or "batch_input_shape" argument to the first layer in your model.')
        return (input_shape[0], self.keep_dim)   # Remove the attention map

    def call(self, x, mask=None):
        x=x[:,:self.keep_dim]
        return K.batch_flatten(x)
