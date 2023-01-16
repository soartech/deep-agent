from keras.layers.merge import _Merge


class Subtract(_Merge):
    """Layer that subtracts the second input from the first.
    It takes as input a list of tensors (exactly 2),
    all of the same shape, and returns
    a single tensor (also of the same shape).
    """

    def _merge_function(self, inputs):
        if len(inputs) != 2 :
            raise ValueError('Subtract layer should have exactly 2 inputs')
        
        output = inputs[0] - inputs[1]
        return output

def subtract(inputs, **kwargs):
    """Functional interface to the `Subtract` layer.
    # Arguments
        inputs: A list of input tensors (exactly 2).
        **kwargs: Standard layer keyword arguments.
    # Returns
        A tensor, the difference of the inputs.
    """
    return Subtract(**kwargs)(inputs)
