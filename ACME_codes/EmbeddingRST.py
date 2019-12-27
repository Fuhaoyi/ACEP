"""Embedding layer.
"""
from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.topology import Layer

#---------------------------EmbeddingRST-----------------------
class EmbeddingRST_model(Layer):

    def __init__(self, input_dim=21,output_dim=64,input_length=200,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,input_dim,)
            else:
                kwargs['input_shape'] = (None,)
        super(EmbeddingRST_model, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.input_length = input_length

    def build(self, input_shape):
        self.embeddingRST = self.add_weight(
            name='EmbeddingRSTmodel_1',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            dtype=self.dtype,
            trainable=True)
        super(EmbeddingRST_model, self).build(input_shape)  

    def call(self, inputs):
        return K.dot(inputs, self.embeddingRST)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.input_length ,self.output_dim)



