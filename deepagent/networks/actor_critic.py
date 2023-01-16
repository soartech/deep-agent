from tensorflow.keras.layers import Dense, Input, Conv2D, Bidirectional, LSTM, Reshape, Flatten, concatenate, Activation
from tensorflow.keras.models import Model

from deepagent.networks.activations import leaky_bounded_linear


def create_actor_critic(input_shape, action_shape):
    cnn = create_cnn(input_shape[1:])
    action_input, critic = create_critic(cnn, input_shape, action_shape)
    actor = create_actor(cnn, input_shape, action_shape)

    return action_input, actor, critic


def create_cnn(input_shape):
    # TODO: BatchNorm doesn't work inside TimeDistributed unless it's been fixed. Try BatchNorm over the
    # whole TimeDistributed layer
    cnn_input = Input(input_shape)
    cnn = Conv2D(32, 7, strides=(3, 3))(cnn_input)
    cnn = Conv2D(64, 3, strides=(2, 2))(cnn)
    cnn = Conv2D(32, 3, strides=(1, 1), padding='same')(cnn)
    cnn = Conv2D(32, 3, strides=(1, 1), padding='same')(cnn)
    cnn = Conv2D(1, 1, strides=(1, 1))(cnn)

    return Model(cnn_input, cnn)


def create_actor(cnn, input_shape, action_shape):
    actor_input = Input(input_shape)
    x = TimeDistributed(cnn)(actor_input)
    x = Reshape((input_shape[0], cnn.output_shape[1] * cnn.output_shape[2]))(x)
    x = Bidirectional(LSTM(16, dropout=0.5, return_sequences=True))(x)
    x = TimeDistributed(Dense(action_shape[-1], activation='linear'))(x)
    x = Activation(leaky_bounded_linear)(x)
    return Model(actor_input, x)


def create_critic(cnn, input_shape, action_shape):
    action_input = Input(action_shape)
    critic_input = Input(input_shape)
    x = TimeDistributed(cnn)(critic_input)
    x = Reshape((input_shape[0], cnn.output_shape[1] * cnn.output_shape[2]))(x)
    x = concatenate([x, action_input])
    x = Bidirectional(LSTM(16, dropout=0.5, return_sequences=True))(x)
    x = Flatten()(x)
    x = Dense(action_shape[0], activation='linear')(x)
    model = Model([critic_input, action_input], x)
    return action_input, model


def _assemble_and_summarize(input_shape, action_shape):
    print('input_shape={} action_shape={}'.format(input_shape, action_shape))
    action_input, actor, critic = create_actor_critic(input_shape, action_shape)
    actor.summary()
    critic.summary()


if __name__ == "__main__":
    _assemble_and_summarize(input_shape=(5, 100, 100, 3), action_shape=(5, 3))
    print('\n{}\n'.format('*' * 150))
    _assemble_and_summarize(input_shape=(15, 65, 65, 5), action_shape=(15, 5))
