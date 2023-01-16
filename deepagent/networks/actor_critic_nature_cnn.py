from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, Dense, Activation, Flatten, PReLU, LeakyReLU
from tensorflow.keras.models import Model

feature_size = 128
freeze = False

def create_actor_critic_building(input_shapes, action_shape):
    print('input shapes', input_shapes)
    state_encoder = create_dense_state_encoder(input_shapes[0])

    x = Dense(12, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(state_encoder.output)
    x = PReLU()(x)
    x = Dense(12, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = PReLU()(x)
    x = Dense(action_shape[-1], activation='softmax', kernel_initializer='glorot_normal',
              kernel_regularizer=regularizers.l2(0.0001))(x)
    actor = Model(state_encoder.inputs, x)

    x = state_encoder.output
    for i in range(10):
        x = Dense(12, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = PReLU()(x)

    base = Model(state_encoder.inputs, x)

    x = Dense(12, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(base.output)
    x = PReLU()(x)
    x = Dense(12, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = PReLU()(x)
    Q = Dense(1, activation='linear', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(x)

    critic = Model(base.inputs, Q)

    y = Dense(12, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(base.output)
    y = PReLU()(y)
    y = Dense(12, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(y)
    y = PReLU()(y)
    s1 = Dense(8, activation='linear', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(y)

    forward_model =  Model(base.inputs, s1)

    x = Dense(12, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(base.output)
    x = PReLU()(x)
    x = Dense(12, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = PReLU()(x)
    Q = Dense(1, activation='linear', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(x)

    forward_model_error = Model(base.inputs, Q)

    return actor, critic, forward_model, state_encoder, forward_model_error

def create_dense_state_encoder(state_shape):
    vector_input = Input(state_shape)
    mlp = Dense(8, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(vector_input)
    mlp = PReLU()(mlp)
    for i in range(5):
        mlp = Dense(8, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(mlp)
        mlp = PReLU()(mlp)

    return Model(vector_input, mlp)

def create_actor_critic_unit(input_shapes, num_actions, num_q_outputs=1):
    """
    Create actor and critic networks to control units
    :param input_shapes: list of shapes for each input (image first, vector second)
    :param num_actions: number of actions
    :return: action input to critic, actor, critic
    """

    state_encoder = create_state_encoder(input_shapes[0])

    actor_critic = create_actor_critic(state_encoder, num_actions, num_q_outputs)

    random_network_input_shape = (input_shapes[0][0], input_shapes[0][1], 1)
    print('random network input shape: ', random_network_input_shape)
    random_network = create_random_network(random_network_input_shape)

    random_network_predictor = create_forward_model(random_network_input_shape)

    return actor_critic, random_network, random_network_predictor

def create_actor_critic_unit_two_networks(input_shapes, num_actions, num_q_outputs=1):
    """
    Create actor and critic networks to control units
    :param input_shapes: list of shapes for each input (image first, vector second)
    :param num_actions: number of actions
    :return: action input to critic, actor, critic
    """

    state_encoder = create_state_encoder(input_shapes[0])

    actor = create_actor(state_encoder, num_actions)

    critic = create_critic(state_encoder, num_q_outputs)

    random_network_input_shape = (input_shapes[0][0], input_shapes[0][1], 1)
    print('random network input shape: ', random_network_input_shape)
    random_network = create_random_network(random_network_input_shape)

    random_network_predictor = create_forward_model(random_network_input_shape)

    return actor, critic, random_network, random_network_predictor

def create_state_encoder(image_state_shape):
    input = Input(image_state_shape)

    x = Conv2D(32, kernel_size=8, strides=4, padding='same', kernel_initializer='glorot_normal')(input)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size=4, strides=2, padding='same', kernel_initializer='glorot_normal')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_normal')(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(512, kernel_initializer='glorot_normal')(x)
    x = Activation('relu')(x)

    cnn = Model(inputs=[input], outputs=[x])

    return cnn

def create_actor_critic(state_encoder, num_actions, num_q_outputs):
    x = state_encoder.outputs[0]

    logits = Dense(num_actions, kernel_initializer='glorot_normal')(x)

    x = Activation('softmax')(logits)

    y = state_encoder.outputs[0]

    Q = Dense(1, activation='linear', kernel_initializer='glorot_normal')(y)
    outputs = [Q]

    if num_q_outputs == 2:
        PE = Dense(1, activation='linear', kernel_initializer='glorot_normal')(y)
        outputs.append(PE)

    if num_q_outputs == 2:
        return Model(state_encoder.inputs, [x, logits, Q, PE])

    return Model(state_encoder.inputs, [x, logits, Q])

ortho_gain = 2.0 ** 0.5

def create_random_network(image_state_shape):
    input = Input(image_state_shape)

    x = Conv2D(32, kernel_size=8, strides=4, padding='same', kernel_initializer=initializers.Orthogonal(gain=ortho_gain, seed=None))(input)
    x = LeakyReLU()(x)
    x = Conv2D(64, kernel_size=4, strides=2, padding='same', kernel_initializer=initializers.Orthogonal(gain=ortho_gain, seed=None))(x)
    x = LeakyReLU()(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=initializers.Orthogonal(gain=ortho_gain, seed=None))(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    x = Dense(512, kernel_initializer='orthogonal')(x)

    cnn = Model(inputs=[input], outputs=[x])

    return cnn

def create_actor(state_encoder, num_actions):
    x = state_encoder.outputs[0]

    logits = Dense(num_actions, kernel_initializer='glorot_normal')(x)

    x = Activation('softmax')(logits)

    return Model(state_encoder.inputs, [x, logits])

def create_critic(base, num_q_outputs):
    x = base.outputs[0]

    Q = Dense(1, activation='linear', kernel_initializer='glorot_normal')(x)
    outputs = [Q]

    if num_q_outputs == 2:
        PE = Dense(1, activation='linear', kernel_initializer='glorot_normal')(x)
        outputs.append(PE)

    return Model(base.inputs, outputs)

def create_forward_model(image_state_shape):
    input = Input(image_state_shape)

    x = Conv2D(32, kernel_size=8, strides=4, padding='same', kernel_initializer=initializers.Orthogonal(gain=ortho_gain, seed=None))(input)
    x = LeakyReLU()(x)
    x = Conv2D(64, kernel_size=4, strides=2, padding='same', kernel_initializer=initializers.Orthogonal(gain=ortho_gain, seed=None))(x)
    x = LeakyReLU()(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=initializers.Orthogonal(gain=ortho_gain, seed=None))(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    x = Dense(512, kernel_initializer=initializers.Orthogonal(gain=ortho_gain, seed=None))(x)
    x = Activation('relu')(x)
    x = Dense(512, kernel_initializer=initializers.Orthogonal(gain=ortho_gain, seed=None))(x)
    x = Activation('relu')(x)
    x = Dense(512, kernel_initializer=initializers.Orthogonal(gain=ortho_gain, seed=None))(x)

    cnn = Model(inputs=[input], outputs=[x])

    return cnn

if __name__ == "__main__":
    import tensorflow as tf
    import keras.backend as K
    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

    actor_critic_network, random_network, random_network_predictor = create_actor_critic_unit(
        input_shapes=[(65, 65, 4)], num_actions=2)

    actor_critic_network.summary()
    random_network_predictor.summary()