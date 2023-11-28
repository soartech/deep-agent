from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from gym import spaces

from deepagent.envs.spaces import DeepAgentSpace

standard_library.install_aliases()
from builtins import *
from tensorflow.keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Activation, Flatten, Reshape, PReLU, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import inspect

from deepagent.experiments.params import params


feature_size = 128
freeze = False


def create_actor_critic_image(state_space: DeepAgentSpace, action_space: DeepAgentSpace):
    kernel_init = params.NetworkInit.kernel_initializer
    kernel_reg = params.NetworkInit.kernel_regularizer
    print('Activation Type Params: ', params.NetworkInit.activation_type)
    is_activation_layer = False
    activation_type = params.NetworkInit.activation_type
    if inspect.isclass(activation_type):
        is_activation_layer = True

    image_inputs = []
    image_activations = []

    cnns = params.NetworkInit.get_cnn_list(state_space.agent_name)

    for image_shape, cnn in zip(state_space.image_shapes, cnns):
        image_input = Input(image_shape)
        image_inputs.append(image_input)
        x = image_input
        for conv in cnn:
            x = conv(x)
            x = activation_type()(x)
        x = Flatten()(x)
        image_activations.append(x)

    if len(image_activations) > 1:
        shared = concatenate(image_activations)
    else:
        shared = image_activations[0]

    for s_num in params.NetworkInit.shared:
        shared = Dense(s_num, kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(shared)
        if is_activation_layer:
            shared = activation_type()(shared)
        else:
            shared = Activation(activation_type)(shared)

    a = shared
    for p_nums in params.NetworkInit.policy:
        a = Dense(p_nums, kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(a)
        if is_activation_layer:
            a = activation_type()(a)
        else:
            a = Activation(activation_type)(a)

    actions = []
    logits_list = []
    if isinstance(action_space.vector_space, spaces.Box):
        action_shape = action_space.vector_shape
        logits = Dense(action_shape[-1], kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(a)
        logits_list.append(logits)
        a = Activation('softmax')(logits)
        actions.append(a)
    elif isinstance(action_space.vector_space, spaces.MultiDiscrete):
        for i in action_space.vector_space.nvec:
            l = Dense(i, kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(a)
            logits_list.append(l)
            act = Activation('softmax')(l)
            actions.append(act)
    else:
        raise Exception('unknown action space')

    c = shared
    for c_nums in params.NetworkInit.q_value:
        c = Dense(c_nums, kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(c)
        if is_activation_layer:
            c = activation_type()(c)
        else:
            c = Activation(activation_type)(c)
    Q = Dense(1, activation='linear', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(c)

    ac = Model(image_inputs, actions + logits_list + [Q])

    return ac, None, None, None

def create_actor_critic_image_vector(state_space: DeepAgentSpace, action_space: DeepAgentSpace):
    kernel_init = params.NetworkInit.kernel_initializer
    kernel_reg = params.NetworkInit.kernel_regularizer
    print('Activation Type Params: ', params.NetworkInit.activation_type)
    is_activation_layer = False
    activation_type = params.NetworkInit.activation_type
    if inspect.isclass(activation_type):
        is_activation_layer=True

    image_inputs = []
    image_activations = []

    cnns = params.NetworkInit.get_cnn_list(state_space.agent_name)

    for image_shape, cnn in zip(state_space.image_shapes, cnns):
        image_input = Input(image_shape)
        image_inputs.append(image_input)
        x = image_input
        for conv in cnn:
            x = conv(x)
            x = activation_type()(x)
        x = Flatten()(x)
        image_activations.append(x)

    vector_input = Input(state_space.vector_shape)
    b = vector_input
    for b_nums in params.NetworkInit.base:
        b = Dense(b_nums, kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(b)
        if is_activation_layer:
            b = activation_type()(b)
        else:
            b = Activation(activation_type)(b)

    shared = concatenate(image_activations + [b])

    for s_num in params.NetworkInit.shared:
        shared = Dense(s_num, kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(shared)
        if is_activation_layer:
            shared = activation_type()(shared)
        else:
            shared = Activation(activation_type)(shared)

    a = shared
    for p_nums in params.NetworkInit.policy:
        a = Dense(p_nums, kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(a)
        if is_activation_layer:
            a = activation_type()(a)
        else:
            a = Activation(activation_type)(a)

    actions = []
    logits_list = []
    if isinstance(action_space.vector_space, spaces.Box):
        action_shape = action_space.vector_shape
        logits = Dense(action_shape[-1], kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(a)
        logits_list.append(logits)
        a = Activation('softmax')(logits)
        actions.append(a)
    elif isinstance(action_space.vector_space, spaces.MultiDiscrete):
        for i in action_space.vector_space.nvec:
            l = Dense(i, kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(a)
            logits_list.append(l)
            act = Activation('softmax')(l)
            actions.append(act)
    else:
        raise Exception('unknown action space')

    c = shared
    for c_nums in params.NetworkInit.q_value:
        c = Dense(c_nums, kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(c)
        if is_activation_layer:
            c = activation_type()(c)
        else:
            c = Activation(activation_type)(c)
    Q = Dense(1, activation='linear', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(c)

    ac = Model(image_inputs + [vector_input], actions + logits_list + [Q])

    return ac, None, None, None

def create_actor_critic_vector(state_space: DeepAgentSpace, action_space: DeepAgentSpace):
    input_shape = state_space.vector_shape
    action_shape = action_space.vector_shape
    print('input shapes', input_shape)
    print('action_shape', action_shape)
    kernel_init = params.NetworkInit.kernel_initializer
    kernel_reg = params.NetworkInit.kernel_regularizer
    print('Activation Type Params: ', params.NetworkInit.activation_type)
    is_activation_layer = False
    activation_type = params.NetworkInit.activation_type
    if inspect.isclass(activation_type):
        is_activation_layer=True

    vector_input = Input(input_shape)
    b = vector_input
    for b_nums in params.NetworkInit.base:
        b = Dense(b_nums, kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(b)
        if is_activation_layer:
            b = activation_type()(b)
        else:
            b = Activation(activation_type)(b)
    base = Model(vector_input, b)

    a = base.output
    for p_nums in params.NetworkInit.policy:
        a = Dense(p_nums, kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(a)
        if is_activation_layer:
            a = activation_type()(a)
        else:
            a = Activation(activation_type)(a)

    actions = []
    logits_list = []
    if isinstance(action_space.vector_space, spaces.Box):
        action_shape = action_space.vector_shape
        logits = Dense(action_shape[-1], kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(a)
        logits_list.append(logits)
        a = Activation('softmax')(logits)
        actions.append(a)
    elif isinstance(action_space.vector_space, spaces.MultiDiscrete):
        for i in action_space.vector_space.nvec:
            l = Dense(i, kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(a)
            logits_list.append(l)
            act = Activation('softmax')(l)
            actions.append(act)
    else:
        raise Exception('unknown action space')

    c = base.output
    for c_nums in params.NetworkInit.q_value:
        c = Dense(c_nums, kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(c)
        if is_activation_layer:
            c = activation_type()(c)
        else:
            c = Activation(activation_type)(c)
    Q = Dense(1, activation='linear', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(c)

    ac = Model([vector_input], actions + logits_list + [Q])

    return ac, None, None, None


if __name__ == '__main__':
    state_space_1 = DeepAgentSpace(image_spaces=[spaces.Box(-1.0, 1.0, shape=(128, 128, 1))], vector_space=spaces.Box(-1.0, 1.0, shape=(287,)))
    state_space_2 = DeepAgentSpace(image_spaces=[spaces.Box(-1.0, 1.0, shape=(128, 128, 1))], vector_space=spaces.Box(-1.0, 1.0, shape=(286,)))
    action_space = DeepAgentSpace(vector_space=spaces.Box(0.0, 1.0, shape=(13,)))
    ac1 = create_actor_critic_image_vector(state_space_1, action_space)
    ac2 = create_actor_critic_image_vector(state_space_2, action_space)

    state_space_just_vec = DeepAgentSpace(vector_space=spaces.Box(-1.0, 1.0, shape=(286,)))
    ac_just_vector = create_actor_critic_vector(state_space_just_vec, action_space)

    default_cnn = params.NetworkInit.get_cnn_list(None)
    params.NetworkInit.set(cnns=lambda: [default_cnn()[0], default_cnn()[0]])
    state_space_multiple_images = DeepAgentSpace(
        image_spaces=[spaces.Box(-1.0, 1.0, shape=(128, 128, 2)), spaces.Box(-1.0, 1.0, shape=(128, 128, 2))], vector_space=spaces.Box(-1.0, 1.0, shape=(286,)))

