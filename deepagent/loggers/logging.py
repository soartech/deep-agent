import numpy as np
import tensorflow as tf
from tensorboard.plugins.mesh import summary_v2 as mesh_summary

tensorboard_writer = None
tensorboard_step = None
tensorboard_prefix = ''


def get_tensorboard_writer():
    global tensorboard_writer
    if tensorboard_writer is None:
        from deepagent.experiments.params import params
        tensorboard_writer = tf.summary.create_file_writer(params.ModuleParams.tensorboard_dir)
    return tensorboard_writer


def set_tensorboard_prefix(prefix):
    global tensorboard_prefix
    tensorboard_prefix = prefix


def update_tensorboard_step(step):
    global tensorboard_step
    tensorboard_step = step


def log_tensorboard_value(name, value, also_print=False):
    global tensorboard_step
    global tensorboard_prefix
    name = tensorboard_prefix + name
    if tensorboard_step is None:
        return
    if also_print:
        print('tensorboard_log', name, value, tensorboard_step)
    writer = get_tensorboard_writer()
    with writer.as_default():
        tf.summary.scalar(name, value, step=tensorboard_step)


def log_tensorboard_hist(name, data, bins=None, also_print=False):
    global tensorboard_step
    global tensorboard_prefix
    name = tensorboard_prefix + name
    if tensorboard_step is None:
        return
    if also_print:
        print('tensorboard_log', name, data, tensorboard_step)
    writer = get_tensorboard_writer()
    with writer.as_default():
        tf.summary.histogram(name, data, buckets=bins, step=tensorboard_step)


def log_tensorboard_count_image(name, count_image):
    img = count_image / count_image.max()
    img = np.reshape(img, (-1, img.shape[0], img.shape[1], 1))
    img = img.astype(np.float32)
    np.clip(img, 0.0, 0.999, out=img)
    log_tensorboard_image(name, img)


def log_tensorboard_image(name, img):
    global tensorboard_step
    global tensorboard_prefix
    name = tensorboard_prefix + name
    writer = get_tensorboard_writer()
    with writer.as_default():
        tf.summary.image(name, img, step=tensorboard_step)


def log_point_cloud(name, points, colors):
    global tensorboard_step
    global tensorboard_prefix
    name = tensorboard_prefix + name
    writer = get_tensorboard_writer()

    points = np.expand_dims(points, 0)
    colors = np.expand_dims(colors, 0)

    # Camera and scene configuration.
    config_dict = {
        'camera': {'cls': 'PerspectiveCamera',
                   'fov': 35,
                   },
        'lights': [
            {
                'cls': 'AmbientLight',
                'color': '#ffffff',
                'intensity': 0.75,
            }, {
                'cls': 'DirectionalLight',
                'color': '#ffffff',
                'intensity': 0.75,
                'position': [0, -1, 2],
            }],
        'material': {
            'cls': 'PointsMaterial',
            'size': 0.1
        }

    }

    vertices_tensor = tf.constant(points, dtype=tf.float32, shape=points.shape)
    colors_tensor = tf.constant(colors, dtype=tf.int32, shape=colors.shape)

    with writer.as_default():
        mesh_summary.mesh(
            name=name, vertices=vertices_tensor,
            colors=colors_tensor, config_dict=config_dict, step=tensorboard_step)
