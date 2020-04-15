import os
import string

import tensorflow as tf


def mount_drive():
    from google.colab import drive
    drive.mount('/content/gdrive')


def get_dataset_from_gdrive(file_name):
    os.system(''.join(['cp "/content/gdrive/My Drive/', file_name, '" .']))


def get_previous_weights_from_gdrive(weight_folder_name):
    os.system(''.join(['cp -r "/content/gdrive/My Drive/', weight_folder_name, '" .']))


def get_latest_model_name(weight_folder_name):
    all_files = [
            (name, os.path.getmtime(''.join(['./', weight_folder_name, '/', name])))
            for name in os.listdir(''.join(['./', weight_folder_name, '/']))]
    latest_uploaded_file = sorted(all_files, key=lambda x: -x[1])[0][0]
    return ''.join(['./', weight_folder_name, '/', latest_uploaded_file])


def read_dataset(file_name):
    with open(file_name, 'r', errors='ignore') as f:
        txt = f.read()
    return txt


def get_list_from_char(char, char_dict, classes):
    num = char_dict[char]
    return [0] * num + [1] + [0] * (classes - 1 - num)


def get_chars():
    chars = string.ascii_lowercase + '.," '
    return chars


def get_character_vars(index_in, chars=None):
    if chars is None:
        chars = get_chars()
    classes = len(chars)
    char_dict = {chars[i]: i for i in range(len(chars))}
    if index_in:
        char_dict_list = {chars[i]: 1 - i / classes / 2 for i in range(classes)}
    else:
        char_dict_list = {chars[i]: get_list_from_char(chars[i], char_dict, classes) for
                          i in
                          range(classes)}
    return chars, char_dict, char_dict_list, classes


def reformat_string(input_string, chars):
    """
    WARNING: This function removes all characters it has no clue about.
    This includes anything related to numbers as well as most symbols.
    If those characters are required in any way, do some preprocessing,
    such as replacing '1' with 'one'.
    """
    input_string = ' '.join(input_string.split())
    input_string = input_string.lower()
    input_string = ''.join([c for c in input_string if c in chars])
    return input_string


def get_neuron_list(neurons_per_layer, layer, class_neurons, classes):
    return [neurons_per_layer] * layer


def get_test_string(char_set=None):
    test_string = """You then insert the tool into the holes evident on the blinkie's 
    surface.  The
location of the hole depends on the color of the blinkie (and manufacturer).
The blue blinkie commonly found in wealthy suburban areas is disabled by
inserting the tool into the hole directly in the middle of the blinkie's body.
You should hear a "click" and the blinkie will cease to function.  The red
blinkie commonly found in highway construction sights uses the offset hole
located in either the upper hand right or left of the blinkie's body.  You will
NOT hear a click with the red one because it uses a slide switch instead of a
pushbutton one.  Again, the blinkie will turn off.  Yellow and black blinkies
turn off in a similar way as the red ones."""
    if char_set is None:
        char_set = get_chars()
    test_string = reformat_string(test_string, char_set)
    return test_string


def get_tf_generator(python_generator, batch_size, outputs):
    output_shape = (outputs, 1) if outputs > 1 else (1,)
    tf_generator = tf.data.Dataset.from_generator(
        generator=lambda: map(tuple, python_generator),
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape((None,)), tf.TensorShape(output_shape))
        )
    tf_generator = tf_generator.batch(batch_size)
    tf_generator = tf_generator.shuffle(16, reshuffle_each_iteration=True)
    tf_generator = tf_generator.repeat(batch_size)
    tf_generator = tf_generator.prefetch(tf.data.experimental.AUTOTUNE)
    return tf_generator
