import os
import tensorflow as tf
from tensorflow import keras

__THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def main(model_path):
    model = keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    tflite_model = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=tflite_model)

    for detail in interpreter.get_input_details():
        print('Input:', detail)

    for detail in interpreter.get_output_details():
        print('Output:', detail)

    model_dir = os.path.join(__THIS_DIR, 'model-store', 'mnist', '1')

    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, 'model.tflite'), 'wb') as file:
        file.write(tflite_model)


if __name__ == '__main__':
    model_path = os.path.join(__THIS_DIR, 'samples/simpleMNIST/model/mnist.h5')
    main(model_path)
