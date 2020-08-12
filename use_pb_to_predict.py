import tensorflow as tf
from configuration import test_image_dir
from prepare_data import load_and_preprocess_image

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    print("-" * 50)
    print("Frozen model layers: ")
    layers = [op.name for op in import_graph.get_operations()]
    if print_graph == True:
        for layer in layers:
            print(layer)
    print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


def main():
    # 测试数据集，

    image_raw = tf.io.read_file(filename=test_image_dir)
    image_tensor = load_and_preprocess_image(image_raw)
    image_tensor = tf.expand_dims(image_tensor, axis=0)


    # Load frozen graph using TensorFlow 1.x functions
    with tf.io.gfile.GFile("./frozen_models2/model.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=["Input:0"],
                                    outputs=["Identity:0"],
                                    print_graph=True)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Get predictions for test images
    predictions = frozen_func(Input=image_tensor)[0]

    # Print the prediction for the first image
    print("-" * 50)
    print("Example prediction reference:")
    print(predictions[0].numpy())


if __name__ == "__main__":
    main()