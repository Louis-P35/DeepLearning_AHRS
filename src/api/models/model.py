# Import from trirdparty
import tensorflow as tf
import tensorflow_text as text # Requiered for Bert model (need to be imported before tensorflow_hub)
import tensorflow_hub as hub # Requiered for Bert model
from official.nlp.optimization import AdamWeightDecay, WarmUp



# Define the custom objects
custom_objects = {
    "KerasLayer": hub.KerasLayer,  # Add KerasLayer from TensorFlow Hub
    "AdamWeightDecay": AdamWeightDecay,
    "WarmUp": WarmUp
}


class ModelHandler:
    """
    Class to handle the model.
    """
    def __init__(self, model_path: str):
        # Load the model from the path
        self.model: tf.keras.models.Model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    def get_model(self) -> tf.keras.models.Model:
        """
        Return the loaded model.

        Returns:
            tf.keras.models.Model: The loaded model.
        """
        return self.model
    

    def predict(self, data: list) -> list:
        """
        Make a prediction using the loaded model.

        Args:
            data (list): List of input texts to predict.

        Returns:
            list: List of predictions.
        """

        # Flatten the input to ensure a single-dimensional list of strings
        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            # Convert the input data to a tensor
            input_tensor = tf.constant(data)
        elif isinstance(data, list) and all(isinstance(item, list) for item in data):
            # Handle nested structures by flattening
            # Convert the input data to a tensor
            input_tensor = tf.constant([item for sublist in data for item in sublist])
        else:
            raise ValueError("Input data must be a list of strings.")

        # Perform predictions
        predictions = tf.sigmoid(self.model(input_tensor))
        return predictions.numpy().tolist()

