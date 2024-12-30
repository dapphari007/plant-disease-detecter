import gradio as gr
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
model = load_model("model/keras_model.h5", compile=False)

# Load the class labels
class_names = open("labels.txt", "r").readlines()

# Define the prediction function
def predict_plant_disease(image):
    """
    Predict the plant disease from an uploaded image.

    Args:
        image: PIL.Image object, the input image.

    Returns:
        str: Prediction result with class name and confidence score.
    """
    # Preprocess the image
    image = image.convert("RGB")  # Ensure image is in RGB mode
    size = (224, 224)  # Resize target size
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # Predict the class
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    
    # Return the result
    return f"Predicted Disease: {class_name}\nConfidence Score: {confidence_score:.2f}"

# Define the Gradio interface
interface = gr.Interface(
    fn=predict_plant_disease,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Plant Disease Detection",
    description=(
        "Upload an image of a plant leaf to classify its disease. "
        "The model identifies diseases in plants and provides the confidence score of the prediction. "
        "Supported plant types include various crops such as apple, grape, tomato, and more."
    ),
    examples=[
        ["examples/healthy_leaf.jpg"],  # Replace with actual example images
        ["examples/diseased_leaf.jpg"]
    ]
)

# Launch the interface
interface.launch()
