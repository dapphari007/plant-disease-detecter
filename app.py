import os
import gradio as gr
import tensorflow as tf
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_DIR = os.path.join(BASE_DIR, "data", "labels")
MODEL_PATH = os.path.join(BASE_DIR, "model", "trained_model.keras")

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot",
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

LANGUAGE_OPTIONS = ["English", "Tamil"]


def load_lines(filename):
    filepath = os.path.join(LABELS_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def load_remedies(filename):
    filepath = os.path.join(LABELS_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().split("---")


# Load label data at startup
causes = load_lines("causes.txt")
climatic_conditions = load_lines("climatic_conditions.txt")
remedies = load_remedies("remedies.txt")
yield_rates = load_lines("yield_rates.txt")
symptoms = load_lines("symptoms.txt")
common_plants = load_lines("common_plants.txt")
categories = load_lines("categories.txt")
pesticides = load_lines("pesticides.txt")
extent_without = load_lines("extent_without.txt")
extent_with = load_lines("extent_with.txt")
seed_resistance = load_lines("seed_resistance.txt")

# Load model once at startup
model = tf.keras.models.load_model(MODEL_PATH)


def get_confidence_color(value):
    if value >= 75:
        return "label_green"
    if value >= 50:
        return "label_blue"
    return "label_red"


def create_colored_box(value):
    color_class = get_confidence_color(value)
    return gr.Label(value=f"{value:.2f}%", elem_id=color_class)


def predict(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = np.array([tf.keras.preprocessing.image.img_to_array(image)])
    predictions = model.predict(input_arr)

    result_index = np.argmax(predictions)
    confidence = predictions[0][result_index] * 100

    extent_with_value = float(extent_with[result_index].replace("%", ""))
    extent_without_value = float(extent_without[result_index].replace("%", ""))

    return (
        CLASS_NAMES[result_index],
        create_colored_box(confidence),
        causes[result_index],
        climatic_conditions[result_index],
        symptoms[result_index],
        remedies[result_index],
        yield_rates[result_index],
        common_plants[result_index],
        categories[result_index],
        pesticides[result_index],
        create_colored_box(extent_without_value),
        create_colored_box(extent_with_value),
        seed_resistance[result_index],
    )


def switch_language(language):
    if language == "Tamil":
        return "தயவுசெய்து செடியின் இலை படத்தை பதிவேற்றவும்"
    return "Please upload an image of the plant leaf"


def home(language):
    if language == "Tamil":
        return (
            "# தாவர நோய் கண்டறியும் அமைப்பிற்கு வரவேற்கிறோம்! \n\n"
            "எங்கள் பணி தாவர நோய்களை திறமையாக அடையாளம் காண உதவுவது. "
            "ஒரு தாவரத்தின் படத்தை பதிவேற்றவும், எங்கள் அமைப்பு அதை பரிசோதித்து நோய்களின் எந்த அறிகுறிகளையும் கண்டறியும்.\n\n"
            "### எப்படி செயல்படுகிறது\n"
            "1. **படத்தை பதிவேற்றவும்:** **நோய் கண்டறிதல்** டேப்பில் சென்று சந்தேகமான தாவரத்தின் படத்தை பதிவேற்றவும்.\n"
            "2. **ஆன்லைஸ்:** எங்கள் அமைப்பு படத்தை செயலாக்கி சாத்தியமான நோய்களை கண்டறியும்.\n"
            "3. **முடிவுகள்:** முடிவுகளை காண்க மற்றும் மேலும் நடவடிக்கை எடுக்க பரிந்துரைகளை பெறுங்கள்.\n\n"
            "### ஏன் எங்களை தேர்வுசெய்க?\n"
            "- **துல்லியம்:** எங்கள் அமைப்பு உயர்தரக் கற்றல் தொழில்நுட்பங்களைக் கொண்டுள்ளது.\n"
            "- **பயனர் நட்பு:** எளிய மற்றும் செயல்பாடான இடைமுகம்.\n"
            "- **வேகமான மற்றும் திறமையானது:** முடிவுகளை சில வினாடிகளில் பெறுங்கள்.\n\n"
            "### தொடங்குங்கள்\n"
            "**நோய் கண்டறிதல்** டேப்பை கிளிக் செய்து படத்தை பதிவேற்றவும்!\n\n"
            "### பற்றி\n"
            "**பற்றி** டேப்பில் திட்டம், எங்கள் குழு மற்றும் எங்கள் நோக்கங்களைப் பற்றிய மேலும் அறியவும்."
        )
    return (
        "# Welcome to the Plant Disease Recognition System!\n\n"
        "Our mission is to help in identifying plant diseases efficiently. "
        "Upload an image of a plant, and our system will analyze it to detect any signs of diseases. "
        "Together, let's protect our crops and ensure a healthier harvest!\n\n"
        "### How It Works\n"
        "1. **Upload Image:** Go to the **Disease Recognition** tab and upload an image of a plant with suspected diseases.\n"
        "2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.\n"
        "3. **Results:** View the results and recommendations for further action.\n\n"
        "### Why Choose Us?\n"
        "- **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.\n"
        "- **User-Friendly:** Simple and intuitive interface for seamless user experience.\n"
        "- **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.\n\n"
        "### Get Started\n"
        "Click on the **Disease Recognition** tab to upload an image and experience the power of our Plant Disease Recognition System!\n\n"
        "### About Us\n"
        "Learn more about the project, our team, and our goals on the **About** tab."
    )


def about(language):
    if language == "Tamil":
        return (
            "#### தரவுத்தொகுப்பு பற்றி\n"
            "இந்த தரவுத்தொகுப்பு மூல தரவுத்தொகுப்பிலிருந்து ஆஃப்லைன் பெருக்கம் பயன்படுத்தி மீண்டும் உருவாக்கப்பட்டது.\n\n"
            "இந்த தரவுத்தொகுப்பில் சுமார் 87K RGB படங்கள் உள்ளன, அவை ஆரோக்கியமான மற்றும் நோயுற்ற பயிர் இலைகளாக "
            "வகைப்படுத்தப்பட்டுள்ளன, 38 வெவ்வேறு வகைகளாக வகைப்படுத்தப்பட்டுள்ளன.\n\n"
            "#### உள்ளடக்கம்\n"
            "1. பயிற்சி (70295 படங்கள்)\n"
            "2. சோதனை (33 படங்கள்)\n"
            "3. சரிபார்ப்பு (17572 படங்கள்)"
        )
    return (
        "#### About Dataset\n"
        "This dataset is recreated using offline augmentation from the original dataset. "
        "It consists of about 87K RGB images of healthy and diseased crop leaves, "
        "categorized into 38 different classes. The total dataset is divided into an 80/20 ratio "
        "of training and validation sets preserving the directory structure.\n\n"
        "A new directory containing 33 test images is created later for prediction purposes.\n\n"
        "#### Content\n"
        "1. Train (70,295 images)\n"
        "2. Test (33 images)\n"
        "3. Validation (17,572 images)"
    )


with gr.Blocks(css="""
.label_green { background-color: green; color: white; }
.label_blue { background-color: blue; color: white; }
.label_red { background-color: red; color: white; }
""") as demo:
    gr.Markdown("# PLANT DISEASE RECOGNITION SYSTEM")

    with gr.Tabs():
        with gr.TabItem("Home"):
            language_home = gr.Dropdown(choices=LANGUAGE_OPTIONS, label="Select Language", value="English")
            home_text = gr.Markdown()
            language_home.change(fn=home, inputs=language_home, outputs=home_text)

        with gr.TabItem("About"):
            language_about = gr.Dropdown(choices=LANGUAGE_OPTIONS, label="Select Language", value="English")
            about_text = gr.Markdown()
            language_about.change(fn=about, inputs=language_about, outputs=about_text)

        with gr.TabItem("Disease Recognition"):
            gr.Markdown("## Upload an image of a plant with suspected diseases:")
            language_recog = gr.Dropdown(choices=LANGUAGE_OPTIONS, label="Select Language", value="English")
            language_text = gr.Textbox(
                label="Instructions",
                value=switch_language("English"),
                interactive=False,
            )
            test_image = gr.Image(type="filepath", label="Upload Image")

            predict_btn = gr.Button("Predict")
            result_label = gr.Label(label="Prediction Result")
            cause_label = gr.Label(label="Cause")
            climate_label = gr.Label(label="Climatic Condition")

            with gr.Row():
                category_label = gr.Label(label="Category of Pest (Fungal/Bacterial/Nutrient Deficiency)")
                pesticides_label = gr.Label(label="Pesticides")

            symptoms_label = gr.Label(label="Symptoms")
            common_plants_label = gr.Label(label="Common Plants")

            with gr.Row():
                percentage_label = gr.Label(label="Prediction Confidence")
                extent_without_label = gr.Label(label="Extent of Attack Without Pesticides")
                extent_with_label = gr.Label(label="Extent of Attack With Pesticides")
                yield_label = gr.Label(label="Possibility of Survival (Yield Rate)")

            with gr.Row():
                seed_resistance_label = gr.Label(label="Seed Resistance Varieties")
                remedy_label = gr.Label(label="Remedies")

            predict_btn.click(
                predict,
                inputs=test_image,
                outputs=[
                    result_label, percentage_label, cause_label, climate_label,
                    symptoms_label, remedy_label, yield_label, common_plants_label,
                    category_label, pesticides_label, extent_without_label,
                    extent_with_label, seed_resistance_label,
                ],
            )
            language_recog.change(fn=switch_language, inputs=language_recog, outputs=language_text)

demo.launch()
