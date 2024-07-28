from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from io import BytesIO
import base64
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define utility functions
def read_and_preprocess_image(image_data, target_size=(128, 128)):
    image = Image.open(image_data).convert("RGB")  # Ensure image is in RGB format
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Rescale
    return np.expand_dims(image, axis=0)

def overlay_mask_on_image(original_image, segmentation_mask):
    original_image = original_image.convert("RGBA")
    segmentation_mask = (segmentation_mask * 255).astype(np.uint8)
    segmentation_mask = Image.fromarray(segmentation_mask).convert("L")
    red_mask = Image.new("RGBA", original_image.size, (255, 0, 0, 0))
    red_mask = Image.composite(red_mask, Image.new("RGBA", original_image.size, (255, 255, 255, 0)), segmentation_mask)
    blended_image = Image.blend(original_image, red_mask, alpha=0.5)
    return blended_image

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load models
classification_model_path = os.path.abspath(r'C:\Users\Tanusree\Desktop\BRAIN_STROKE\model\classificationmodel.h5')
model = tf.keras.models.load_model(classification_model_path)

unet_model_path = os.path.abspath(r'C:\Users\Tanusree\Desktop\BRAIN_STROKE\model\unet_brain_stroke_detection.h5')
unet_model = tf.keras.models.load_model(unet_model_path)

gpt2_model_dir = r'C:\Users\Tanusree\Desktop\BRAIN_STROKE\model\gpt2_model'
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_dir)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_dir)

# Define class names
CLASS_NAMES = ["Normal", "Ischemic Stroke", "Hemorrhagic Stroke", "Ischemic-Thrombotic strokes","Ischemic-Embolic strokes.","Transient Ischemic Attack or Mini-Stroke","Brain Stem Stroke","Other"]

# Define routes
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if file:
        try:
            image = read_and_preprocess_image(file)
            predictions = model.predict(image)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])

            return jsonify({
                'class': predicted_class,
                'confidence': float(confidence),
            })

        except Exception as e:
            print(f'Error during processing: {e}')
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'No file uploaded'}), 400

@app.route('/segment', methods=['POST'])
def segment():
    file = request.files.get('file')
    if file:
        try:
            # Read and preprocess the image
            image = read_and_preprocess_image(file)
            
            # Get the segmentation mask from the model
            predictions = unet_model.predict(image)
            print(f"Predictions shape: {predictions.shape}")
            segmentation_mask = np.squeeze(predictions[0])
            print(f"Segmentation mask shape: {segmentation_mask.shape}")

            # Convert the segmentation mask to a PIL image
            mask_image = Image.fromarray((segmentation_mask * 255).astype(np.uint8))

            # Save mask image to a buffer
            buffer = BytesIO()
            mask_image.save(buffer, format="PNG")
            buffer.seek(0)

            # Return the image as a response
            return send_file(buffer, mimetype='image/png')

        except Exception as e:
            print(f"Error in /segment: {e}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'No file uploaded'}), 400

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    question = data.get('question', '')

    if question:
        try:
            print("Question received for /chatbot")
            inputs = tokenizer.encode(question, return_tensors='pt')
            outputs = gpt2_model.generate(inputs, max_length=150, num_return_sequences=1)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return jsonify({'question': question, 'answer': answer})

        except Exception as e:
            print(f"Error in /chatbot: {e}")
            return jsonify({'error': 'Error processing question'}), 500

    return jsonify({'error': 'No question provided'}), 400

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting Flask app")
    app.run(debug=True)
