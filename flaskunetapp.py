from flask import Flask, request, render_template, redirect, url_for
from flask import send_file
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Load your trained U-Net model
model = tf.keras.models.load_model('path/to/your/unet_brain_stroke_detection.h5')  # Update this path

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize image to match model input
    image_array = np.array(image) / 255.0  # Normalize image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Image postprocessing function
def postprocess_mask(mask):
    mask = mask.squeeze()  # Remove batch dimension
    mask = (mask > 0.5).astype(np.uint8) * 255  # Apply threshold
    return Image.fromarray(mask)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    try:
        image = Image.open(file.stream)
        image_array = preprocess_image(image)
        
        # Predict mask
        mask_array = model.predict(image_array)
        mask_image = postprocess_mask(mask_array[0])

        # Save the mask image to a BytesIO object and send it as a response
        buf = io.BytesIO()
        mask_image.save(buf, format='PNG')
        buf.seek(0)
        
        return send_file(buf, mimetype='image/png', as_attachment=True, attachment_filename='mask.png')

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
