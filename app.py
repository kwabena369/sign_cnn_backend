import base64
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import onnxruntime as ort

app = Flask(__name__)
CORS(app)

LETTER_NAMES = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

session = ort.InferenceSession('sign_with_cnn.onnx')
print("ONNX Model loaded successfully!")


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('L')
    image = image.resize((28, 28), Image.LANCZOS)
    image_array = np.array(image, dtype=np.float32) / 255.0
    tensor = image_array.reshape(1, 1, 28, 28)  
    return tensor


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        image_bytes = base64.b64decode(data['image'])
        tensor = preprocess_image(image_bytes)

        # Run inference
        outputs = session.run(['output'], {'input': tensor})
        logits = outputs[0][0]

        # Softmax manually
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()

        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

        return jsonify({
            'letter': LETTER_NAMES[predicted_index],
            'confidence': round(confidence, 4),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'SignLanguageCNN ONNX'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
