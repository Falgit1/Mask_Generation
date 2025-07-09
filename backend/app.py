
import io
import base64

from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
CORS(app)

model = load_model("backend/model/segmentation_VB.hdf5")


def preprocess_image(image_file, target_size=(128, 128)):
    img = Image.open(image_file).convert("RGB")
    img = img.resize(target_size, resample=Image.LANCZOS)

    img_array = np.array(img).astype(np.uint8)
    return np.expand_dims(img_array, axis=0)  # shape: (1, 128, 128, 3)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    preprocessed = preprocess_image(image_file)

    prediction = model.predict(preprocessed)[0]
    mask_image = Image.fromarray((prediction.squeeze() * 255).astype(np.uint8))

    img_io = io.BytesIO()
    mask_image.save(img_io, format="PNG")
    img_io.seek(0)
    mask_base64 = base64.b64encode(img_io.getvalue()).decode("utf-8")

    return jsonify({"mask": mask_base64})


if __name__ == "__main__":
    app.run(debug=True)
