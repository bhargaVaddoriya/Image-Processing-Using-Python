from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import os

app = Flask(__name__)

def apply_gym_lighting_filter(image):
    
    alpha = 1.3  
    beta = 30    
    enhanced_img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * 0.8
    sharpened_img = cv2.filter2D(enhanced_img, -1, kernel)

    # Adjust color balance (saturation)
    hsv_img = cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2HSV)
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] * 1.15, 0, 255)  # Increase saturation by 15%
    adjusted_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    # Adjust highlights
    adjusted_img = cv2.convertScaleAbs(adjusted_img, alpha=1, beta=-40)  # Decrease highlights by 40%

    return adjusted_img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file selected')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No file selected')

        if file:
            input_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            output_image = apply_gym_lighting_filter(input_image)
            cv2.imwrite('static/output_image.jpg', output_image, [cv2.IMWRITE_JPEG_QUALITY, 95])  # Save with high quality
            return render_template('index.html', output_image='output_image.jpg')

    return render_template('index.html')

@app.route('/download')
def download():
    path = 'static/output_image.jpg'
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)