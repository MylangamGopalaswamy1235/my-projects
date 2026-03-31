import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from utils import DATA_DF, match_symptoms, analyze_face_image, analyze_sensor_values, simulate_sensor_value

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    symptoms_list = sorted({s.strip() for col in DATA_DF.columns if 'symptom' in col.lower()
                            for s in DATA_DF[col].dropna().astype(str).tolist()})
    if request.method == 'POST':
        selected = request.form.getlist('symptom')
        top_matches = match_symptoms(selected, DATA_DF, top_k=5)
        return render_template('result.html', mode='symptom', input_data=selected, results=top_matches)
    return render_template('detect.html', symptoms=symptoms_list)

@app.route('/detect/facial', methods=['GET', 'POST'])
def detect_facial():
    if request.method == 'POST':
        file = request.files.get('face_image')
        if not file:
            return redirect(request.url)
        filename = file.filename
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        results = analyze_face_image(path)
        return render_template('result.html', mode='facial', input_data=filename, results=results, image_url=url_for('uploaded_file', filename=filename))
    return render_template('detect_facial.html')

@app.route('/detect/sensor', methods=['GET', 'POST'])
def detect_sensor():
    if request.method == 'POST':
        heart = float(request.form.get('heart_rate', 0))
        temp = float(request.form.get('temperature', 0))
        spo2 = float(request.form.get('oxygen', 0))
        results = analyze_sensor_values({'heart_rate': heart, 'temperature': temp, 'oxygen': spo2})
        return render_template('result.html', mode='sensor', input_data={'heart_rate': heart, 'temperature': temp, 'oxygen': spo2}, results=results)
    simulated = simulate_sensor_value()
    return render_template('detect_sensor.html', simulated=simulated)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
@app.route('/library')
def library():
    return render_template('library.html', df=DATA_DF)

if __name__ == '__main__':
    app.run(debug=True)
