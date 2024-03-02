from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import joblib
import docx2txt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model = joblib.load('model.pk1')
# model = joblib.load('/Users/priyankarajbanshi/Downloads/Cosines/model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    resume = request.files['resume']
    job_description = request.files['job_description']

    resume_text = docx2txt.process(resume)
    job_text = docx2txt.process(job_description)

    # Calculate similarity
    text = [resume_text, job_text]
    count_matrix = model.transform(text)
    similarity_score = cosine_similarity(count_matrix)[0][1] * 100

    return redirect(url_for('result', similarity_score=similarity_score))

@app.route('/result')
def result():
    similarity_score = request.args.get('similarity_score', default=0, type=float)
    return render_template('result.html', similarity_score=similarity_score)

if __name__ == '__main__':
    app.run(debug=True)
