from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved XGBoost model
model = pickle.load(open('best_model_xgb.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        # Fetch all input values from form
        quarter = int(request.form['quarter'])
        department = int(request.form['department'])
        day = int(request.form['day'])
        team = int(request.form['team'])
        targeted_productivity = float(request.form['targeted_productivity'])
        smv = float(request.form['smv'])
        over_time = int(request.form['over_time'])
        incentive = int(request.form['incentive'])
        idle_time = float(request.form['idle_time'])
        idle_men = int(request.form['idle_men'])
        no_of_style_change = int(request.form['no_of_style_change'])
        no_of_workers = int(request.form['no_of_workers'])
        month = int(request.form['month'])

        data = np.array([[quarter, department, day, team, targeted_productivity,
                          smv, over_time, incentive, idle_time, idle_men,
                          no_of_style_change, no_of_workers, month]])

        prediction = model.predict(data)[0]

        if prediction >= 0.9:
            result = "highly productive"
        elif prediction >= 0.7:
            result = "medium productive"
        else:
            result = "less productive"

        return render_template('submit.html', result=result, prediction=round(prediction, 2))


if __name__ == '__main__':
    app.run(debug=True)
