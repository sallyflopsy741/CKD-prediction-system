import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import time
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the model safely
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    app.logger.error(f"Error loading model: {e}")
    model = None  # Prevent crashing if the model isn't found

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure form data is present
        if not request.form:
            return "Please submit the form data using POST."

        # Get user details
        name = request.form.get('name', 'Unknown')  
        bg = request.form.get('bg', 'Unknown')
        sex = request.form.get('sex', 'Unknown')

        # Helper function to get float values safely
        def get_float(field):
            try:
                return float(request.form[field])
            except (ValueError, KeyError, TypeError):
                return 0.0  # Default to 0.0 if invalid

        # Helper function to get categorical integer values
        def get_int(field):
            try:
                return int(request.form.get(field, 0))
            except Exception as e:
                app.logger.error(f"Error converting field '{field}': {e}")
                return 0  # Default to 0 if invalid

        # Get input values
        age = get_float('age')
        bp = get_float('bp')
        sg = get_float('sg')
        al = get_float('al')
        su = get_float('su')
        bgr = get_float('bgr')
        bu = get_float('bu')
        sc = get_float('sc')
        sod = get_float('sod')
        pot = get_float('pot')
        hemo = get_float('hemo')
        wc = get_float('wc')
        pcv = get_float('pcv')
        rc = get_float('rc')

        # Get categorical fields
        rbc = get_int('rbc')  
        pc = get_int('pc')
        pcc = get_int('pcc')
        ba = get_int('ba')
        htn = get_int('htn')
        dm = get_int('diabetes')  # Make sure 'diabetes' is the correct key from the form
        cad = get_int('cad')
        appet = get_int('appet')
        pe = get_int('pe')
        ane = get_int('ane')

        # Ensure model is loaded
        if model is None:
            return "Error: Model not loaded. Check model.pkl file."

        # Make prediction
        start_time = time.time()
        prediction = model.predict([[age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]])
        end_time = time.time()
        testing_time = round(end_time - start_time, 4)

        # Convert categorical values to meaningful text
        rbc_text = "Normal" if rbc == 1 else "Abnormal"
        pc_text = "Normal" if pc == 1 else "Abnormal"
        pcc_text = "Present" if pcc == 1 else "Absent"
        ba_text = "Present" if ba == 1 else "Absent"
        htn_text = "Yes" if htn == 1 else "No"
        dm_text = "Yes" if dm == 1 else "No"
        cad_text = "Yes" if cad == 1 else "No"
        appet_text = "Good" if appet == 0 else "Poor"
        pe_text = "Yes" if pe == 1 else "No"
        ane_text = "Yes" if ane == 1 else "No"

        # Determine the result
        if prediction[0] == 0:
            result = "No Kidney Disease"
            color = 'green'
            status = 'Kidney Disease Negative'
        else:
            result = "Kidney Disease"
            color = 'red'
            status = 'Kidney Disease Positive'

        return render_template('report.html', 
                               name=name, bg=bg, sex=sex, age=age, bp=bp, 
                               result=result, color=color, status=status, 
                               sg=sg, al=al, su=su, bgr=bgr, bu=bu, sc=sc, sod=sod, pot=pot, hemo=hemo, 
                               rbc=rbc_text, pc=pc_text, pcc=pcc_text, ba=ba_text, wc=wc, 
                               htn=htn_text, dm=dm_text, cad=cad_text, 
                               appet=appet_text, pe=pe_text, ane=ane_text, testing_time=testing_time)

    except Exception as e:
        app.logger.error(f"Error in predict route: {e}", exc_info=True)
        return f"Error: {e}"

# Custom error handlers
@app.errorhandler(405)
def method_not_allowed(e):
    app.logger.error(f"Method Not Allowed: {e}", exc_info=True)
    return f"Error 405: Method Not Allowed. Details: {e}", 405

@app.errorhandler(404)
def page_not_found(e):
    app.logger.error(f"Page Not Found: {e}", exc_info=True)
    return f"Error 404: Page Not Found. Details: {e}", 404

if __name__ == "__main__":
    app.run(debug=True)
