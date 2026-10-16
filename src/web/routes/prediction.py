from flask_login import login_required
from flask import Blueprint, render_template, request, flash, redirect, url_for, send_file, Response, abort
import io
import os
import pandas as pd
from src.web.services.prediction_service import predict_single, predict_batch, get_model_features

prediction_bp = Blueprint('prediction', __name__)

supported_models = {
    'gradient_boosting_final': 'claimwise_final_model.pkl',
    'linear_regression': 'linear_regression.pkl',
    'random_forest': 'random_forest.pkl'
}

ui_models = {
    'gradient_boosting_final': 'ClaimWise Final Model'
}

@prediction_bp.route('/prediction/', methods=['GET', 'POST'])
@login_required
def single_prediction():
    if request.method == 'POST':
        model_id = request.form.get('model_id')
        if model_id not in supported_models:
            model_id = 'gradient_boosting_final'
        
        try:
            prediction_result = predict_single(model_id, request.form.to_dict())
            return render_template('prediction.html', supported_models=ui_models, prediction_result=prediction_result, current_model=model_id)
        except ValueError as e:
            flash(str(e), "error")
            return render_template('prediction.html', supported_models=ui_models)
        except Exception as e:
            flash("An internal error occurred", "error")
            return render_template('prediction.html', supported_models=ui_models)
    
    return render_template('prediction.html', supported_models=ui_models)

@prediction_bp.route('/prediction/batch', methods=['GET', 'POST'])
@login_required
def batch_prediction():
    if request.method == 'POST':
        model_id = request.form.get('model_id')
        if model_id not in supported_models:
            model_id = 'gradient_boosting_final'
        
        file = request.files.get('csv_file')
        if not file:
            flash("No file provided", "error")
            return redirect(url_for('prediction.batch_prediction'))
            
        try:
            df = pd.read_csv(file)
            if len(df) == 0:
                flash("empty", "error")
                return render_template('prediction.html', supported_models=ui_models)
            if len(df) > 50000:
                flash("Batch prediction limited to 50,000 rows", "error")
                return render_template('prediction.html', supported_models=ui_models)
            
            if 'loss' in df.columns:
                flash("Target leakage handled with warning: 'loss' column ignored.", "warning")
                
            predictions = predict_batch(model_id, df)
            df['prediction'] = predictions
            
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            return Response(
                output.getvalue(),
                mimetype="text/csv",
                headers={"Content-Disposition": "attachment;filename=predictions.csv"}
            )
        except Exception as e:
            if "No columns to parse" in str(e):
                flash("Empty CSV file provided.", "error")
            else:
                flash(str(e), "error")
            return render_template('prediction.html', supported_models=ui_models)
                
    return render_template('prediction.html', supported_models=ui_models)
