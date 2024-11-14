import logging
from flask import Flask, render_template, jsonify, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(message)s')

app = Flask(__name__)
data_store = []

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/ingest-data', methods=['POST'])
def ingest_data():
    data = request.get_json()
    data_store.append(data)
    logging.info(f"Data received: {data}")
    return jsonify({"message": "Data received"}), 200

@app.route('/data')
def data():
    df = pd.DataFrame(data_store)
    return jsonify(df.to_dict(orient='records'))

@app.route('/heatmap')
def heatmap():
    df = pd.DataFrame(data_store)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[['temperature', 'vibration', 'wind_speed']].corr(), annot=True, cmap='coolwarm')
    plt.savefig('static/heatmap.png')
    return render_template('heatmap.html', plot_url='/static/heatmap.png')

@app.route('/forecast')
def forecast():
    df = pd.DataFrame(data_store)
    y = df['temperature']
    model = auto_arima(y, seasonal=True, m=7)
    forecast, conf_int = model.predict(n_periods=30, return_conf_int=True)
    logging.info("Generated temperature forecast")
    return jsonify({"forecast": forecast.tolist(), "conf_int": conf_int.tolist()})

if __name__ == '__main__':
    app.run(debug=False)  # Disable debug mode for production
