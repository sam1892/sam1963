from flask import Flask, render_template, request, redirect, url_for
from flask_mail import Mail, Message
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

app = Flask(__name__)
app.secret_key = "super_secret_key"

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)

# Flask-Mail setup
app.config.update(
    MAIL_SERVER="smtp.gmail.com",
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME="your_email@gmail.com",  # Replace with your email
    MAIL_PASSWORD="your_email_password",  # Replace with your email password
)
mail = Mail(app)

# In-memory data store
data_store = []
thresholds = {"vibration": 0.004, "temperature": 85}

# List of technicians and their schedules
technicians = [
    {"name": "John Doe", "schedule": "Mon-Fri, 8am-5pm"},
    {"name": "Jane Smith", "schedule": "Tue-Sat, 9am-6pm"},
    {"name": "Mike Johnson", "schedule": "Wed-Sun, 7am-4pm"},
]

# Turbine info with random Texas locations
turbine_info = [
    {"id": "1", "name": "Turbine A", "location": "Austin, TX", "status": "Operational", "lastInspection": "2024-11-01", "nextMaintenance": "2024-12-15"},
    {"id": "2", "name": "Turbine B", "location": "Dallas, TX", "status": "Operational", "lastInspection": "2024-11-10", "nextMaintenance": "2024-12-20"},
    {"id": "3", "name": "Turbine C", "location": "Houston, TX", "status": "Under Maintenance", "lastInspection": "2024-11-15", "nextMaintenance": "2024-12-25"},
]

class User(UserMixin):
    pass

@login_manager.user_loader
def load_user(user_id):
    user = User()
    user.id = user_id
    return user

sample_user = {"id": "admin", "password": "admin"}

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == sample_user["id"] and password == sample_user["password"]:
            user = User()
            user.id = username
            login_user(user)
            return redirect(url_for("dashboard"))
        else:
            return "Invalid credentials", 401
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/ingest-data", methods=["POST"])
def ingest_data():
    data = request.get_json()
    data_store.append(data)
    return "Data received", 200

@app.route("/dashboard")
@login_required
def dashboard():
    alerts, predictions = [], []
    repair_cost, efficiency_metrics, improvement_metrics = 0, [], {}

    if data_store:
        df = pd.DataFrame(data_store)

        # Maintenance Alerts
        high_vibration = df[df['vibration'] > thresholds["vibration"]]
        high_temperature = df[df['temperature'] > thresholds["temperature"]]

        for _, row in high_vibration.iterrows():
            alert = f"High vibration at {row['timestamp']} - {row['vibration']}"
            alerts.append({"alert": alert, "technician": technicians[0]["name"]})
        for _, row in high_temperature.iterrows():
            alert = f"High temperature at {row['timestamp']} - {row['temperature']}°F"
            alerts.append({"alert": alert, "technician": technicians[1]["name"]})

        # Cost Analysis
        repair_cost = df['failure'].sum() * 5000

        # Efficiency Metrics
        df['efficiency'] = (df['wind_speed'] * 10) / 100
        average_efficiency = df['efficiency'].mean()
        efficiency_metrics.append(f"Average Efficiency: {average_efficiency:.2f}%")

        # Improvement Metrics
        if len(df) > 10:
            recent_data = df.tail(10)
            older_data = df.iloc[:-10]
            improvement_metrics['vibration'] = older_data['vibration'].mean() - recent_data['vibration'].mean()
            improvement_metrics['temperature'] = older_data['temperature'].mean() - recent_data['temperature'].mean()
            improvement_metrics['efficiency'] = recent_data['efficiency'].mean() - older_data['efficiency'].mean()

        # Predictions
        if len(df) > 10:
            X = df[['temperature', 'vibration', 'wind_speed']]
            y = df['failure']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions = [
                f"Prediction for data point {i}: {'Failure' if pred else 'Operational'}"
                for i, pred in enumerate(y_pred)
            ]

        # Generate Graphs
        generate_graphs(df, improvement_metrics)

    return render_template(
        "dashboard.html",
        alerts=alerts,
        predictions=predictions,
        repair_cost=repair_cost,
        efficiency_metrics=efficiency_metrics,
        improvement_metrics=improvement_metrics,
        turbine_info=turbine_info,
        technicians=technicians,
        heatmap="/static/heatmap.png",
        scatter="/static/scatter.png",
        bar_chart="/static/bar_chart.png",
        pie_chart="/static/pie_chart.png",
        efficiency_line="/static/efficiency_line.png",
        improvement_bar="/static/improvement_bar.png",
    )

def generate_graphs(df, improvement_metrics):
    # Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[['temperature', 'vibration', 'wind_speed']].corr(), annot=True, cmap="coolwarm")
    plt.savefig("static/heatmap.png")
    plt.close()

    # Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="wind_speed", y="temperature", hue="failure", style="failure")
    plt.savefig("static/scatter.png")
    plt.close()

    # Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="wind_speed", y="temperature", hue="failure")
    plt.savefig("static/bar_chart.png")
    plt.close()

    # Pie Chart
    failure_counts = df['failure'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(failure_counts, labels=["Operational", "Failed"], autopct='%1.1f%%')
    plt.savefig("static/pie_chart.png")
    plt.close()

    # Line Chart
    plt.figure(figsize=(10, 6))
    df['efficiency'].plot(kind='line', title="Turbine Efficiency Over Time")
    plt.savefig("static/efficiency_line.png")
    plt.close()

    # Improvement Metrics Bar Chart
    if improvement_metrics:
        improvement_df = pd.DataFrame([improvement_metrics])
        plt.figure(figsize=(10, 6))
        improvement_df.plot(kind='bar')
        plt.savefig("static/improvement_bar.png")
        plt.close()

if __name__ == "__main__":
    app.run(debug=True)
