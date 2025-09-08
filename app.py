from flask import Flask, render_template, request
import numpy as np
from Motor_Maintainance import give_pred
app = Flask(__name__)
@app.route("/")
@app.route("/Hello")
def home():
    default_values = {
        "current": "",
        "voltage": "",
        "temperature": "",
        "humidity": "",
        "vibration": ""
    }
    return render_template("Motor.html", **default_values)
@app.route("/result", methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        outpt = request.form.to_dict()
        current = outpt.get("current", "")
        voltage = outpt.get("voltage", "")
        temperature = outpt.get("temperature", "")
        humidity = outpt.get("humidity", "")
        vibration = outpt.get("vibration", "")
        test = np.array([[current, voltage, temperature, humidity, vibration]])
        results = give_pred(test)

        return render_template("Motor.html", current=current, voltage=voltage,
                               temperature=temperature, humidity=humidity,
                               vibration=vibration, name=results)
    else:
        return "Method not allowed", 405
if __name__ == '__main__':
    app.run(debug=True, port=5000)
