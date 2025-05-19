from flask import Flask, render_template
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def index():
    try:
        df = pd.read_csv("resultados.csv")
        df = df.sort_values(by="Ganancia Estimada/mes (USD)", ascending=False)
        return render_template("index.html", tables=[df.to_html(classes='data', header="true", index=False)])
    except Exception as e:
        return f"<h3>Error cargando datos: {e}</h3>"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
