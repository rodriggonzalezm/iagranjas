from flask import Flask, render_template
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

TOKENS_RELEVANTES = ["usdc", "usdt", "eth", "dai", "link", "curve", "ethena", "lybra", "pendle", "aerodrome", "crv"]

def obtener_pools():
    url = "https://yields.llama.fi/pools"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json().get("data", [])
        return data
    except Exception as e:
        print(f"Error descargando datos: {e}")
        return []

def preparar_datos(data):
    pools = []
    monto_inicial_usd = 600
    for pool in data:
        apy = pool.get("apy", 0)
        tvl = pool.get("tvlUsd", 0)
        pool_name = pool.get("pool", "").lower()
        url = pool.get("url", "")
        if apy > 0 and tvl > 1000:
            ganancia_mensual = (apy * monto_inicial_usd) / 12
            pools.append({
                "Pool": pool_name,
                "Protocolo": pool.get("project", ""),
                "Red": pool.get("chain", ""),
                "APY (%)": apy * 100,
                "TVL (USD)": tvl,
                "URL": url,
                "Ganancia Estimada/mes (USD)": ganancia_mensual,
            })
    df = pd.DataFrame(pools)
    return df

def etiquetar_pools(df):
    labels = []
    for _, row in df.iterrows():
        apy = row["APY (%)"] / 100
        tvl = row["TVL (USD)"]
        pool_name = row["Pool"]
        if apy >= 0.12 and tvl > 100000 and any(token in pool_name for token in TOKENS_RELEVANTES):
            labels.append("Excelente")
        elif 0.06 <= apy < 0.12:
            labels.append("Bueno")
        else:
            labels.append("Evitar")
    df["Recomendación"] = labels
    return df

def entrenar_modelo(df):
    le_protocolo = LabelEncoder()
    le_red = LabelEncoder()

    df["protocolo_encoded"] = le_protocolo.fit_transform(df["Protocolo"].str.lower())
    df["red_encoded"] = le_red.fit_transform(df["Red"].str.lower())

    X = df[["APY (%)", "TVL (USD)", "protocolo_encoded", "red_encoded"]]
    y = df["Recomendación"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le_protocolo, le_red

def predecir(df, model, le_protocolo, le_red):
    df["protocolo_encoded"] = le_protocolo.transform(df["Protocolo"].str.lower())
    df["red_encoded"] = le_red.transform(df["Red"].str.lower())

    X_new = df[["APY (%)", "TVL (USD)", "protocolo_encoded", "red_encoded"]]
    df["Predicción IA"] = model.predict(X_new)
    return df

@app.route("/")
def index():
    data = obtener_pools()
    if not data:
        return "<h3>Error descargando datos.</h3>"
    
    df = preparar_datos(data)
    df = etiquetar_pools(df)

    model, le_protocolo, le_red = entrenar_modelo(df)
    df = predecir(df, model, le_protocolo, le_red)

    # Mostrar solo top 5 "Excelente"
    df_top = df[df["Predicción IA"] == "Excelente"].sort_values(by="Ganancia Estimada/mes (USD)", ascending=False).head(5)
    if df_top.empty:
        html_table = "<h3>No se encontraron pools recomendados por IA hoy.</h3>"
    else:
        html_table = df_top.to_html(classes='table table-striped', index=False, escape=False)
    
    return render_template("index.html", tables=[html_table])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
