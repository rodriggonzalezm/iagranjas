from flask import Flask, render_template
import requests
import pandas as pd
import os

app = Flask(__name__)

MONTOS_INICIALES_USD = 600

TOKENS_RELEVANTES = ["usdc", "usdt", "eth", "dai", "link", "curve", "ethena", "lybra", "pendle", "aerodrome", "crv"]

def obtener_pools():
    url = "https://yields.llama.fi/pools"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json().get("data", [])
        return data
    except Exception as e:
        return []

def preparar_datos(data):
    pools = []
    for pool in data:
        apy = pool.get("apy", 0)
        tvl = pool.get("tvlUsd", 0)
        project = pool.get("project", "")
        chain = pool.get("chain", "")
        pool_name = pool.get("pool", "").lower()
        url = pool.get("url", "")
        if apy and apy > 0 and tvl and tvl > 1000:
            ganancia_mensual = (apy * MONTOS_INICIALES_USD) / 12
            stable = any(stable in pool_name for stable in ["usdc", "usdt", "dai"])
            pools.append({
                "Pool": pool_name,
                "Protocolo": project,
                "Red": chain,
                "APY (%)": round(apy * 100, 2),
                "TVL (USD)": round(tvl, 2),
                "URL": url,
                "Ganancia Estimada/mes (USD)": round(ganancia_mensual, 2),
                "Stablecoin": stable,
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
    df["Etiqueta IA"] = labels
    return df

@app.route('/')
def index():
    data = obtener_pools()
    if not data:
        return "<h3>Error obteniendo datos de la API</h3>"
    
    df = preparar_datos(data)
    df = etiquetar_pools(df)
    df = df.sort_values(by="Ganancia Estimada/mes (USD)", ascending=False)
    # Links clickeables
    df["URL"] = df["URL"].apply(lambda x: f'<a href="{x}" target="_blank">Link</a>')
    
    return render_template("index.html", tables=[df.to_html(classes='table table-striped table-hover', escape=False, index=False)])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
