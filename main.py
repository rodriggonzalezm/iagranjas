import os
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import logging
from telegram import Bot

# Variables de entorno
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)

# Parámetros
monto_inicial_usd = 600
NUM_ALERTAS = 5

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Lista de tokens/protocolos preferidos
TOKENS_RELEVANTES = ["usdc", "usdt", "eth", "dai", "link", "curve", "ethena", "lybra", "pendle", "aerodrome", "crv"]

def obtener_pools():
    url = "https://yields.llama.fi/pools"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json().get("data", [])
        logging.info(f"Datos descargados, {len(data)} pools obtenidos.")
        return data
    except Exception as e:
        logging.error(f"Error descargando datos: {e}")
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
            ganancia_mensual = (apy * monto_inicial_usd) / 12
            stable = any(stable in pool_name for stable in ["usdc", "usdt", "dai"])
            pools.append({
                "Pool": pool_name,
                "Protocolo": project,
                "Red": chain,
                "APY (%)": apy * 100,
                "TVL (USD)": tvl,
                "URL": url,
                "Ganancia Estimada/mes (USD)": ganancia_mensual,
                "Stablecoin": stable,
                "Score": (apy * 100) * (tvl ** 0.3)  # Nuevo score
            })
    df = pd.DataFrame(pools)
    logging.info(f"{len(df)} pools filtrados para análisis.")
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
    df["label"] = labels
    return df

def entrenar_modelo(df):
    le_protocolo = LabelEncoder()
    le_red = LabelEncoder()

    df["protocolo_encoded"] = le_protocolo.fit_transform(df["Protocolo"].str.lower())
    df["red_encoded"] = le_red.fit_transform(df["Red"].str.lower())

    X = df[["APY (%)", "TVL (USD)", "protocolo_encoded", "red_encoded"]]
    y = df["label"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le_protocolo, le_red

def predecir(df, model, le_protocolo, le_red):
    df["protocolo_encoded"] = le_protocolo.transform(df["Protocolo"].str.lower())
    df["red_encoded"] = le_red.transform(df["Red"].str.lower())

    X_new = df[["APY (%)", "TVL (USD)", "protocolo_encoded", "red_encoded"]]
    df["Predicción IA"] = model.predict(X_new)
    return df

def armar_mensaje(df):
    top = df[df["Predicción IA"] == "Excelente"].sort_values(by="Score", ascending=False).head(NUM_ALERTAS)
    if top.empty:
        return "No se encontraron pools con etiqueta Excelente hoy."

    mensaje = "🔥 *Top Pools recomendados por IA* 🔥\n\n"
    for _, row in top.iterrows():
        mensaje += f"*{row['Pool']}*\n"
        mensaje += f"Protocolo: {row['Protocolo']}\n"
        mensaje += f"Red: {row['Red']}\n"
        mensaje += f"APY: {row['APY (%)']:.2f}%\n"
        mensaje += f"TVL: ${row['TVL (USD)']:,.0f}\n"
        mensaje += f"Ganancia Estimada/mes: ${row['Ganancia Estimada/mes (USD)']:.2f}\n"
        mensaje += f"[Más info]({row['URL']})\n\n"
    mensaje += "🧠 *Análisis automatizado con IA. Evalúa riesgos antes de invertir.*"
    return mensaje

def enviar_alerta_telegram(mensaje):
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=mensaje, parse_mode="Markdown", disable_web_page_preview=True)
        logging.info("Mensaje enviado a Telegram correctamente.")
    except Exception as e:
        logging.error(f"Error enviando mensaje a Telegram: {e}")

def job_diario():
    logging.info("Inicio del análisis diario.")
    data = obtener_pools()
    if not data:
        logging.warning("No hay datos para analizar.")
        return
    df = preparar_datos(data)
    df = etiquetar_pools(df)
    model, le_protocolo, le_red = entrenar_modelo(df)
    df = predecir(df, model, le_protocolo, le_red)
    
    # Guardar resultados para dashboard
    df.to_csv("resultados.csv", index=False)
    
    mensaje = armar_mensaje(df)
    enviar_alerta_telegram(mensaje)
    logging.info("Análisis diario finalizado.")


if __name__ == "__main__":
    job_diario()
