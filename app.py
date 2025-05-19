import os
import requests
import pandas as pd
from flask import Flask, render_template_string
import logging
from telegram import Bot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from apscheduler.schedulers.background import BackgroundScheduler
import lightgbm as lgb
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MONTO_INICIAL_USD = float(os.getenv("MONTO_INICIAL_USD", "600"))
NUM_ALERTAS = int(os.getenv("NUM_ALERTAS", "5"))
GANANCIA_MINIMA_MENSUAL = float(os.getenv("GANANCIA_MINIMA_MENSUAL", "20"))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None

TOKENS_RELEVANTES = [
    "usdc", "usdt", "eth", "dai", "link",
    "curve", "ethena", "lybra", "pendle",
    "aerodrome", "crv"
]

app = Flask(__name__)

# Variable global para cachear resultados (evitar leer disco cada request)
cache_resultados_html = "<p>No hay datos disponibles.</p>"

def obtener_pools():
    url = "https://yields.llama.fi/pools"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json().get("data", [])
        logging.info(f"Pools descargados: {len(data)}")
        return data
    except Exception as e:
        logging.error(f"Error obteniendo pools: {e}")
        return []

def preparar_datos(data):
    pools = []
    for pool in data:
        apy = pool.get("apy", 0)
        tvl = pool.get("tvlUsd", 0)
        pool_name = pool.get("pool", "").lower()
        if apy and apy > 0 and tvl and tvl > 1000:
            ganancia_mes = (apy * MONTO_INICIAL_USD) / 12
            if ganancia_mes < GANANCIA_MINIMA_MENSUAL:
                continue
            stable = any(s in pool_name for s in ["usdc", "usdt", "dai"])
            pools.append({
                "Pool": pool_name,
                "Protocolo": pool.get("project", ""),
                "Red": pool.get("chain", ""),
                "APY": apy,
                "APY_pct": apy * 100,
                "TVL": tvl,
                "URL": pool.get("url", ""),
                "GananciaMes": ganancia_mes,
                "Stablecoin": stable,
            })
    df = pd.DataFrame(pools)
    logging.info(f"Pools filtrados (ganancia mínima {GANANCIA_MINIMA_MENSUAL}$): {len(df)}")
    return df

def crear_label(row):
    apy = row["APY"]
    tvl = row["TVL"]
    pool_name = row["Pool"]
    if apy >= 0.18 and tvl > 200000 and any(token in pool_name for token in TOKENS_RELEVANTES):
        return "Excelente"
    elif 0.10 <= apy < 0.18:
        return "Bueno"
    else:
        return "Evitar"

def preparar_features_y_labels(df):
    df = df.copy()
    df["label"] = df.apply(crear_label, axis=1)
    df["label"] = df["label"].astype("category")
    df["log_TVL"] = np.log1p(df["TVL"])
    df["apy_tvl_interaction"] = df["APY"] * df["log_TVL"]
    df["pool_len"] = df["Pool"].apply(len)
    df["stablecoin_int"] = df["Stablecoin"].astype(int)

    features = ["APY", "log_TVL", "apy_tvl_interaction", "pool_len", "stablecoin_int", "Protocolo", "Red"]
    X = df[features]
    y = df["label"]
    return X, y

def entrenar_modelo(X, y):
    cat_cols = ["Protocolo", "Red"]
    num_cols = [col for col in X.columns if col not in cat_cols]

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

    model = Pipeline([
        ('preproc', preprocessor),
        ('clf', lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42))
    ])

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred)
    logging.info(f"Reporte de clasificación:\n{report}")

    scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
    logging.info(f"F1 Macro CV Score: {scores.mean():.3f} ± {scores.std():.3f}")

    return model

def predecir(model, df):
    df_pred = df.copy()
    df_pred["log_TVL"] = np.log1p(df_pred["TVL"])
    df_pred["apy_tvl_interaction"] = df_pred["APY"] * df_pred["log_TVL"]
    df_pred["pool_len"] = df_pred["Pool"].apply(len)
    df_pred["stablecoin_int"] = df_pred["Stablecoin"].astype(int)

    features = ["APY", "log_TVL", "apy_tvl_interaction", "pool_len", "stablecoin_int", "Protocolo", "Red"]
    X_pred = df_pred[features]

    preds = model.predict(X_pred)
    df_pred["Predicción IA"] = preds
    df_pred = df_pred[(df_pred["Predicción IA"] == "Excelente") & (df_pred["GananciaMes"] >= GANANCIA_MINIMA_MENSUAL)]
    return df_pred

def armar_mensaje(df):
    top = df.sort_values(by="GananciaMes", ascending=False).head(NUM_ALERTAS)
    if top.empty:
        return "No se encontraron pools *Excelente* con ganancia mensual mínima hoy."

    mensaje = "🔥 *Top Pools recomendados por IA* 🔥\n\n"
    for _, row in top.iterrows():
        mensaje += f"*{row['Pool']}*\n"
        mensaje += f"Protocolo: {row['Protocolo']}\n"
        mensaje += f"Red: {row['Red']}\n"
        mensaje += f"APY: {row['APY_pct']:.2f}%\n"
        mensaje += f"TVL: ${row['TVL']:,.0f}\n"
        mensaje += f"Ganancia estimada/mes: ${row['GananciaMes']:.2f}\n"
        mensaje += f"[Más info]({row['URL']})\n\n"
    mensaje += "🧠 *Análisis automatizado con IA. Evalúa riesgos antes de invertir.*"
    return mensaje

def enviar_alerta_telegram(mensaje):
    if bot and TELEGRAM_CHAT_ID:
        try:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=mensaje, parse_mode="Markdown", disable_web_page_preview=True)
            logging.info("Mensaje enviado a Telegram")
        except Exception as e:
            logging.error(f"Error enviando mensaje Telegram: {e}")
    else:
        logging.warning("Bot o chat ID no configurados; no se envía mensaje")

def actualizar_cache(df_pred):
    global cache_resultados_html
    if df_pred.empty:
        cache_resultados_html = "<p>No se encontraron pools recomendados hoy.</p>"
    else:
        df_pred_sorted = df_pred.sort_values(by="GananciaMes", ascending=False)
        cache_resultados_html = df_pred_sorted.to_html(
            classes='table table-striped table-hover',
            index=False,
            justify='center',
            border=0,
            escape=False
        )
    logging.info("Cache HTML actualizado")

def job_analisis():
    logging.info("Inicio análisis")
    data = obtener_pools()
    if not data:
        logging.warning("No data to analyze")
        actualizar_cache(pd.DataFrame())  # vaciar cache
        return None
    df = preparar_datos(data)
    if df.empty:
        logging.warning("No hay pools que cumplan con ganancia mínima")
        actualizar_cache(pd.DataFrame())
        return None
    X, y = preparar_features_y_labels(df)
    model = entrenar_modelo(X, y)
    df_pred = predecir(model, df)
    actualizar_cache(df_pred)
    mensaje = armar_mensaje(df_pred)
    enviar_alerta_telegram(mensaje)
    logging.info("Análisis terminado")
    return df_pred

@app.route("/")
def index():
    global cache_resultados_html
    template = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Dashboard Pools IA</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" />
    </head>
    <body class="bg-light">
        <div class="container my-4">
            <h1 class="mb-4 text-center">Pools recomendados por IA</h1>
            <div class="table-responsive">
                {{ resultados|safe }}
            </div>
            <footer class="text-center mt-4">
                <small>Actualizado cada 6 horas automáticamente</small>
            </footer>
        </div>
    </body>
    </html>
    """
    return render_template_string(template, resultados=cache_resultados_html)

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    # Ejecuta la función job_analisis cada 6 horas
    scheduler.add_job(job_analisis, 'interval', hours=6, next_run_time=None)
    scheduler.start()

    # Ejecuta el análisis al iniciar el servidor para tener datos cacheados
    job_analisis()

    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
