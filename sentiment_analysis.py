import re
import zipfile
from pathlib import Path
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ==============================
# CONFIGURACIÓN INICIAL
# ==============================
nltk.data.path.append(r'C:\Users\rburc\AppData\Roaming\nltk_data')#cambiar según tu ruta
# Asegurarse de que los recursos necesarios están descargados
for recurso in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f"corpora/{recurso}")
    except LookupError:
        nltk.download(recurso)

stop_words = set(stopwords.words('english'))# puedes agregar más stopwords si lo deseas
lemmatizer = WordNetLemmatizer()# inicializa el lematizador
tqdm.pandas()# para mostrar barras de progreso en pandas

# ==============================
# FUNCIONES
# ==============================
# Extraer archivos del ZIP
def extraer_zip():
    zip_path = "archive.zip"
    extract_path = Path("extracted_files")
    if not extract_path.exists():
        print("Creando directorio y extrayendo ZIP...")
        extract_path.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Archivos extraídos.")
    else:
        print("Archivos ya existentes.")

# Limpiar y preprocesar texto
def limpiar_texto(texto):
    try:
        texto = texto.lower()# pasa a minúsculas
        texto = re.sub(r"<[^>]+>", " ", texto)  # quita etiquetas HTML
        texto = re.sub(r"&nbsp;|&amp;", " ", texto)  # limpia entidades HTML
        texto = re.sub(r"'", " ", texto)# quita apóstrofes
        texto = re.sub(r"[^a-z]", " ", texto)# quita caracteres no alfabéticos
        texto = re.sub(r"\s+", " ", texto).strip()# quita espacios extra
        palabras = word_tokenize(texto)# tokeniza el texto
        palabras = [p for p in palabras if p not in stop_words]# quita stopwords
        palabras = [lemmatizer.lemmatize(p) for p in palabras]# lematiza las palabras
        return " ".join(palabras)# une las palabras de nuevo en un string
    except:
        return ""

# Normalizar dataset
def normalizar_dataset(path_entrada, path_salida):
    if not Path(path_salida).exists():
        print("Normalizando dataset...")
        df = pd.read_csv(path_entrada)
        df['review'] = df['review'].astype(str).fillna('')
        df['review_limpia'] = df['review'].progress_apply(limpiar_texto)
        # Selecciona solo las columnas que quieres guardar en el nuevo archivo
        df_limpio = df[['review_limpia', 'sentiment']]
        # Guarda el DataFrame limpio
        df_limpio.to_csv(path_salida, index=False)
        print(f"Dataset normalizado guardado en {path_salida}")
    else:
        print("Dataset limpio ya existe.")

# Entrenar modelo de clasificación
def entrenar_modelo(path_datos):
    print("\nEntrenando modelo de clasificación de sentimientos...")
    df = pd.read_csv(path_datos)# carga el dataset limpio
    X = df['review_limpia']
    y = df['sentiment']

# Divide en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorización TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vect = vectorizer.fit_transform(X_train)# ajusta y transforma los datos de entrenamiento
    X_test_vect = vectorizer.transform(X_test)# transforma los datos de prueba

    modelo = MultinomialNB()# usa Naive Bayes para clasificación
    modelo.fit(X_train_vect, y_train)# entrena el modelo
    y_pred = modelo.predict(X_test_vect)# predice en el conjunto de prueba

    print("Accuracy:", accuracy_score(y_test, y_pred))# muestra la precisión
    print(classification_report(y_test, y_pred))# muestra reporte de clasificación

    joblib.dump(modelo, "modelo_sentimientos.pkl")# guarda el modelo entrenado
    joblib.dump(vectorizer, "vectorizador.pkl")# guarda el vectorizador
    print("Modelo y vectorizador guardados.")

# Predecir sentimiento de nuevo texto
def predecir_texto():
    modelo = joblib.load("modelo_sentimientos.pkl")# carga el modelo entrenado
    vectorizer = joblib.load("vectorizador.pkl")# carga el vectorizador

    nuevo_texto = input("\nEscribe una reseña para predecir su sentimiento:\n> ")# entrada del usuario
    texto_limpio = limpiar_texto(nuevo_texto)# limpia el texto de entrada
    vector = vectorizer.transform([texto_limpio])# vectoriza el texto limpio
    prediccion = modelo.predict(vector)# predice el sentimiento

    print("\nPredicción:", "Positivo" if prediccion[0] == "positive" else "Negativo")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":# solo se ejecuta si es el script principal
    # 1. Extraer datos
    extraer_zip()

    # 2. Cargar y mostrar info inicial
    csv_path = "extracted_files/IMDB Dataset.csv"# ruta del CSV
    df = pd.read_csv(csv_path, encoding="utf-8")# carga el CSV
    print(f"\nRegistros: {df.shape[0]}, Columnas: {df.shape[1]}")# muestra número de registros y columnas
    print(df.head())

    # 3. Normalizar
    normalizar_dataset(csv_path, "extracted_files/IMDB_dataset_limpiov2.csv")

    # 4. Entrenar
    entrenar_modelo("extracted_files/IMDB_dataset_limpiov2.csv")

    # 5. Probar con texto nuevo
    predecir_texto()
