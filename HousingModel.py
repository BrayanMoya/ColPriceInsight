import joblib, json
import numpy as np
import pandas as pd
from keras.api.models import load_model
from keras.api.losses import MeanSquaredError

# Cargar el modelo guardado
model = load_model('modelo_precio_vivienda2.h5', compile=False)
model.compile(optimizer='adam', loss=MeanSquaredError())
print("Modelo cargado correctamente.")

# Cargar el escalador desde el archivo
scaler = joblib.load('scaler_precio_vivienda.pkl')
print("Escalador cargado correctamente.")

# Función para redondear precios
def redondearPrecio(precio):
    potencia = 10 ** (len(str(int(precio))) - 3)
    return int(np.ceil(precio / potencia) * potencia)

def predecirPrecio(area, habitaciones, baños, estrato, ubicacion, tipo):
    # DataFrame temporal con las mismas columnas que el conjunto de entrenamiento
    datos = pd.DataFrame({
        'Área': [area],
        'Habitaciones': [habitaciones],
        'Baños': [baños],
        'Estrato': [estrato],
        'Ubicación': [ubicacion],
        'Tipo': [tipo]
    })

    # Escalar el área usando el escalador cargado
    datos[['Área']] = scaler.transform(datos[['Área']])

    # Codificar variables categóricas con One-Hot Encoding
    datos = pd.get_dummies(datos, columns=['Ubicación', 'Tipo'], drop_first=False)

    # Cargar las columnas de X
    with open('columnas_X.json', 'r', encoding='utf-8') as f:
        xColumns = json.load(f)

    # Asegurar que todas las columnas necesarias están presentes y en el orden correcto
    datos = datos.reindex(columns=xColumns, fill_value=0)

    # Predicción
    prediccion_log = model.predict(datos)
    prediccion = np.expm1(prediccion_log)  # Revertir el logaritmo para la escala original
    return redondearPrecio(prediccion[0][0])  # Redondear el precio final

if __name__ == "__main__":
    print("=== Predicción de Precio de Vivienda ===")
    try:
        # Solicitar atributos al usuario
        area = float(input("Ingrese el área de la vivienda (En m²): "))
        habitaciones = int(input("Ingrese el número de habitaciones: "))
        baños = int(input("Ingrese el número de baños: "))
        estrato = int(input("Ingrese el estrato social (1-6): "))
        ubicacion = input("Ingrese la ubicación (ejemplo: 'Ciudad 2000'): ")
        tipo = input("Ingrese el tipo de inmueble ('Casa' o 'Apartamento'): ")

        # Realizar predicción
        precio = predecirPrecio(area, habitaciones, baños, estrato, ubicacion, tipo)
        print(f"El precio estimado de la vivienda es: ${precio:,}")
    except Exception as e:
        print(f"Error en la predicción: {e}")