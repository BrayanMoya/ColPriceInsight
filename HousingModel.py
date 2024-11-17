import joblib, json
import numpy as np
import pandas as pd
from keras.api.models import load_model
from keras.api.losses import MeanSquaredError
from zmq import NULL


# Cargar el modelo guardado
model = load_model('modelo_precio_vivienda2.h5', compile=False)
model.compile(optimizer='adam', loss=MeanSquaredError())
print("Modelo cargado correctamente.")

# Cargar el escalador desde el archivo
scaler = joblib.load('scaler_precio_vivienda.pkl')
print("Escalador cargado correctamente.")



def predecirPrecio(area, habitaciones, baños, estrato, ubicacion, tipo):    
    # Crear un DataFrame temporal con las mismas columnas que el conjunto de entrenamiento
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
    
    # Realizar la predicción
    prediccion_log = model.predict(datos)
    prediccion = np.expm1(prediccion_log)  # Revertir el logaritmo para la escala original
    
    return prediccion[0][0]

if __name__ == "__main__":
    area = 89
    habitaciones = 4
    baños = 2
    estrato = 4
    ubicacion = 'Ciudad 2000'  # Ejemplo de ubicación
    tipo = 'Apartamento'      # Ejemplo de tipo de inmueble

    precio = predecirPrecio(area, habitaciones, baños, estrato, ubicacion, tipo)
    print(f"El precio estimado de la vivienda es: ${precio:.2f}")