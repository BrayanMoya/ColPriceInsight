import joblib, json
import numpy as np
import pandas as pd
from keras.api.models import load_model
from keras.api.losses import MeanSquaredError

# Cargar el modelo guardado
model = load_model('modelo_precio_vivienda3.h5', compile=False)
model.compile(optimizer='adam', loss=MeanSquaredError())
print("Modelo cargado correctamente.")

# Cargar el escalador desde el archivo
scaler = joblib.load('scaler_precio_vivienda.pkl')
print("Escalador cargado correctamente.")

# Función para redondear precios
def redondearPrecio(precio):
    potencia = 10 ** (len(str(int(precio))) - 3)
    return int(np.ceil(precio / potencia) * potencia)

def obtenerValoresPredeterminados():
    # Datos con ubicación
    # dfApartment = pd.read_csv('data/cleaned_apartments_cali1.csv')
    # dfHouse = pd.read_csv('data/cleaned_houses_cali1.csv')

    # Datos sin ubicación
    dfApartment = pd.read_csv('data/cleaned_apartments_cali2.csv')
    dfHouse = pd.read_csv('data/cleaned_houses_cali2.csv')
    dfMain = pd.concat([dfApartment, dfHouse], ignore_index=True)

    valoresPredeterminados = {
        'Área': dfMain['Área'].median(),
        'Habitaciones': dfMain['Habitaciones'].mode()[0],
        'Baños': dfMain['Baños'].mode()[0],
        'Estrato': dfMain['Estrato'].mode()[0],
        'Tipo': dfMain['Tipo'].mode()[0]
    }
    return valoresPredeterminados

def predecirPrecioOpcional(area=None, habitaciones=None, baños=None, estrato=None, tipo=None):
    valoresPredeterminados = obtenerValoresPredeterminados()

    area = area if area is not None else valoresPredeterminados['Área']
    habitaciones = habitaciones if habitaciones is not None else valoresPredeterminados['Habitaciones']
    baños = baños if baños is not None else valoresPredeterminados['Baños']
    estrato = estrato if estrato is not None else valoresPredeterminados['Estrato']
    tipo = tipo if tipo is not None else valoresPredeterminados['Tipo']

    datos = pd.DataFrame({
        'Área': [area],
        'Habitaciones': [habitaciones],
        'Baños': [baños],
        'Estrato': [estrato],
        'Tipo': [tipo]
    })

    # Escalar el área usando el escalador cargado
    datos[['Área']] = scaler.transform(datos[['Área']])

    # Codificar variables categóricas con One-Hot Encoding
    datos = pd.get_dummies(datos, columns=['Tipo'], drop_first=False)

    # Cargar las columnas de X
    with open('columnas_X.json', 'r', encoding='utf-8') as f:
        xColumns = json.load(f)

    # Asegurar que todas las columnas necesarias están presentes y en el orden correcto
    datos = datos.reindex(columns=xColumns, fill_value=0)

    # Predicción
    prediccionLog = model.predict(datos)
    prediccion = np.expm1(prediccionLog)  # Revertir el logaritmo para la escala original

    precios = {
        'original': prediccion[0][0],
        'redondeado': redondearPrecio(prediccion[0][0])
    }

    return precios  # Redondear el precio final

if __name__ == "__main__":
    print("=== Predicción de Precio de Vivienda con Atributos Opcionales ===\nIngrese como mínimo 3 atributos (El tipo de inmueble es obligatorio)")
    try:
        tipo = input("Ingrese el tipo de inmueble ('Casa' o 'Apartamento'): ")
        tipo = tipo if tipo else None

        area = input("Ingrese el área de la vivienda (m²) [Opcional]: ")
        area = float(area) if area else None

        habitaciones = input("Ingrese el número de habitaciones [Opcional]: ")
        habitaciones = int(habitaciones) if habitaciones else None

        baños = input("Ingrese el número de baños [Opcional]: ")
        baños = int(baños) if baños else None

        estrato = input("Ingrese el estrato social (1-6) [Opcional]: ")
        estrato = int(estrato) if estrato else None

        precio = predecirPrecioOpcional(area, habitaciones, baños, estrato, tipo)
        # print(f"El precio estimado de la vivienda es (original): ${precio['original']:,}")
        print(f"El precio estimado de la vivienda es: ${precio['redondeado']:,}")
    except Exception as e:
        print(f"Error en la predicción: {e}")