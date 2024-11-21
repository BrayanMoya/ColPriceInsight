import joblib, json
import tkinter as tk
import numpy as np
import pandas as pd
from keras.api.models import load_model
from keras.api.losses import MeanSquaredError
from tkinter import ttk, messagebox

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

def realizarPrediccion():
    try:
        # Recuperar los valores
        area = float(areaVar.get()) if areaVar.get() else None
        habitaciones = int(habitacionesVar.get()) if habitacionesVar.get() else None
        baños = int(bañosVar.get()) if bañosVar.get() else None
        estrato = int(estratoVar.get()) if estratoVar.get() else None
        tipo = tipoVar.get()

        # Realizar la predicción
        precio = predecirPrecioOpcional(area, habitaciones, baños, estrato, tipo)
        resultadoLabel.config(text=f"Precio estimado: ${precio['redondeado']:,}")
    except ValueError as ve:
        messagebox.showerror("Entrada inválida", f"Entrada no válida: {ve}")
    except Exception as e:
        messagebox.showerror("Error", f"Error en la predicción: {e}")

# Configuración de la ventana principal
ventana = tk.Tk()
ventana.title("Predicción de Precios de Viviendas en Cali")
ventana.geometry("400x450")
ventana.resizable(True, True)

# Configuración de campos
areaVar = tk.StringVar()
habitacionesVar = tk.StringVar()
bañosVar = tk.StringVar()
estratoVar = tk.StringVar()
tipoVar = tk.StringVar()

# Widgets
ttk.Label(ventana, text="Predicción de Precios de Viviendas", font=("Helvetica", 16)).pack(pady=10)

ttk.Label(ventana, text="Tipo de Inmueble:").pack(anchor="w", padx=20)
ttk.Combobox(ventana, textvariable=tipoVar, values=["Casa", "Apartamento"], state="readonly").pack(fill="x", padx=20, pady=5)

ttk.Label(ventana, text="Área (m²) [Opcional]:").pack(anchor="w", padx=20)
ttk.Entry(ventana, textvariable=areaVar).pack(fill="x", padx=20, pady=5)

ttk.Label(ventana, text="Número de Habitaciones [Opcional]:").pack(anchor="w", padx=20)
ttk.Entry(ventana, textvariable=habitacionesVar).pack(fill="x", padx=20, pady=5)

ttk.Label(ventana, text="Número de Baños [Opcional]:").pack(anchor="w", padx=20)
ttk.Entry(ventana, textvariable=bañosVar).pack(fill="x", padx=20, pady=5)

ttk.Label(ventana, text="Estrato Social (1-6) [Opcional]:").pack(anchor="w", padx=20)
ttk.Entry(ventana, textvariable=estratoVar).pack(fill="x", padx=20, pady=5)

ttk.Button(ventana, text="Predecir Precio", command=realizarPrediccion).pack(pady=20)

resultadoLabel = ttk.Label(ventana, text="", font=("Helvetica", 12))
resultadoLabel.pack()

# Iniciar la aplicación
ventana.mainloop()