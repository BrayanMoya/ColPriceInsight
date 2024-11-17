import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib, json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Dense

# Cargar los datos

#Con menos datos (alrededor de 500)
# dfApartment = pd.read_csv('data/cleaned_apartments_cali.csv')
# dfHouse = pd.read_csv('data/cleaned_houses_cali.csv')

#Con mas datos (alrededor de 3000)
dfApartment = pd.read_csv('data/cleaned_apartments_cali1.csv')
dfHouse = pd.read_csv('data/cleaned_houses_cali1.csv')
dfMain = pd.concat([dfApartment, dfHouse], ignore_index=True)

# Convertir columnas a numérico
dfMain['Precio'] = pd.to_numeric(dfMain['Precio'], errors='coerce')
dfMain['Área'] = pd.to_numeric(dfMain['Área'], errors='coerce')
dfMain['Habitaciones'] = pd.to_numeric(dfMain['Habitaciones'], errors='coerce')
dfMain['Baños'] = pd.to_numeric(dfMain['Baños'], errors='coerce')

# Convertir 'Estrato' a categórico
dfMain['Estrato'] = dfMain['Estrato'].astype('category')

# One-Hot Encoding (Categorizar) para 'Ubicación' y 'Tipo'
dfMain = pd.get_dummies(dfMain, columns=['Ubicación', 'Tipo'], drop_first=False)

# Escalar 'Área'
scaler = StandardScaler()
dfMain[['Área']] = scaler.fit_transform(dfMain[['Área']])

# Guardar el escalador en un archivo
joblib.dump(scaler, 'scaler_precio_vivienda.pkl')
print("Escalador guardado en 'scaler_precio_vivienda.pkl'")

# Separar variables independientes y dependientes
X = dfMain.drop('Precio', axis=1)
y = np.log1p(dfMain['Precio'])  # Aplicar log(Precio + 1) para reducir la escala de precios

xColumns = list(X.columns)
with open('columnas_X.json', 'w') as f:
    json.dump(xColumns, f)

print("Columnas de X guardadas en 'columnas_X.json'")

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir una arquitectura para el modelo
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Salida con un solo valor

# Compilar el modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluar el modelo en el conjunto de prueba
loss, mae = model.evaluate(X_test, y_test)
mae_exp = np.expm1(mae)  # Revertir logaritmo para interpretar en escala original de precios
print(f'Loss en conjunto de prueba: {loss}')
print(f'MAE en conjunto de prueba (log): {mae}')
print(f'MAE en conjunto de prueba (escala original): {mae_exp}')



# plt.plot(history.history['mae'], label='MAE Entrenamiento')
# plt.plot(history.history['val_mae'], label='MAE Validación')
# plt.xlabel('Épocas')
# plt.ylabel('MAE')
# plt.legend()
# plt.show()


# Guardar el modelo en formato HDF5
# model.save('modelo_precio_vivienda.h5') #Con menos datos (alrededor de 500)

#Con mas datos (alrededor de 4300)
# model.save('modelo_precio_vivienda2.h5', include_optimizer=True) # Incluye optimizador para seguirlo entrenando
model.save('modelo_precio_vivienda2.h5')