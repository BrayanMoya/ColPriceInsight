import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Dense


# Reemplaza 'ruta_al_dataset.csv' con la ruta real al archivo de datos
dfApartment = pd.read_csv('data/cleaned_apartments_cali.csv')
dfHouse = pd.read_csv('data/cleaned_houses_cali.csv')
dfMain = pd.concat([dfApartment, dfHouse], ignore_index=True)

#'Precio', 'Área total', 'Habitaciones' y 'Baños' a numérico
dfMain['Precio'] = pd.to_numeric(dfMain['Precio'], errors='coerce')
dfMain['Área'] = pd.to_numeric(dfMain['Área'], errors='coerce')
dfMain['Habitaciones'] = pd.to_numeric(dfMain['Habitaciones'], errors='coerce')
dfMain['Baños'] = pd.to_numeric(dfMain['Baños'], errors='coerce')

# Convertir 'Estrato' a categórico
dfMain['Estrato'] = dfMain['Estrato'].astype('category')

#Categorizar 'Ubicación' y 'Tipo' (One-Hot Encoding)
dfMain = pd.get_dummies(dfMain, columns=['Ubicación', 'Tipo'], drop_first=False)

#Escalar 'Área'
scaler = StandardScaler()
dfMain[['Área']] = scaler.fit_transform(dfMain[['Área']])

X = dfMain.drop('Precio', axis=1)
y = dfMain['Precio']


#Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Salida con un solo valor: el precio

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

loss, mae = model.evaluate(X_test, y_test)
print(f'MAE en conjunto de prueba: {mae}')
