import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Reemplaza 'ruta_al_dataset.csv' con la ruta real al archivo de datos
dfApartment = pd.read_csv('data/final_cleaned_mercadolibre_cali_properties_apartamento.csv')
dfHouse = pd.read_csv('data/final_cleaned_mercadolibre_cali_properties_casa.csv')
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

#Escalar 'Área', 'Habitaciones', 'Baños'
scaler = StandardScaler()
dfMain[['Área', 'Habitaciones', 'Baños']] = scaler.fit_transform(dfMain[['Área', 'Habitaciones', 'Baños']])

X = dfMain.drop('Precio', axis=1)
y = dfMain['Precio']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificar las primeras filas
# print(dfMain['Tipo_Casa'])
# print(dfMain.head())
# print(dfMain.info())
# print(dfMain.describe())
print(dfMain.isnull().sum())
print(X)
print(y)