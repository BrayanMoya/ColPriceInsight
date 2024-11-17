import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Cargar los datos

#Con menos datos (alrededor de 500)
# dfApartment = pd.read_csv('data/cleaned_apartments_cali.csv')
# dfHouse = pd.read_csv('data/cleaned_houses_cali.csv')

#Con mas datos (alrededor de 3000)
dfApartment = pd.read_csv('data/cleaned_apartments_cali1.csv')
dfHouse = pd.read_csv('data/cleaned_houses_cali1.csv')

# Unir datos
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



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Distribución del Precio
plt.hist(dfMain['Precio'], bins=30)
plt.xlabel('Precio')
plt.ylabel('Frecuencia')
plt.title('Distribución del Precio')
plt.show()
plt.close()

#Relación entre Área y Precio
plt.scatter(dfMain['Área'], dfMain['Precio'])
plt.xlabel('Área')
plt.ylabel('Precio')
plt.title('Relación entre Área y Precio')
plt.show()
plt.close()

#Relación entre Habitaciones y Precio
plt.scatter(dfMain['Habitaciones'], dfMain['Precio'], color='green')
plt.xlabel('Habitaciones')
plt.ylabel('Precio')
plt.title('Relación entre Habitaciones y Precio')
plt.show()
plt.close()

#Relación entre Baños y Precio
plt.scatter(dfMain['Baños'], dfMain['Precio'])
plt.xlabel('Baños')
plt.ylabel('Precio')
plt.title('Relación entre Baños y Precio')
plt.show()
plt.close()

# Verificar las primeras filas
# print(dfMain['Tipo_Casa'])
# print(dfMain.head())
# print(dfMain.info())
# print(dfMain.describe())


# print(dfMain.isnull().sum())
# print(X)
# print(y)
# print(X_train, X_test, y_train, y_test)


correlation_matrix = dfMain.corr()
print(correlation_matrix)