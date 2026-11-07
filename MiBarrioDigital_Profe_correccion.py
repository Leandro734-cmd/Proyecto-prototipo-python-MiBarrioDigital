import psycopg2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib 

conexion = psycopg2.connect(
    host="localhost",
    database="MiBarrioDigital",
    user="postgres",
    password="1234",
    port="5432"
)

print ("Conexión exitosa a la base de datos MiBarrioDigital")

query ="""
SELECT 
    clientes.nombre AS nombre_cliente, 
    ventas.cantidad, 
    productos.nombre AS nombre_producto, 
    productos.precio_venta, 
    productos.precio_original, 
    ventas.monto
FROM clientes
INNER JOIN ventas
    ON clientes.id_cliente = ventas.id_cliente
INNER JOIN productos
    ON ventas.id_producto = productos.id_producto;
"""
df = pd.read_sql(query, conexion)

conexion.close()

df['ganancias'] = df['monto'] - (df['precio_original'] * df['cantidad'])
df['alta_demanda'] = df['cantidad'].apply(lambda x: 1 if x >= 3 else 0)

df['id_producto_num'] = df['nombre_producto'].astype('category').cat.codes

X = df[['id_producto_num', 'precio_venta', 'precio_original', 'monto', 'ganancias']]
Y = df['alta_demanda'] 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

modelo = DecisionTreeClassifier()
modelo.fit(X_train, Y_train)

predicciones = modelo.predict(X_test)
precision = accuracy_score(Y_test, predicciones)

print(f"Precisión del modelo: {precision * 100}%")


print("\n---SIMULADOR DE DEMANDA EN VIVO ---")


venta_nueva = pd.DataFrame({
    'id_producto_num': [2],          
    'precio_venta': [15000],
    'precio_original': [10000],
    'monto': [30000],                
    'ganancias': [10000]             
})


venta_nueva['prediccion'] = modelo.predict(venta_nueva)


venta_nueva['decision_final'] = venta_nueva['prediccion'].apply(
    lambda x: "🔥 ALTA DEMANDA (Pedir más al proveedor)" if x == 1 else "✅ DEMANDA NORMAL"
)

print(venta_nueva[['ganancias', 'decision_final']])


joblib.dump(modelo, 'modelo_mibarriodigital.pkl')
print("\n📦 ¡Modelo exportado con éxito a 'modelo_mibarriodigital.pkl'!")