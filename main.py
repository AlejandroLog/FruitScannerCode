import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from sklearn.ensemble import RandomForestClassifier
import joblib

# Rutas de las imágenes de entrenamiento
rutas = {
    'platano2': r'C:\Users\samue\Downloads\platano2',
    'mango': r'C:\Users\samue\Downloads\mango',
    'manzana': r'C:\Users\samue\Downloads\manzana'
}

info_frutas = {
    'platano2': {'calorias': '89 kcal', 'carbohidratos': '23g', 'precio': '$1.00 c/u', 'receta': 'Ver receta en YouTube'},
    'mango': {'calorias': '60 kcal', 'carbohidratos': '15g', 'precio': '$1.50 c/u', 'receta': 'Ver receta en YouTube'},
    'manzana': {'calorias': '95 kcal', 'carbohidratos': '25g', 'precio': '$0.80 c/u', 'receta': 'Ver receta en YouTube'}
}

def entrenar_modelo():
    X, y = [], []
    for fruta, ruta in rutas.items():
        if not os.path.exists(ruta):
            print(f"Error: La ruta {ruta} no existe.")
            continue
        print(f"Procesando carpeta: {ruta}")
        archivos = os.listdir(ruta)
        print(f"Archivos encontrados en {ruta}: {archivos}")
        for archivo in archivos:
            if archivo.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(ruta, archivo)
                print(f"Intentando cargar: {img_path}")
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Error: No se pudo cargar la imagen {img_path}")
                        continue
                    img = cv2.resize(img, (100, 100))  # Redimensionar para consistencia
                    # Extraer características (promedio de color RGB)
                    avg_color = np.mean(img, axis=(0, 1))
                    print(f"Características extraídas de {img_path}: {avg_color}")
                    X.append(avg_color)
                    y.append(fruta)
                except Exception as e:
                    print(f"Error procesando {img_path}: {e}")
    
    if not X:
        raise ValueError("No se encontraron imágenes válidas para entrenar el modelo. Verifica las rutas y las imágenes.")

    print(f"Datos para entrenamiento: X={len(X)} muestras, y={y}")
    # Convertir a array 2D
    X = np.array(X).reshape(len(X), -1)
    
    # Entrenar un clasificador Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    joblib.dump(clf, 'modelo_frutas.pkl')
    print("Modelo entrenado y guardado como 'modelo_frutas.pkl'")
    return clf

if os.path.exists('modelo_frutas.pkl'):
    print("Cargando modelo existente...")
    clf = joblib.load('modelo_frutas.pkl')
else:
    try:
        print("Entrenando modelo...")
        clf = entrenar_modelo()
    except ValueError as e:
        messagebox.showerror("Error", str(e))
        exit()

class FruitScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Reconocimiento de Frutas")
        self.root.geometry("800x500")

        # Título
        self.label_titulo = tk.Label(root, text="Sistema de Reconocimiento de Frutas", font=("Arial", 16, "bold"))
        self.label_titulo.pack(pady=10)

        self.boton_camara = tk.Button(root, text="Cámara Web", command=self.toggle_camara, bg="blue", fg="white", font=("Arial", 10))
        self.boton_camara.pack()

        self.frame_principal = tk.Frame(root)
        self.frame_principal.pack(pady=10)

        self.frame_camara = tk.Frame(self.frame_principal)
        self.frame_camara.pack(side=tk.LEFT, padx=10)

        self.label_monitor = tk.Label(self.frame_camara, text="Monitor de Cámara", font=("Arial", 12))
        self.label_monitor.pack()

        self.canvas = tk.Canvas(self.frame_camara, width=400, height=300, bg="white")
        self.canvas.pack()

        self.frame_info = tk.Frame(self.frame_principal)
        self.frame_info.pack(side=tk.RIGHT, padx=10)

        self.label_frutas = tk.Label(self.frame_info, text="Plátano", font=("Arial", 12, "bold"), bg="white")
        self.label_frutas.pack(anchor="ne")
        self.label_frutas2 = tk.Label(self.frame_info, text="Mango", font=("Arial", 12), bg="white")
        self.label_frutas2.pack(anchor="ne")
        self.label_frutas3 = tk.Label(self.frame_info, text="Manzana", font=("Arial", 12), bg="white")
        self.label_frutas3.pack(anchor="ne")

        self.label_info = tk.Label(self.frame_info, text="Calorías: --\nCarbohidratos: --\nPrecio: --\nReceta: --", font=("Arial", 12), justify=tk.LEFT)
        self.label_info.pack(anchor="ne", pady=20)

        self.boton_foto = tk.Button(self.frame_camara, text="Tomar Foto", command=self.tomar_foto, font=("Arial", 10))
        self.boton_foto.pack(pady=10)

        self.cap = None
        self.camara_activa = False

    def toggle_camara(self):
        if not self.camara_activa:
            print("Intentando abrir la cámara...")
            self.cap = cv2.VideoCapture(0)  # Cámara USB (puede ser 1 si 0 no funciona)
            if not self.cap.isOpened():
                print("Error: No se pudo abrir la cámara")
                messagebox.showerror("Error", "No se pudo abrir la cámara. Asegúrate de que esté conectada y no esté siendo usada por otra aplicación.")
                return
            print("Cámara abierta correctamente")
            self.camara_activa = True
            self.boton_camara.config(text="Apagar Cámara")
            self.mostrar_video()
        else:
            print("Apagando cámara...")
            self.camara_activa = False
            self.boton_camara.config(text="Cámara Web")
            if self.cap:
                self.cap.release()
            self.canvas.delete("all")
            self.canvas.create_text(200, 150, text="Cámara apagada", font=("Arial", 12))

    def mostrar_video(self):
        if self.camara_activa:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (400, 300))
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            else:
                print("Error: No se pudo leer el frame de la cámara")
            self.root.after(10, self.mostrar_video)

    def tomar_foto(self):
        if not self.camara_activa:
            messagebox.showwarning("Advertencia", "Por favor, activa la cámara primero")
            return

        ret, frame = self.cap.read()
        if ret:
            print("Foto tomada, procesando...")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (100, 100))
            # Extraer características (promedio de color RGB)
            avg_color = np.mean(frame, axis=(0, 1))
            print(f"Características de la foto: {avg_color}")
            # Predecir la fruta
            try:
                prediccion = clf.predict([avg_color])[0]
                print(f"Predicción: {prediccion}")
                # Actualizar la interfaz
                self.label_frutas.config(bg="yellow" if prediccion == "platano2" else "white")
                self.label_frutas2.config(bg="yellow" if prediccion == "mango" else "white")
                self.label_frutas3.config(bg="yellow" if prediccion == "manzana" else "white")
                # Mostrar información
                info = info_frutas.get(prediccion, {'calorias': '--', 'carbohidratos': '--', 'precio': '--', 'receta': '--'})
                self.label_info.config(text=f"Calorías: {info['calorias']}\nCarbohidratos: {info['carbohidratos']}\nPrecio: {info['precio']}\nReceta: {info['receta']}")
            except Exception as e:
                print(f"Error al predecir: {e}")
                messagebox.showerror("Error", "No se pudo realizar la predicción. Verifica el modelo.")
        else:
            print("Error: No se pudo tomar la foto")
            messagebox.showerror("Error", "No se pudo tomar la foto. Verifica la cámara.")

    def __del__(self):
        if self.cap:
            self.cap.release()

class LoginApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Login")
        self.root.geometry("300x200")

        # Etiquetas y campos de entrada
        label_usuario = tk.Label(root, text="Usuario:")
        label_usuario.pack(pady=10)
        self.entry_usuario = tk.Entry(root)
        self.entry_usuario.pack()

        label_contraseña = tk.Label(root, text="Contraseña:")
        label_contraseña.pack(pady=10)
        self.entry_contraseña = tk.Entry(root, show="*")
        self.entry_contraseña.pack()

        boton_login = tk.Button(root, text="Iniciar Sesión", command=self.verificar_login)
        boton_login.pack(pady=20)

    def verificar_login(self):
        usuario = self.entry_usuario.get()
        contraseña = self.entry_contraseña.get()
        
        if usuario == "user" and contraseña == "123456":
            self.root.destroy()  # Cierra la ventana de login
            root = tk.Tk()
            app = FruitScannerApp(root)
            root.mainloop()
        else:
            messagebox.showerror("Error", "Usuario o contraseña incorrectos")

if __name__ == "__main__":
    root = tk.Tk()
    app = LoginApp(root)
    root.mainloop()