#-*-coding:utf-8-*-

def read_image(path):
	images = []
	image = []
	with open(path) as f:
		cont = 1
		while True:
			c = f.read(1)

			if not c:
				break

			if cont%29==0:

				if cont%812==0:
					images.append(image)
					image = []

				# Linebreak
				cont += 1
				continue

			image.append(c=='+' or c=='#')

			cont += 1
				
	return images

def read_labels(path):
	labels = []
	with open(path) as f:
		for line in f:
			if line:
				labels.append(int(line))
	return labels

# Atributo de clasificación
# -------------------------

digitos_atributo_clasificacion='digito'


# Valores de clasificación:
# -------------------------

digitos_clases=list(range(10))


# Atributos (o características):
# ------------------------------
# Para simplificar usaremos el formato casillaX, que corresponde al píxel
# X de la imagen cuadrada de tamaño 28x28.

digitos_atributos=['casilla'+str(i) for i in list(range(1,28*28))]

# Valores posibles de cada atributo:
# ----------------------------------

digitos_valores_atributos={}

for atributo in digitos_atributos:
	digitos_valores_atributos[atributo] = [True, False]

# Ejemplos del conjunto de entrenamiento:
# ---------------------------------------

digitos_entr = read_image('files/digitdata/trainingimages')

# Clasificación de los ejemplos del conjunto de entrenamiemto:
# ------------------------------------------------------------

digitos_entr_clas = read_labels('files/digitdata/traininglabels')

# Ejemplos del conjunto de validación:
# ------------------------------------

digitos_valid = read_image('files/digitdata/validationimages')

# Clasificación de los ejemplos del conjunto de validación:
# =========================================================

digitos_valid_clas = read_labels('files/digitdata/validationlabels')

# Ejemplos del conjunto de prueba (o test):
# =========================================

digitos_test = read_image('files/digitdata/testimages')

# Clasificación de los ejemplos del conjunto de prueba (o test):
# ==============================================================

digitos_test_clas = read_labels('files/digitdata/testlabels')