import argparse
import practica_06_NB as practica

def run(source, k):
	if source=='digitos':
		import files.digits as digits

		if k:
			clasificador = practica.ClasificadorNaiveBayes(digits.digitos_atributo_clasificacion,
				digits.digitos_clases,digits.digitos_atributos, digits.digitos_valores_atributos, int(k) or 1)

			clasificador.entrena(digits.digitos_entr,digits.digitos_entr_clas,digits.digitos_valid, 
				digits.digitos_valid_clas, autoajuste=False)

			print("Rendimiento de pruebas: " + str(clasificador.test(digits.digitos_test, digits.digitos_test_clas)))
		else:
			clasificador = practica.ClasificadorNaiveBayes(digits.digitos_atributo_clasificacion,
				digits.digitos_clases,digits.digitos_atributos, digits.digitos_valores_atributos)

			clasificador.entrena(digits.digitos_entr,digits.digitos_entr_clas,digits.digitos_valid, 
				digits.digitos_valid_clas)


	else:
		import files.votes as votes

		if k:
			clasificador = practica.ClasificadorNaiveBayes(votes.votos_atributo_clasificacion,
				votes.votos_clases,votes.votos_atributos, votes.votos_valores_atributos, int(k) or 1)

			clasificador.entrena(votes.votos_entr,votes.votos_entr_clas,votes.votos_valid, 
				votes.votos_valid_clas, autoajuste=False)

			print("Rendimiento de pruebas: " + str(clasificador.test(votes.votos_test, votes.votos_test_clas)))
		else:
			clasificador = practica.ClasificadorNaiveBayes(votes.votos_atributo_clasificacion,
				votes.votos_clases,votes.votos_atributos, votes.votos_valores_atributos)

			clasificador.entrena(votes.votos_entr,votes.votos_entr_clas,votes.votos_valid, 
				votes.votos_valid_clas)

if __name__=='__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-s", "--source", required=True, choices=[
                    'digitos', 'votos'],
                    help="Data source used: Digitos /\
                     Votos")
	ap.add_argument("-k", "--k", required=False,
					help="k suavizado de Laplace")
	args = vars(ap.parse_args())
	run(args['source'], args['k'])