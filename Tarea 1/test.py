import practica_06_NB as pra
import files.votes as votes


asd = pra.ClasificadorNaiveBayes(votes.votos_atributo_clasificacion, votes.votos_clases,
	votes.votos_atributos, votes.votos_valores_atributos)

asd.entrena(votes.votos_entr,votes.votos_entr_clas,None, None)

print(asd.probabilidades)