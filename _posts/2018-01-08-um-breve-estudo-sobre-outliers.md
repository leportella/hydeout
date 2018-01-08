---
layout: post
title: "Um breve estudo sobre outliers"
categories:
  - pt-br
tags:
  - pt-br
  - python
  - data science
  - outliers
last_modified_at: 2018-08-01T13:25:52-05:00
---

Nas últimas semanas tenho me dedicado a estudar e entender melhor sobre dados anômalos ou *outliers* em séries de dados. 

Na oceanografia, nossos dados anômalos são facilmente reconhecíveis. Uma corrente em uma região calma não tem como chegar em 
10m/s de repente ou uma região com um variação de 1m na coluna de água não vai ter 1 dado com 17m. Mas nem todo conjunto de dados 
te permite saber o que é um *outlier* de forma tão "simples". 

Este post é uma primeira tentativa em descrever o que eu encontrei e refleti sobre as técnicas para identificar dados anômalos 
focando em um conjunto de dados em que o meu *outlier* prejudica uma avaliação dos dados ou prejudica uma análise preditiva, por exemplo.

![](https://media.giphy.com/media/26gJA9SSe4m54MYec/giphy.gif)

## Definição de Outliers

"Um *outlier* é uma observação que se diferencia tanto das demais observações que levanta suspeitas de que aquela observação 
foi gerada por um mecanismo distinto" (Hawkins, 1980)

Muitas aplicações precisam definir se uma determinada observação pertence à mesma distribuição das demais observações 
(é um *inlier*) ou se ela deve ser considerada distinta das demais observaçes (é um *outlier*) ([Sklearn](http://scikit-learn.org/stable/modules/outlier_detection.html)). Nesse contexto, devemos fazer duas distinções no contexto em que os dados podem ser encontrados:

* Detecção de *novelty*: Quando o seu dado de treino não é poluído por *outliers* e estamos interessados em observar 
anomalias em novas observações (dados de teste, por exemplo).
* Detecção de *outliers*: O seu dado de treinamento contém *outliers* e nós precisamos em entender o comportamento padrão 
dos dados para ignorar as observaçes anômalas.

Nesse artigo, vamos fazer análises referentes apenas a *outliers*, quando não temos uma referência que nos indique o que de fato é um *outlier* e o que não é. É interessante dizer que é muito difícil detectar *outliers* em um espaço n-dimensional quando n > 2 porque não é mais possível fazer inspeções visuais (Rousseeuw & Van Driessen, 1999). Então, como é possível avaliar outliers em espaços n-dimensionais?

## Porque entender Outliers?

A detecção de anomalias pode ser utilizada para diversos campos, em que o *outlier* na realidade é a expressão de algo que deve 
ser avaliado com cuidado. Podemos citar:

* Identificação de movimentações fraudulentas em cartões de 
crédito;
* Em uma imagem do espaço, uma anomalia pode ser [a forma de identificar uma nova estrela](https://www.technologyreview.com/the-download/609785/artificial-intelligence-just-discovered-new-planets/);
* Sintomas não usuais podem indicar problemas de saúde de um paciente, considerando a sua idade e gênero;
* A ocorrência anômala de um determinado caso de doença em um hospital ou cidade pode gerar alertas para que haja uma 
investigação das possíveis causas para tal.

Além de casos em que os *outliers* são o foco de determinado estudo, também podemos citar os casos em que a identificação destes 
pode ser útil para fazer uma "limpeza" na bases de dados. Muitas vezes estamos estudando processos que sabidamente não conseguem 
gerar esse tipo de observação ou tiveram alguma influência externa que fez com que a observação se apresentasse muito diferente 
das demais. Nesse caso, a identificação de observaçes anômalas e limpeza da base servem para facilitar o estudo do processo e até 
otimizar algortimos de aprendizado de máquina, por exemplo.

### O Caso de Hadlum vs Hadlum [Barnett, 1978]

A identificação de um caso real de *outlier* teve profunda influência no destino da família Hadlum em 1949. O Senhor Hadlum 
estava apelando para uma corte porque o seu pedido inicial de divórcio havia sido negado. O Senhor Hadlum estava entrando com o 
pedido de divórcio alegando adultério da esposa, a Senhora Hadlum. A única evidência apresentada consistia na data que a Senhora 
Hadlum havia dado a luz a uma criança: 349 dias depois do Senhor Haldum ir embora para prestar serviço militar.

Neste caso, o tamanho da gestação (de 349 dias) era muito maior que a média de gestaçes, de 280 dias e, portanto, poderia ser 
considerado um *outlier*. O juiz do caso decidiu que deveria haver uma limite máximo de credibilidade em algum lugar. Ele 
determinou, então, que baseado em evidencias médicas, 349 era um número bem improvável mas cientificamente possível e negou o 
pedido do Senhor Hadlum. Em 1951 uma outra corte definiu o limite como 360 dias. 

Fascinante, não?

![](https://i.imgur.com/8b73OnB.png)

## Causas da ocorrência de um Outlier

*Outliers* podem refletir (Barnett, 1978):

* Erros de medida (Ex: o aparelho quebrou ou está desregulado, sujeira, etc)
* Falhas na execução da medida
* Processos distintos daqueles que geram as demais observações.

## Análise de outliers via Boxplot

Tukey (1977) definiu o conceito de outliers via boxplots, que é um dos jeitos mais simples de se detectar outliers de forma 
visual para uma variável. A metodologia consiste na identificação via quartis. Dado que o quartil de 25% (Q1)  o valor em que 25% dos dados estão abaixo dele e o quartil 75% (Q3) é o valor onde 25% dos dados estão acima desse valor, temos duas "cercas" definidas por Tukey (1977):

`IQR = Q1 - Q3`

`f1 = Q1 - 1,5IQR` e `f3 = Q3 + 1,5IQR`

e

`F1 = Q1 - 3IQR` e `F3 = Q3 + 3IQR`

Onde dados que ficam entre f1 e F1 ou f3 e F3 são chamados de *outliers* externos ("*outside outliers*") enquanto 
dados maiores que F1 e F3 são chamados de *outliers* longínquos ("*far out outliers").

## Identificação de Outliers - Estudo de caso com o Elliptic Envelope

Nesta sessão vamos fazer análises mais complexas usando o algoritmo EllipticEnvelope do Sklearn para tentar entender análises de *outliers* em ambientes mais complexos e multidimensionais.

Lembrando que detalhes das análises e gráficos estão disponíveis [aqui](https://github.com/leportella/outlier-analysis).

### 1. Criação de datasets

Para fazer análise de *outliers* vamos criar 1 dataset especfico para o que queremos. Detalhes da criação deste set de dados 
pode ser encontrada [aqui](https://github.com/leportella/outlier-analysis/blob/master/create_dataset.ipynb).

Basicamente temos uma variável y (*store_profit*) que tem um hitograma de cauda longa:

![](https://i.imgur.com/ygXccBS.png)

Duas variáveis independentes distribuda em torno de um centróide (*products in stock* e *product rating*):

![](https://i.imgur.com/EvbWMC5.png)

E mais uma variável categrica (*business_type*) e duas variáveis que são números internos, sem padrão de distribuição.

O Dataframe resultante:

![](https://i.imgur.com/sH7Rk7F.png)

Com essas características:

![](https://i.imgur.com/EvbWMC5.png)

### EllipticEnvelope em X e variáveis numéricas

Uma forma simples de identificar *outliers* é assumir que o dado tem uma determinada distribuição conhecida (por exemplo, 
Gaussiana). Uma vez que assumimos isso, podemos definir uma "forma" para o dado, e definir que observações que ficam 
longe desse formato como anomalias.

No Sklearn temos a técnica do ([*covariance.EllipticEnvelope*](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html)) que estima uma covariância robusta ao dado, adaptando uma 
elipse na parte central da nuvem de dados e ignorando o fora desse centro.

Vamos usar o Elliptic Envelope para avaliar nossos dados considerando que temos 1% de contaminação. Para facilitar a compreensão, vamos utilizar apenas as variáveis *products_ind_stock* e *product_rating* como as variáveis no nosso X:

```python
X = df[['products_in_stock', 'product_rating']]
y = df.store_profit
```

Agora vamos analisar nossos dados apenas considerando nossas variáveis independentes:

```python
from sklearn.covariance import EllipticEnvelope

clf = EllipticEnvelope(contamination=0.01)
clf.fit(X)
inliers = clf.predict(X)
```

O resultado da análise é um vetor com o mesmo tamanho do nosso DataFrame contendo valores ou 1 ou -1. 
Todos as observações que tem valor de 1 no vetor foram consideradas pelo algoritmo como *inliers*, ou seja, valores que são 
coerentes com a distribuição, enquanto que valores -1 são observaçes consideradas outliers. 

O resultado, quando observamos graficamente, é este:

![](https://i.imgur.com/Os4hDhg.png)

Mas repare que não consideramos que os valor da nossa variável y nesse processo. Dessa forma, ao observamos as amostras em y que 
foram consideradas *outliers* elas estão distribudas quase aleatóriamente:

![](https://i.imgur.com/2ooxSD3.png)

Muito legal hein?

### EllipticEnvelope em Y

Agora vamos fazer a mesma análise. Mas, ao invés de passarmos minhas variáveis X para o algoritmo, vamos analisar a minha 
variável y. Qual resultado é esperado?

```python
y  = y.values.reshape(-1,1)

from sklearn.covariance import EllipticEnvelope

clf = EllipticEnvelope(contamination=0.01)

clf.fit(y)
inliers = clf.predict(y)
```

Na verdade, quando avaliamos uma variável só, especialmente quando ela tem uma longa cauda como observamos lá em cima, o algoritmo basicamente faz um "corte" da cauda. 

Dessa forma, temos que o nosso gráfico de y vai aparecer com um "corte", removendo valores muito extremos:

![](https://i.imgur.com/xUj90Kg.png)

Agora, se olharmos o gráfico dos valores das variáveis independentes temos a aleatoriadade das amostras:

![](https://i.imgur.com/JMddE1Q.png)

### EllipticEnvelope em X e variáveis categóricas

E se, ao invés de usarmos apenas duas variáveis numéricas como vimos anteriormente, usássemos todas as variáveis que temos 
disponíveis? Categóricas ou não? 

```python
y = df.store_profit
X = df.drop('store_profit', axis=1) # pega todos os valores exceto a coluna do y
```

Bom, primeiro temos que fazer um *encoding* das variáveis categóricas para que o algoritmo as entenda:

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
X['business'] = encoder.fit_transform(X['business'])
```

Depois a análise fica exatamente do mesmo jeito que anteriormente:

```python
from sklearn.covariance import EllipticEnvelope

clf = EllipticEnvelope(contamination=0.01)
clf.fit(X)
inliers = clf.predict(X)
```

O resultado é diferente daquele que encontramos anteriormente, considerando apenas duas variveis:

![](https://i.imgur.com/sbQvPqO.png)

Entretanto, não tem como fazermos a avaliação visual de como isso se comporta no espaço, como fizemos no exemplo anterior :/


## Reflexão

A análise de *outliers* depende muito do dado que você está vendo e o que você quer observar. Isso já é um senso-comum. 
Porém, ao se fazer uma análise, devemos sempre ter em mente que diferentes técnicas resultam em diferentes *outliers*, dependendo do enfoque que você der para ela ou da forma como ela faz a identificação. Muitas vezes, o melhor é tentar fazer diferentes técnicas para ver qual delas é mais eficiente. 

Considerando os casos que apresentei, a conclusão que eu obtive é que existem 3 casos diferentes que podem ocorrer quando 
se tenta identificar um *outlier* para fazer análises preditivas a partir dos dados:

**Caso 1: Temos uma anomalia em y**

O primeiro caso teríamos a anomalia presente em y, como mostra a tabela abaixo:

<center>
<img src="https://i.imgur.com/NPKmR2U.png" height="100" style="max-width: 20%" />
</center>

Se fizermos a análise baseando-nos em y, temos a análise removendo corretamente o quinto elemento, que é claramente um *outlier*.
Agora, se fizermos a análise baseando-nos em X, essa observação não será removida. 

**Caso 2: Temos uma anomalia em X**

No segundo caso, a observação anômala ocorrem X. Dessa vez, a análise apenas de y pode não remover esse caso que é, claramente, 
um anômalo.

<center>
<img src="https://i.imgur.com/xCetOdk.png" height="100" style="max-width: 20%"/>
</center>


**Caso 3: Temos anomalia em X e y**

No terceiro caso temos a aparição de dados anômalos justamente em X e em y. A análise, de qualquer forma que se olhe, poderia nos dar uma impressão errada de um dado anômalo quando, na verdade, temos um comportamento relativamente fácil de prever. Quando X é grande, y será grande também.

<center>
<img src="https://i.imgur.com/BsPPlVU.png" height="100" style="max-width: 20%" />
</center>


## Fim?

Por enquanto foram esses os tópicos que eu cheguei e as análises que fiz para tentar entender melhor o assunto. O tema é vasto, 
as técnicas são várias e tem muita coisa pela frente. Se você sabe de algo que não está aqui, 
por favor se sinta a vontade para comentar :)

![](https://media.giphy.com/media/q9lNzUPfLAbBK/giphy.gif)

## Referências

Barnett, V. 1978. The study of Outliers: purpose and model. Appl. Statics, 27, no.3, pp.242-250.

Hawkis, D. M. 1980. Identification of Outliers. Monographs on applied probability and statistics. DOI: 10.1007/978-94-015-3994-4.

Rousseeuw, P. J.; Van Driessen, K.. A Fast Algorithm for the minimum covariance determinant estimator

Tukey, J.W., 1977. Exploratory Data Analysis. Addison-Wesley, Reading, MA.

http://www.dbs.ifi.lmu.de/~zimek/publications/SDM2010/sdm10-outlier-tutorial.pdf
