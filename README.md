# Tutorial-Naive-Bayes

Naive Bayes é um método de aprendizado de máquina supervisionado usado para classificação que considera as variáveis como independentes, por esse motivo é tido como ingênuo (naive em inglês). É um bom método, simples de compreender e de fácil implementação, frequentemente aplicado em processamento de linguagem natural e diagnósticos médicos. Esse método pode ser usado quando os atributos que descrevem as instâncias forem condicionalmente independentes dada a classificação.
Para entendermos como funciona o classificador Naive-Bayes, temos que primeiro entender o teorema de Bayes.
O teorema de Bayes trata sobre probabilidade condicional. Isto é qual a probabilidade de o evento A ocorrer, dado o evento B. Para explicar melhor, vamos a um exemplo. Imagine uma família tenha duas crianças. Qual a probabilidade de que, se uma delas for menina, ambas sejam meninas? Pense um pouco e tente responder por conta própria… ½ ? ¼ ? Nada disso! A resposta correta é ⅓ e o teorema de Bayes explica o porquê.
Se a condição (“se uma delas for menina”) não estivesse presente a resposta seria ¼ , com as seguintes ordens de nascimento (menino, menino), (menino, menina), (menina, menino), (menina, menina), ou seja 1 em 4. Porém dada a informação adicional que a família tem uma menina, a probabilidade passa a ser ⅓ , pois só restam 3 situações possíveis para a família (menino, menina), (menina, menino), (menina, menina) e 1 das 3 corresponde ao resultado de que ambas são meninas.

 

O teorema de Bayes pode ser resumido pela seguinte fórmula:

P (A | B) = P (B | A) . P (A) / P (B), lê-se: a probabilidade do evento A ocorrer dado o evento B é igual probabilidade do evento B ocorrer dado o evento A vezes a probabilidade do evento A sobre a probabilidade do evento B.

 

Aplicando a fórmula no nosso exemplo, consideremos:
P (2M | 1M): probabilidade de que ambas sejam meninas dado que pelo menos 1 é menina
P (1M | 2M): probabilidade de que pelo menos 1 seja menina dado que ambas são meninas
P (2M): probabilidade das 2 serem meninas
P (1M): probabilidade de que pelo menos 1 seja menina

P (2M | 1M) = P (1M | 2M) * P (2M) / P (1M) = 1 * ¼ ÷ ¾ = ⅓

 

Essa mesma lógica pode ser utilizada para o cálculo das probabilidades necessárias para problemas de classificação. Para entendermos, basta substituir um dos argumentos da fórmula pela classe a ser calculada.

 

P (classe | B) = P (B | classe) . P (classe) / P (B), lê-se: a probabilidade de pertencer a classe escolhida dado o atributo B é igual probabilidade do atributo B ocorrer dado que ele pertence a classe escolhida vezes a probabilidade de ocorrer a classe sobre a probabilidade do evento B.

 

Para calcular a classe mais provável da nova instância, calcula-se a probabilidade de todas as possíveis classes e, no fim, escolhe-se a classe com a maior probabilidade como rótulo da nova instância. Em termos estatísticos, isso é o mesmo do que maximizar a P (classe | a1…an). Para tanto, deve-se maximizar o valor do numerador P(a1…an | classe) × P(classe) e minimizar o valor do denominador P(a1…an). Como o denominador P(a1…an) é uma constante, pois não depende da variável classe que se está procurando, pode-se anulá-lo no Teorema de Bayes, resultando na fórmula abaixo, na qual se procura a classe que maximize o valor do termo P ( classe | a1…an ) = P ( a1…an | classe ) × P ( classe ):

argmax P ( classe | a1 … an ) = argmax P (ai… an | classe) * P ( classe)

 

A suposição “ingênua” que o classificador naive-Bayes faz é que todos os atributos a1…an da instância que se quer classificar são independentes. Sendo assim, o cálculo do valor do termo P(a1…an | classe) reduz-se ao simples cálculo de P(a1 | classe)× … ×P(an | classe). Assim, a fórmula final utilizada pelo classificador é:

argmax P ( classe | a1 … an ) = argmax ∏i⇨n P (ai | classe) * P ( classe)

 

Sabe-se, entretanto, que, na maioria dos casos, a suposição de independência dos atributos de uma instância é falsa. Mesmo assim, o classificador naive-Bayes produz resultados bastante satisfatórios. Quando os atributos são realmente independentes, o classificador fornece a solução ótima. Como dito anteriormente, o cálculo da classe de uma nova instância consiste no cálculo da probabilidade de todas as possíveis classes, escolhendo-se, a seguir, a classe com maior probabilidade. De acordo com a fórmula acima, devem ser calculados os termos P(ai | classe) e P(classe). P(classe) é simplesmente o número de casos pertencente a classe em questão sobre o número total de casos. P(ai|classe), por sua vez, é o número de casos pertencente a classe em questão com o atributo i com valor ai sobre o número total de casos.

 

Como exemplo do uso do classificador naive-Bayes, considere o conjunto de dados para treinamento, no qual se deve decidir se vai jogar tênis ou não com base na previsão do tempo. Perceba que esse é um problema de classificação, no qual temos duas classes possíveis : Jogar Tênis = sim ou Jogar Tênis = não.


!!!!!!!!!!!!!!!
VER tabela1.png
!!!!!!!!!!!!!!!

 

Imagine que você queira descobrir a probabilidade de jogar tênis nas seguintes condições: Tempo = sol, Temperatura = frio, Umidade = alta, Vento = forte, baseado nos dados de treinamento podemos chegar a resposta seguindo as seguintes etapas:

 

1. Calcular a probabilidade de cada classe ocorrer:

P  (Jogar Tênis = sim) = 9/14 = 0,64
P  (Jogar Tênis = não) = 5/14 = 0,36

 

2. Calcular a probabilidade de cada um dos atributos em relação a cada classe possível. Por exemplo, para o atributo Tempo = sol :

P (Tempo = sol | Joga Tênis = sim) = 2/9
P (Tempo = sol | Joga Tênis = não) = 3/5

 

3. Tendo todas as probabilidades calculadas, aplicar a fórmula para calcular a probabilidade de cada classe (neste caso, “sim” ou “não”) ocorrer:

P (Jogar tênis = sim | Tempo = sol, Temperatura = frio, Umidade = alta, Vento = forte) =

P(sol | sim) * P(frio | sim) * P(alta | sim) * P (forte | sim) * P (JogarTênis = sim)

= 2/9 * 3/9 * 3/9 * 3/9 * 9/14
= 0,0053

P (Jogar tênis = não | Tempo = sol, Temperatura = frio, Umidade = alta, Vento = forte) =

P(sol | não) * P(frio | não) * P(alta | não) * P (forte|não) * P (JogarTênis = não)

= 3/5 * 1/5 * 4/5 * 3/5 * 5/14
= 0,020

 

Assim, dado os atributos da instância que se quer classificar, concluímos que a probabilidade de não jogar tênis é maior do que a probabilidade de jogar (0,02 > 0,0053), portanto classificamos a instância como Jogar Tênis = não!

É assim que, grosso modo, funciona o classificador Naive-Bayes.

 

Fontes:

//scikit-learn.org/stable/modules/naive_bayes.html
//www.youtube.com/watch?v=oQ1OyqvL7dQ
//www.youtube.com/watch?v=3HJVRBEMwoU&t=424s
O andar do bêbado – Leonard Mlodinow
//conteudo.icmc.usp.br/pessoas/taspardo/NILCTR0225-PardoNunes.pdf
//en.wikipedia.org/wiki/Naive_Bayes_classifier
//www.vooo.pro/insights/6-passos-faceis-para-aprender-o-algoritmo-naive-bayes-com-o-codigo-em-python/
