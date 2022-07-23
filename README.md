Repositorio hau Martin Bikandi San Millanen gradu amaierako lanean garatutako kode-liburutegia gordetzen du.

# DLTOOLBOX Dokumentazioa

Artikulu honetan DLTOOLBOX kode-liburutegiaren inplementazioa azaltzen da. Azalpen hau minimoa da, ez da funtzio askoren eta hainbat geruzen inplementazioa erakusten, askotan kodea nahiko antzekoa delako.

Liburutegia erabiltzeko, liburutegiaren esteka PATH ingurumen aldagaian egon behar da, horretarako kode hau exekutatu daiteke:

```
import sys
sys.path.append("Karpetaren helbidea/dltoolbox")
```

Liburutegia erabili ahal izateko ```numpy``` eta ```tensorflow``` beharrezkoak dira.

## class Sequential

````Sequential```` klasea liburutegiko klase garrantsitsuena da, eredu sekuentzialaren kasea. Hasieratze oso sinplea du, eta aldagai bakarra: zerrenda bat.

```
    def __init__(self):
        self.layers = []
    def add(self, layer):
        self.layers.append(layer)
    def initializeRandomWeights(self, rng):
        for layer in self.layers:
            layer.initializeRandomWeights(rng)
```

```Sequential.add``` metodoaren bidez ````Layer```` motako objektua zerrendan sartzen da. Eta ````Sequential.initializeRandomWeights```` metodoaren bidez parametro moduan ematen den ausazko zenbaki-sortzailearen bidez geruza bakoitzeko pisuak hasieratzen ditu.

```
    def inference(self, input, argmax=False):
        (...)
    def forwardProp(self, input):
        output = input
        for layer in self.layers:
            output = layer.forwardProp(output)
        return output
```

````Sequential.inference```` metodoaren bidez datu-base osoaren ereduak ematen duen iteerak bueltatzen ditu. ````argmax=True```` denean probabilitateak/irteerak bueltatu beharrean sareak igartzen dituen klaseak bueltatzen ditu. ````Sequential.forwardProp```` metodoaren bidez gauza berdina egiten da, baina ````Layer.inference```` metodoa erabili beharrean ````Layer.forwardProp```` erabiltzen du hurrengo geruzaren sarrerak kalkulatzeko.

````
    def computeGradients(self, error):
        tempError = error
        gradients = []
        for layer in reversed(self.layers):
            layerGradient, tempError = layer.backProp(tempError)
            gradients.append(layerGradient)
        return list(reversed(gradients))
````

````Sequential.computeGradients```` metodoaren bidez backpropagation algoritmoa erabiltzen da. Horretarako kostu funtzioaren bidez kalkulatutako azken geruzako erroreak erabiltzen dira. Metodoak geruzen ````Layer.backProp```` metodoa erabiltzen du. Geruza bakoitzeko pisuen gradienteak bueltatzen ditu, normalizatu barik.

## class Layer

Geruza batzuk besteak baino sinpleagoak dira, parametroak ez dituzten geruzetan metodo batzuk definitu behar ez izateko superklase honen bidez egiten da. ````Layer```` klaseak ez du aldagairik, eta definituta dituen metodoak hauek dira:

```
    def initializeRandomWeights(self, rng=None):
        pass
    def updateWeights(self, newWeights=None):
        pass
    def forwardProp(self, layerInput):
        pass
    def inference(self, layerInput):
        return self.forwardProp(layerInput)
    def weights(self):
        return np.array([])
    def regularizerCost(self):
        return 0
```

Metodo hauen bidez geruza guztiek funtzionamendu berdintsua izango dute, parametrorik gabeko geruzak izan arren. Hala ere garrantzi gutxikoak dira.

## class Dense(Layer)

Geruza dentsuan lanaren 2. atalean azaldutako formulak inplementatzen dira. Aldagai bakarrak parametroen aldagaia eta geruzaren regularizatzailea dira. ````inputSize```` tamainako sarrerak hartzen ditu (bektoreak), eta ````outputSize```` tamainako irteerak sortzen ditu.

```
    def __init__(self, inputSize, outputSize, regularizer = None):
        super().__init__()
        self._W = np.empty((inputSize + 1, outputSize), dtype=np.float32)
        self._regularizer = regularizer
    def inference(self, layerInput):
        (...)
    def forwardProp(self, layerInput):
        self._lastInput = np.hstack([np.ones(layerInput.shape[0], dtype=np.float32).reshape(layerInput.shape[0], 1), layerInput])
        return np.matmul(self._lastInput, self._W)
    def initializeRandomWeights(self, rng):
        newWeights = np.array([rng() for i in range(self._W.size)], dtype=np.float32).reshape(self._W.shape)
        self.updateWeights(newWeights)
    def updateWeights(self, newWeights):
        self._W = newWeights
    def weights(self):
        return self._W
    def regularizerCost(self):
        return self._regularizer.f(self._W) if self._regularizer is not None else 0
    def backProp(self, error):
        partialDerivatives = np.matmul(self._lastInput.T, error)
        errorPreviousLayer = np.matmul(error, self._W[1:,:].T)
        if self._regularizer is not None:
            regularizer_derivatives = self._regularizer.df(self._W)
            regularizer_derivatives[0,:] = 0
            partialDerivatives += regularizer_derivatives
        return partialDerivatives, errorPreviousLayer
```

````Dense.inference```` eta ````Dense.forwardProp```` funtzioak gauza berdina egiten dute, sarreraren aurreranzko propagazioa.

````Dense.initializeRandomWeights```` metodoaren bidez pisuen hasieratzea egiten da, pisuen tamaina duen lista bat sortzen du eta gero pisuen tamainari egokitu (geruza konboluzionaletan gauza berdina egiten da).

````Dense.updateWeights```` pisu berriak eguneratzen ditu, eta ````Dense.weights```` pisuak bueltatu.

````Dense.regularizerCost```` regularizatzailearen kostua bueltatzen du.

````Dense.backProp```` geruzaren irteeren errorea hartuz geruzen sarreren erroreak kalkulatzen ditu, 2. atalean azaldutako formulen bidez. Regularizatzailea egotekotan pisuen regularizazioa egiten du (alborapen balioak ez ditu erregularizatzen). Azkenik pisuen gradienteak bueltatzen ditu, eta sarreren erroreak.

## class conv2DLayer(Layer)

Geruza konboluzionala 3. atalean azaltzen den moduan irteerak kalkulatzen ditu. Sarrerak ````inputs```` ezaugarri mapadun irudiak dira, eta ````filters```` iragazki kopuru batekin konboluzioa egin ostean ````filters```` ezaugarri mapadun irteerak sortzen ditu. Iragazkien tamaina ````filter_size````-ren bidez zehazten da.

```
    def __init__(self, inputs, filters, filter_size, regularizer=None):
        super().__init__()
        self._filters = np.empty((filter_size[0], filter_size[1], inputs, filters), dtype=np.float32)
        self._regularizer = regularizer
    def forwardProp(self, layerInput):
        self._lastInput = layerInput
        return tf.nn.conv2d(tf.constant(self._lastInput),
                            tf.constant(self._filters),
                            strides=(1,1),
                            data_format="NHWC",
                            padding="VALID").numpy()
    def backProp(self, error):
        input_tensor = tf.constant(self._lastInput)
        error_tensor = tf.constant(error)
        filters_tensor = tf.constant(self._filters)
        reshaped_input = tf.transpose(input_tensor, perm=(3,1,2,0))
        error_filters = tf.transpose(error_tensor, perm=(1,2,0,3))
        flipped_filters = tf.transpose(tf.reverse(filters_tensor, axis=(0,1)), perm=(0,1,3,2))
        pad = [ [0, 0],
                [self._filters.shape[0] - 1, self._filters.shape[0] - 1],
                [self._filters.shape[1] - 1, self._filters.shape[1] - 1],
                [0, 0]]
        filter_gradients = tf.nn.conv2d(reshaped_input,
                                        error_filters,
                                        strides=1,
                                        data_format="NHWC",
                                        padding="VALID")
        filter_gradients = tf.transpose(filter_gradients, perm=(1,2,0,3)).numpy() # /error.shape[0]
        error_gradients = tf.nn.conv2d(error_tensor,
                                       flipped_filters,
                                       strides=1,
                                       data_format="NHWC",
                                       padding=pad).numpy()
        if self._regularizer is not None:
            filter_gradients += self._regularizer.df(self._filters)
        return filter_gradients, error_gradients
```

````conv2DLayer.forwardProp```` [tensorflow liburutegiko funtzio baten](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d) bidez egiten da, zehazki 3. atalean azaltzen den moduan funtzionatzen duena.

Era berean ````conv2DLayer.backprop```` ere tensorflow liburutegiko funtzio hori erabiltzen du iragazkien gradienteak eta sarreren erroreak kalkulatzeko.

## class maxPool2D(Layer)

Bilketa geruza 3. atalean azaltzen den moduan irteerak kalkulatzen ditu. Geruza honek sarreren ezaugarri mapa bakoitzari ````window_size```` tamainako bilketa aplikatzen dio[tensorflow liburutegiko funtzio honen](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool2d) bidez.

```
    def __init__(self, window_size):
        super().__init__()
        self._WSize = window_size
        self._lastInput = None
        self._repeatOutput = None
    def forwardProp(self, layerInput):
        self._lastInput = tf.constant(layerInput)
        output = tf.nn.max_pool2d(self._lastInput,
                                  ksize=self._WSize,
                                  strides=self._WSize,
                                  padding="VALID",
                                  data_format="NHWC").numpy()
        self._repeatOutput = np.repeat(np.repeat(output, self._WSize[0], axis=1), self._WSize[1], axis=2)
        return output
    def backProp(self, error):
        maxPositions = np.equal(self._repeatOutput, self._lastInput)
        repeatError = np.repeat(np.repeat(error, self._WSize[0], axis=1), self._WSize[1], axis=2)
        return np.array([]), np.multiply(maxPositions, repeatError)
```

````maxPool2D.backProp```` 3. atalean azaldutako eran egiten da, eta pisu gabeko geruza denez bueltatzen den pisuen gradienteak zerrenda hutsa da.

## class flatten(Layer)

Sarrerak bektorizatzen dituen geruza, hau da, 3 dimentsioko tentsorea hartuta, suposatzen dugu lehen dimentsioa laginei dagokiola, orduan azken bi dimentsioak biltzen ditu, matrize bat bueltatuz (lagin bakoitza bektore bihurtuz). Geruza hau aurreko sarreraren egitura gordetzen du, ````flatten.backprop```` egiterakoan erroreari sarreraren egitura emateko.

```
    def __init__(self):
        self._lastShape = None
    def forwardProp(self, layerInput):
        self._lastShape = layerInput.shape
        return layerInput.reshape((self._lastShape[0], np.prod(self._lastShape[1:])))
    def backProp(self, error):
        return np.array([]), error.reshape(self._lastShape)
```

## class dropoutLayer(Layer)

Dropout eragiketa aplikatzen dio sarrerako datuei. Geruzaren parametroa ````rate```` da, hau da, zero ezartzen diren neuronen ehunekoa. ````dropoutLayer.inference```` egiterakoan ez du ezer egiten, inferentzian ez delako eragiketa hau aplikatzen. ````dropoutLayer.backProp```` egiterakoan mantentzen diren erroreak aurreko aurrerako propagazioan aktibo mantendu diren neuronen erroreak bakarrik mantentzen dira.

```
    def __init__(self, rate):
        self._rate = rate
        self._lastActive = None
    def forwardProp(self, layerInput):
        self._lastActive = np.random.binomial(1, 1 - self._rate, layerInput.shape).astype(np.float32)
        return layerInput * self._lastActive
    def inference(self, layerInput):
        return layerInput
    def backProp(self, error):
        return np.array([]), error * self._lastActive
```

## class activationLayer(Layer)

Aktibazio funtzio bat aplikatzen zaio sarreraren elementu bakoitzari, geruza honen parametroa aktibazio funtzioa da.

```
    def __init__(self, activationFunction):
        super().__init__()
        self._activation = activationFunction
        self._lastOutput = None
    def forwardProp(self, layerInput):
        self._lastOutput = self._activation.f(layerInput)
        return self._lastOutput
    def backProp(self, error):
        partialDerivatives = self._activation.backward(self._lastOutput)
        errorPreviousLayer = np.multiply(error, partialDerivatives)
        return np.array([]), errorPreviousLayer
```

````class softMaxLayer```` eta ````softMaxLayer2```` gauza berdina egiten dute, baina atzeranzko propagazioa bi era desberdinetan egiten dute respektiboki. ````softMaxLayer2```` erabili behar da ````NLLHcost2```` kostu funtzioarekin funtzionatu ahal izateko.

## class id

Identitate funtzioa, aktibazio funtzio guztiak estruktura hau dute. Hala ere ````softMax```` aktibazio funtzioaren backward metodoak sarreraren errenkada bakoitzeko matrize jakobiarra bueltatzen du. Sigmoidea (````sigmoid````) eta zuzentzailea (````relu````) era berean inplementatuta daude.

```
    @staticmethod
    def f(x):
        return x
    @staticmethod
    def df(x):
        return np.ones(x.shape, dtype=np.float32)
    @staticmethod
    def backward(x):
        return np.ones(x.shape, dtype=np.float32)
```

## class NLLHcost

Log-itxaropen negatiboaren klasea. Aktibazio funtzioak dituen metodoekin.

```
    @staticmethod
    def f(X, Y):
        return -np.sum(np.multiply(Y, np.log(X + 1e-9)))/X.shape[0]

    @staticmethod
    def backward(X, Y):
        return -Y/(X + 1e-9)
```

## class HyperParameters

Hiperparametroen klasea, aldagaiak biltzen dituen klasea besterik ez da.

## class GDoptimizer(Optimizer)

Optimizatzaileen oinarrizko klasea ````Optimizer```` kalsea da, eredu neuronala, kostu funtzioa eta hiperparemetroak biltzen dituen klasea besterik ez da. Optimizatzaile guztiak gradiente beherapen optimizatzailean oinarrituta daude.

```
    def __init__(self, model, costFunction, hyperparameters):
        super().__init__(model, costFunction, hyperparameters)
    def accuracy(self, predictedLabels, correctLabels):
        correct = np.sum(np.equal(predictedLabels, correctLabels))
        return (correct/predictedLabels.size)
    def fit(self, X, Y, epochs, show_progress=0, validation_data=(None, None), compute_costs=0, compute_test_costs=0):
        (X_test, Y_test) = validation_data
        updates_count = 0
        costHistory = []
        costHistory_test = []
        accuracies = []
        accuracies_test = []
        if compute_costs>0:
            (...)
        if compute_test_costs>0:
            (...)
        _i = lambda x : min(X.shape[0], x + self._hp.batchSize)
        for epoch in range(1, epochs + 1):
            for i in range(0, X.shape[0], self._hp.batchSize):
                self._fitMiniBatch(X[i:_i(i)], Y[i:_i(i)])
                updates_count += 1
                if compute_costs>0 and updates_count%compute_costs==0:
                    (...)
                if compute_test_costs>0 and updates_count%compute_test_costs==0:
                    (...)
            if show_progress>0 and epoch%show_progress==0:
                (...)
        return costHistory, costHistory_test, accuracies, accuracies_test
```

````GDoptimizer.accuracy```` ereduak igarritako klaseak eta klase zuzenak konparatzen ditu zehaztasuna naurtzeko.

````GDoptimizer.fit```` entrenamendu eta balidazio datu-baseak hartzen ditu, eta epochs aroetan zehar entrenatzen du eredua. ````show_progress```` zero ez bada pantailan prozesuaren informazioa ikusten da. Azken bi parametroak kostuak zenbat eguneratzero kalkulatzen diren zehazten dute. Entrenamendua amaitzerakoan kalkulatutako zehaztasun eta kostuak bueltatzen ditu.

````fit```` algoritmoaren oinarrizko funtzioa ````GDoptimizer._fitMiniBatch```` da, multzo txikiaren gradienteak kalkulatu eta pisuak eguneratzen ditu.

```
    def _fitMiniBatch(self, mX, mY):
        omX = self._model.forwardProp(mX)
        propagatedError = self._costFunction.backward(omX, mY)
        gradients = self._model.computeGradients(propagatedError)
        for i, layer in enumerate(self._model.layers):
            layer.updateWeights(layer.weights() - self._hp.lr * gradients[i])
```

## class Adams(Optimizer)

Adam optimizatzailea 2. atalean azaldutako moduan inplementatzen da. ````GDoptimizer```` klasearekiko aldatzen den gauza bakarra multzo txikiarekin egiten duena da, beraz klase honek beste hiru aldagai ditu, ````self._step````, ````self._deltaS```` eta ````self._deltaW```` algoritmoan azaldutako moduan.

```
    def __init__(self, model, costFunction, hyperparameters):
        super().__init__(model, costFunction, hyperparameters)
        self._step = 0
        self._deltaS = []
        self._deltaW = []
        for layer in model.layers:
            self._deltaW.append(np.zeros_like(layer.weights()))
            self._deltaS.append(np.zeros_like(layer.weights()))
    def _fitMiniBatch(self, mX, mY):
        self._step +=1
        omX = self._model.forwardProp(mX)
        propagatedError = self._costFunction.backward(omX, mY)
        gradients = self._model.computeGradients(propagatedError)
        lr = self._hp.lr / mX.shape[0]
        lr = self._hp.lr * np.sqrt(1.0 - np.power(self._hp.beta2, self._step)) / (1.0 - np.power(self._hp.beta2, self._step))
        for i, layer in enumerate(self._model.layers):
            self._deltaW[i] = self._hp.beta1 * self._deltaW[i] + (1.0 - self._hp.beta1) * gradients[i]
            self._deltaS[i] = self._hp.beta2 * self._deltaS[i] + (1.0 - self._hp.beta2) * np.square(gradients[i])
            layer.updateWeights(layer.weights() - (lr * self._deltaW[i]) / (np.sqrt(self._deltaS[i]) + self._hp.epsilon) )
```

Beste bi optimizatzaileak, ````RMSprop```` eta ````Momentum```` era antzeko batean inplementatu dira.

## regularizatzaileak

Regularizatzaileak nahiko sinpleak dira, hemen ````L1regularizer```` klasearen inplementazioa ikusten da. Klasea ratio bat du parametro moduan eta aktibazio funtzioen inplementazioaren antzekoa da.

````
class L1regularizer:
    def __init__(self, rate):
        self._rate = rate
    def f(self, W):
        return self._rate * np.sum(np.abs(W))
    def df(self, W):
        return self._rate * np.sign(W)
````