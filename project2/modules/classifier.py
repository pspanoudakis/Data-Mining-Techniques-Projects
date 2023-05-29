from typing import Sequence, Union, Callable
import string

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from gensim.models import Word2Vec

from tqdm.notebook import tqdm

from .utils import printMd, DataColumn, STOP_WORDS

class SentencesVectorizer:
    def __init__(self, word2vec: Word2Vec) -> None:
        self.vectors: np.ndarray
        self.__word2vec__ = word2vec

    def getVectorSize(self):
        return self.__word2vec__.vector_size

    def __saveSentenceVector__(self, idx: int, s: str):
        words = s.encode('utf-8').split()
        for w in words:
            self.vectors[idx] += self.__word2vec__.wv[w][0]

        self.vectors[idx] /= max(len(words), 1)
    
    def createVectors(self, sentences: pd.Series):
        self.vectors = np.zeros(shape=(len(sentences), self.getVectorSize()))
        for i, s in enumerate(sentences):
            self.__saveSentenceVector__(i, s)

        return self.vectors

class BookGenreClassifier:

    SupportedEstimator = Union[GaussianNB, RandomForestClassifier]
    EstimatorFactory = Callable[[], SupportedEstimator]

    def __init__(self) -> None:
        self.vectorSize: int
        self.vectorizer: SentencesVectorizer

        self.__trainX__: np.ndarray
        self.__testX__: np.ndarray

        self.__trainY__: np.ndarray
        self.__testY__: np.ndarray

        self.labels: Sequence[str]

    @staticmethod
    def cleanDescription(s: str):
        s = ''.join(c.lower() if (ord(c) < 128 and c not in (string.punctuation + string.digits)) else ' ' for c in s)    
        return (' '.join((w for w in s.split() if w not in STOP_WORDS))).strip()
        
    def createTrainingData(self, booksDf: pd.DataFrame, vectorSize: int):
        self.vectorSize = vectorSize
        self.vectorizer = SentencesVectorizer(Word2Vec(
            sentences=[s.encode('utf-8').split() for s in booksDf[DataColumn.DESCRIPTION].values],
            vector_size=self.vectorSize
        ))

        cleanDesc = booksDf[DataColumn.DESCRIPTION].apply(self.cleanDescription)
        cleanDesc = cleanDesc.loc[lambda d: d != '']

        X = self.vectorizer.createVectors(cleanDesc)
        Y = booksDf[DataColumn.GENRESINGLE].loc[cleanDesc.index].values

        self.labels = np.unique(Y)
        (
            self.__trainX__,
            self.__testX__,
            self.__trainY__,
            self.__testY__,
        ) = train_test_split(X, Y, train_size=0.8, shuffle=False)

        return self
    
    def setEstimator(self, estimatorFactory: EstimatorFactory):
        self.__estimator__ = estimatorFactory()    

    def __doCrossValidation__(
        self,
        estimator: SupportedEstimator,
        trainX: np.ndarray,
        trainY: np.ndarray,
        testX: np.ndarray,
        testY: np.ndarray,
        verbose: bool = True
    ):
        estimator.fit(trainX, trainY)
        # trainPred = estimator.predict(trainX)
        # testPred = estimator.predict(testX)

        trainScore = estimator.score(trainX, trainY)
        testScore = estimator.score(testX, testY)

        if verbose:
            printMd(f'Train Set Accuracy: {trainScore}')
            printMd(f'Test Set Accuracy: {testScore}')

        return trainScore, testScore

    def performCrossValidation(self, estimatorFactory: EstimatorFactory, verbose: bool = True):
        self.__doCrossValidation__(
            estimatorFactory(),
            self.__trainX__,
            self.__trainY__,
            self.__testX__,
            self.__testY__,
            verbose=verbose
        )

        return self

    def performKFold(self, k: int, estimatorFactory: EstimatorFactory, verbose: bool = True):

        kfold = KFold(n_splits=k)
        foldItr = kfold.split(self.__trainX__)

        if verbose:
            foldItr = tqdm(
                foldItr,
                total=kfold.get_n_splits(),
                bar_format='{desc}{bar} {n}/{total} -- Time Elapsed: {elapsed}'
            )

        for i, (trainIds, testIds) in enumerate(foldItr):
            trainX = self.__trainX__[trainIds]
            trainY = self.__trainY__[trainIds]

            testX = self.__trainX__[testIds]
            testY = self.__trainY__[testIds]

            printMd(f'***\n**Fold {i + 1}**')

            self.__doCrossValidation__(
                estimatorFactory(),
                trainX,
                trainY,
                testX,
                testY,
                verbose=True
            )
