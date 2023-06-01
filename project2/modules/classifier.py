from typing import Sequence, Union, Callable, Optional, Dict, Tuple, Literal, Any
import string

import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from gensim.models import Word2Vec

from .utils import DataColumn, STOP_WORDS, progressBarItr, runWithNoWarnings
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

    SupportedModel = Union[GaussianNB, RandomForestClassifier, SVC]
    ModelFactory = Callable[[], SupportedModel]
    MetricScores = Dict[
        Literal['Accuracy', 'F-Score', 'Precision', 'Recall'], Any
    ]
    CrossValidationResults = Tuple[
        Dict[
            Literal['Train', 'Test'], MetricScores
        ],
        pd.DataFrame
    ]
        

    def __init__(self) -> None:
        self.vectorSize: int
        self.vectorizer: SentencesVectorizer

        self.__trainX__: np.ndarray
        self.__testX__: np.ndarray

        self.__trainY__: np.ndarray
        self.__testY__: np.ndarray

        self.labels: Sequence[str]
        self.model: Optional[BookGenreClassifier.SupportedModel]

    @staticmethod
    def cleanDescription(s: str):
        s = ''.join(c.lower() if (ord(c) < 128 and c not in (string.punctuation + string.digits)) else ' ' for c in s)    
        return (' '.join((w for w in s.split() if w not in STOP_WORDS))).strip()

    @staticmethod
    def __shuffleXY__(x: np.ndarray, y: np.ndarray):
        if len(x) != len(y):
            raise AssertionError(f'Cannot unison-shuffle arrays of lengths [{len(x)}, {len(y)}]')
        
        ids = np.random.permutation(len(x))
        return x[ids], y[ids]

    def createTrainingData(self, booksDf: pd.DataFrame, vectorSize: int, scale: bool = False, shuffle: bool = True):
        self.vectorSize = vectorSize
        self.vectorizer = SentencesVectorizer(Word2Vec(
            sentences=[s.encode('utf-8').split() for s in booksDf[DataColumn.DESCRIPTION].values],
            vector_size=self.vectorSize
        ))

        cleanDesc = booksDf[DataColumn.DESCRIPTION].apply(self.cleanDescription)
        cleanDesc = cleanDesc.loc[lambda d: d != '']

        X = self.vectorizer.createVectors(cleanDesc)
        Y = booksDf[DataColumn.GENRESINGLE].loc[cleanDesc.index].values

        if shuffle:
            X, Y = self.__shuffleXY__(X, Y)

        self.labels = np.unique(Y)
        (
            self.__trainX__,
            self.__testX__,
            self.__trainY__,
            self.__testY__,
        ) = train_test_split(X, Y, train_size=0.8, shuffle=False)

        if scale:
            self.__trainX__ = StandardScaler().fit_transform(self.__trainX__)

        return X, Y
    
    @staticmethod
    def calculateMetricScores(yTrue: Sequence[str], yPred: Sequence[str]) -> MetricScores:
        
        pre, rec, f, _ = precision_recall_fscore_support(yTrue, yPred, average='macro')
        return {
            'Accuracy': accuracy_score(yTrue, yPred),
            'F-Score': f,
            'Precision': pre,
            'Recall': rec
        }

    def __doCrossValidation__(
        self,
        model: SupportedModel,
        trainX: np.ndarray,
        trainY: np.ndarray,
        testX: np.ndarray,
        testY: np.ndarray
    ) -> CrossValidationResults:
        
        self.model = model

        self.model.fit(trainX, trainY)
        trainPred = self.model.predict(trainX)
        testPred = self.model.predict(testX)

        results: Dict[Literal['Train', 'Test'], BookGenreClassifier.MetricScores] = {}
        for s, y, pred in (
            ('Train', trainY, trainPred),
            ('Test', testY, testPred)
        ):
            results[s] = runWithNoWarnings(lambda: self.calculateMetricScores(y, pred))

        df = pd.DataFrame.from_dict({
            s: {metric: results[s][metric] for metric in results[s]} for s in results
        }, orient='index')

        return results, df

    def performCrossValidation(self, model: SupportedModel):
        _, df = self.__doCrossValidation__(
            model,
            self.__trainX__,
            self.__trainY__,
            self.__testX__,
            self.__testY__
        )

        return df

    def performKFold(self, k: int, modelFactory: ModelFactory, verbose: bool = True):

        kfold = KFold(n_splits=k)

        foldItr = progressBarItr(
            itr=kfold.split(self.__trainX__),
            showBar=verbose,
            totalIterations=kfold.get_n_splits()
        )

        dfs = []
        for trainIds, testIds in foldItr:
            trainX = self.__trainX__[trainIds]
            trainY = self.__trainY__[trainIds]

            testX = self.__trainX__[testIds]
            testY = self.__trainY__[testIds]

            _, df = self.__doCrossValidation__(
                modelFactory(),
                trainX,
                trainY,
                testX,
                testY
            )
            dfs.append(df)

        result = dfs[0]
        for df in dfs[1:]:
            result += df
        
        result /= len(dfs)
        return result
