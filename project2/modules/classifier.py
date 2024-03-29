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
    """
        A helper class to generate vectors for sentences, using a pretrained
        `Word2Vec` model.

        Each sentence is assigned the mean vector of the words included in it
        (if it is splitted).
    """
    def __init__(self, word2vec: Word2Vec) -> None:
        self.vectors: np.ndarray
        self.__word2vec__ = word2vec

    def getVectorSize(self):
        return self.__word2vec__.vector_size

    def __saveSentenceVector__(self, idx: int, s: str):
        words = s.split()
        for w in words:
            self.vectors[idx] += self.__word2vec__.wv[w]

        self.vectors[idx] /= max(len(words), 1)
    
    def createVectors(self, sentences: pd.Series):
        self.vectors = np.zeros(shape=(len(sentences), self.getVectorSize()))
        for i, s in enumerate(sentences):
            self.__saveSentenceVector__(i, s)

        return self.vectors

class BookGenreClassifier:
    """
        A classifier for finding the genre of books based on their description texts.

        The classifier acts as a wrapper for the sentence vectorization process,
        the (K-Fold) cross-validation logic, and the metric scores calculation.

        Any further dependencies, such as the scikit-learn estimator to be used
        for classification, are injected.
    """

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

    def createTrainingData(self, booksDf: pd.DataFrame, vectorSize: int, scale: bool = False, shuffle: bool = True):
        """
            Vectorizes the book description sentences in `booksDf`,
            using `Word2Vec` vectors with the given `vectorSize`.

            After vectorization, it creates train & test sets using `train_test_split`.
            It returns the generated `X` and `Y` `np.ndarray`'s (without splitting) for
            external use.
        """

        self.vectorSize = vectorSize

        cleanDesc = booksDf[DataColumn.DESCRIPTION].apply(self.cleanDescription)
        cleanDesc = cleanDesc.loc[lambda d: d != '']

        self.vectorizer = SentencesVectorizer(Word2Vec(
            sentences=[s.split() for s in cleanDesc],
            vector_size=self.vectorSize,
            window=10, min_count=1
        ))        

        X = self.vectorizer.createVectors(cleanDesc)
        Y = booksDf[DataColumn.GENRESINGLE].loc[cleanDesc.index].values

        self.labels = np.unique(Y)
        (
            self.__trainX__,
            self.__testX__,
            self.__trainY__,
            self.__testY__,
        ) = train_test_split(X, Y, train_size=0.8, shuffle=shuffle)

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
        """
            Performs cross-validation on the given `model`, using the pre-stored
            train & test data.

            It returns a `DataFrame` with all the metric scores for both train & test sets.
        """

        _, df = self.__doCrossValidation__(
            model,
            self.__trainX__,
            self.__trainY__,
            self.__testX__,
            self.__testY__
        )

        return df

    def performKFold(self, k: int, modelFactory: ModelFactory, verbose: bool = True):
        """
            Performs `k`-fold cross-validation, using the pre-stored train & test data.
            In each fold, a new estimator is instantiated using `modelFactory`.
        """

        kfold = KFold(n_splits=k)

        foldItr = progressBarItr(
            itr=kfold.split(self.__trainX__),
            showBar=verbose,
            totalIterations=kfold.get_n_splits()
        )

        dfs = []
        for i, (trainIds, testIds) in enumerate(foldItr):
            trainX = self.__trainX__[trainIds]
            trainY = self.__trainY__[trainIds]

            testX = self.__trainX__[testIds]
            testY = self.__trainY__[testIds]

            foldResults, df = self.__doCrossValidation__(
                modelFactory(),
                trainX,
                trainY,
                testX,
                testY
            )
            #print(f'{i}: {foldResults}')
            dfs.append(df)

        result = dfs[0]
        for df in dfs[1:]:
            result += df
        
        result /= len(dfs)
        return result
