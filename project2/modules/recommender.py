from typing import TypeVar, Generic, Any, Sequence, Optional, Tuple, Callable, Iterable
from numbers import Number

from sortedcontainers import SortedKeyList
from collections import defaultdict
import textwrap

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from .utils import getStopWordsSet, DataColumn


IT = TypeVar('IT')
RT = TypeVar('RT')
class PairwiseCalculator(Generic[IT, RT]):

    def __init__(
        self,
        samples: Sequence[IT],
        calcFn: Callable[[IT, IT], RT],
        dtype: type
    ) -> None:
        
        self.__samples__ = samples
        self.__initResults__(len(self.__samples__), dtype=dtype)
        self.__calc__ = calcFn

    def __initResults__(self, numInstances: int, dtype: type):
        self.__results__ = np.empty(shape=numInstances, dtype=object)
        for i in range(numInstances):
            self.__results__[i] = np.empty(shape=(numInstances - (1 + i)), dtype=dtype)

    def calculate(self, resultProcessor: Optional[Callable[[int, int, RT], Optional[Any]]] = None):    
        for i in range(len(self.__samples__)):
            for j in range(i + 1, len(self.__samples__)):
                res = self.__calc__(self.__samples__[i], self.__samples__[j])
                self.__results__[i][j - (i + 1)] = res
                if resultProcessor:
                    resultProcessor(i, j, res)
        return self

T = TypeVar('T')
class MaxLengthSortedList(SortedKeyList, Generic[T]):
    
    @staticmethod
    def create(compareFn: Callable[[T, T], "Number | float"], maxLength: int):
        ret = MaxLengthSortedList([], key=compareFn)
        ret.setMaxLength(maxLength)
        return ret

    def setMaxLength(self, len: int):
        self.__maxLength__ = len

    def __discardExtra__(self):
        while len(self) > self.__maxLength__:
            super().pop(-1)

    def add(self, value: T):
        super().add(value)
        self.__discardExtra__()
    
    def update(self, iterable: Iterable[T]):
        super().update(iterable)
        self.__discardExtra__()

class BookRecommender:

    StoredSimilarity = Tuple[int, float]

    def __init__(self, books: pd.DataFrame, numBooks: int, numTop: int):
        
        bookDesc = books[DataColumn.DESCRIPTION]
        vectorizedDesc = self.__vectorizeDescriptions__(bookDesc[:min(numBooks, len(bookDesc))])

        self.__calculator__ = PairwiseCalculator(
            vectorizedDesc,
            cosine,
            np.float32
        )        

        self.__mostSimilar__: defaultdict[
            int,
            MaxLengthSortedList[BookRecommender.StoredSimilarity]
        ] = defaultdict(
            lambda: MaxLengthSortedList.create(
                compareFn=lambda s: -(self.getStoredSimilarityScore(s)),
                maxLength=numTop
            )
        )

        self.__books__ = books

    @staticmethod
    def getStoredSimilarityScore(
        s: StoredSimilarity
    ):
        return s[1]

    @staticmethod
    def __vectorizeDescriptions__(df: pd.DataFrame):
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # How to minimize dimensions?
        vectorizer = TfidfVectorizer(
            stop_words=list(getStopWordsSet()),
            ngram_range=(1, 1),
            max_df=0.85, min_df=0.01
        )
        
        # return vectorizer.fit_transform(raw_documents=df).toarray()
        return vectorizer.fit_transform(raw_documents=df).todense()

    def storeSimilarity(self, i: int, j: int, res: float):
        self.__mostSimilar__[i].add( (j, res) )
        self.__mostSimilar__[j].add( (i, res) )

    def findMostSimilar(self):
        self.__calculator__.calculate(self.storeSimilarity)

    def getRecommendationResult(self, idx: int, s: StoredSimilarity):
        book = self.__books__.loc[s[0]]
        txt = f"{idx}. {book[DataColumn.TITLE]}\nDescription: {book[DataColumn.DESCRIPTION]}\nScore: {s[1]}"
        return '\n'.join(
            textwrap.wrap(
                txt,
                width=80,
                break_long_words=False,
                replace_whitespace=False
            )
        )

    def showRecommendations(self, bookId: str, num: int = 5):
        foundIdx = np.where(self.__books__[DataColumn.BOOKID] == bookId)[0]

        res = []
        if len(foundIdx):

            idx = foundIdx[0]
            book = self.__books__.loc[idx]
            print(f"Recommending up to {num} books similar to: '{book[DataColumn.TITLE]}'")
            print('---------------------------------------\n')

            for i, s in enumerate(self.__mostSimilar__[idx]):
                res.append(self.getRecommendationResult(i + 1, s) + '\n')
                if i + 1 >= num:
                    break

            print('\n'.join(res))        
            print('---------------------------------------')
        else:
            raise Exception('Unknown `bookId` given.')        
