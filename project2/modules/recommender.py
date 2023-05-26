from typing import TypeVar, Generic, Any, Sequence, Optional, Tuple, Callable, Iterable
from numbers import Number
import string
from sortedcontainers import SortedKeyList
from collections import defaultdict
import textwrap

import numpy as np
import pandas as pd
#from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine

from .utils import getStopWordsSet, DataColumn, printMd


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
        self.__discardExtra__()

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

    StoredSimilarity = Tuple[int, np.double]

    def __init__(self, books: pd.DataFrame, numBooks: int, numTop: int):
        
        bookDesc = books[DataColumn.DESCRIPTION]
        vectorizedDesc = self.__vectorizeDescriptions__(bookDesc[:min(numBooks, len(bookDesc))])

        self.__calculator__ = PairwiseCalculator(
            samples=vectorizedDesc,
            calcFn=self.calculateSimilarity,
            dtype=np.double
        )        

        self.__mostSimilar__: defaultdict[
            int,
            MaxLengthSortedList[BookRecommender.StoredSimilarity]
        ] = defaultdict(
            lambda: MaxLengthSortedList.create(
                # Using negative key to sort in descending order
                compareFn=lambda s: -(self.getStoredSimilarityScore(s)),
                maxLength=numTop
            )
        )

        self.__books__ = books

    @staticmethod
    def calculateSimilarity(x: np.ndarray, y: np.ndarray) -> np.double:
        # return cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))

        # Basically, 1 - `cosine_similarity` = `cosine`
        # https://stackoverflow.com/a/55623717/12184528

        # `cosine` is MUCH faster for 1-D vectors:
        # https://stackoverflow.com/questions/61490351/scipy-cosine-similarity-vs-sklearn-cosine-similarity
        cos = cosine(x, y)
        if cos == 0:
            # If cos == 0, probably a division by 0 occured (which should generate a warning as well),
            # most likely due to precision loss.
            return 0
        else:
            return 1 - cos

    @staticmethod
    def getStoredSimilarityScore(
        s: StoredSimilarity
    ):
        return s[1]
    
    @staticmethod
    def __preprocessText__(s: str):
        return s.translate(str.maketrans('', '', string.punctuation)).lower()

    @staticmethod
    def __vectorizeDescriptions__(df: pd.DataFrame):
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(
            stop_words=list(getStopWordsSet()),
            ngram_range=(1, 1),
            max_df=0.85, min_df=0.01,
            strip_accents='unicode'
        )
        
        return vectorizer.fit_transform(raw_documents=list(df)).toarray()

    def storeSimilarity(self, i: int, j: int, res: np.double):
        self.__mostSimilar__[i].add( (j, res) )
        self.__mostSimilar__[j].add( (i, res) )

    def findMostSimilar(self):
        self.__calculator__.calculate(self.storeSimilarity)

    def __displayBookRecommendation__(self, recIdx: int, s: StoredSimilarity):

        recBook = self.__books__.loc[s[0]]

        printMd(f"{recIdx}. **{recBook[DataColumn.TITLE]}**")

        desc = '\\\n'.join(
            textwrap.wrap(
                recBook[DataColumn.DESCRIPTION],
                width=100,
                break_long_words=False,
                replace_whitespace=False,
                fix_sentence_endings=True
            )
        )
        printMd(f"<u>Description</u>: {desc}")

        printMd(f"<u>Score</u>: {s[1]}")

    def showRecommendations(self, bookId: str, num: int = 5):
        foundIdx = np.where(self.__books__[DataColumn.BOOKID] == bookId)[0]

        if len(foundIdx):

            idx = foundIdx[0]
            book = self.__books__.loc[idx]
            printMd(f"Recommending up to {num} books similar to **{book[DataColumn.TITLE]}**")
            printMd('***')

            for i, s in enumerate(self.__mostSimilar__[idx]):
                self.__displayBookRecommendation__(i + 1, s)
                if i + 1 >= num:
                    break
     
            printMd('***')
        else:
            raise Exception('Unknown `bookId` given.')        
