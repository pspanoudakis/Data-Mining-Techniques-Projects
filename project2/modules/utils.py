from typing import Callable, TypeVar, Set, Iterable
from typing_extensions import ParamSpec

import warnings

import wordcloud
import nltk
import pandas as pd

from IPython.display import display, Markdown
from tqdm.notebook import tqdm

TRet = TypeVar('TRet')
TParams = ParamSpec('TParams')
def runWithNoWarnings(fn: Callable[TParams, TRet], *args: TParams.args, **kwargs: TParams.kwargs) -> TRet:
    """
        Executes `fn` with the specified `*args` & `**kwargs` while ignoring any raised warnings,
        and returns the function return value.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        return fn(*args, **kwargs)

def printMd(s: str):
    """ Displays `s` as Markdown text in the output cell. """
    display(Markdown(s))

T = TypeVar('T')
def progressBarItr(
    itr: Iterable[T],
    showBar: bool = True,
    totalIterations: "int | None" = None
) -> Iterable[T]:
    if showBar:
        return tqdm(
            itr,
            total=totalIterations,
            bar_format='{desc}{bar} {n}/{total} -- Time Elapsed: {elapsed}'
        )

    return itr

def printDatasetShape(dataset: pd.DataFrame):
    print(f'Dataset Shape:\nRows: {dataset.shape[0]}, Columns: {dataset.shape[1]}')

def getStopWordsSet() -> Set[str]:
    stopWordsSet = set()
    stopWordsSet.update(wordcloud.STOPWORDS)
    stopWordsSet.update(nltk.corpus.stopwords.words('english'))
    return stopWordsSet

STOP_WORDS = getStopWordsSet()

class DataColumn:
    """ Autocomplete (& typo prevention) helper for Column Titles. """

    BOOKID = 'bookId'
    TITLE = 'title'
    SERIES = 'series'
    AUTHOR = 'author'
    RATING = 'rating'
    DESCRIPTION = 'description'
    LANGUAGE = 'language'
    ISBN = 'isbn'
    GENRES = 'genres'
    CHARACTERS = 'characters'
    BOOKFORMAT = 'bookFormat'
    EDITION = 'edition'
    PAGES = 'pages'
    PUBLISHER = 'publisher'
    PUBLISHDATE = 'publishDate'
    FIRSTPUBLISHDATE = 'firstPublishDate'
    AWARDS = 'awards'
    NUMRATINGS = 'numRatings'
    RATINGSBYSTARS = 'ratingsByStars'
    LIKEDPERCENT = 'likedPercent'
    SETTING = 'setting'
    COVERIMG = 'coverImg'
    BBESCORE = 'bbeScore'
    BBEVOTES = 'bbeVotes'
    PRICE = 'price'
    RATINGSTAR1 = 'ratingStar1'
    RATINGSTAR2 = 'ratingStar2'
    RATINGSTAR3 = 'ratingStar3'
    RATINGSTAR4 = 'ratingStar4'
    RATINGSTAR5 = 'ratingStar5'
    PUBLISHYEAR = 'publishYear'
    GENRESINGLE = 'genreSingle'
