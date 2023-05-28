from typing import TypedDict, Callable, Optional, Any
from typing_extensions import NotRequired

from collections import defaultdict

import pandas as pd

class AnswerGeneratorQuestion(TypedDict):
    """ Used to store question-specific information. """
    
    qnum: NotRequired[str]
    """ Question Number. """
    qtitle: NotRequired[str]
    """ Question Title. """
    col: str
    """ Column title. """
    dataFilter: NotRequired[
        Optional[
            Callable[[pd.DataFrame], pd.DataFrame]
        ]
    ]
    """  """
    limit: NotRequired[int]
    """ Limit for number of results. """

class TopColumnValuesQuestion(AnswerGeneratorQuestion):
    """
        A question regarding the top values of a primary column (e.g. top Authors) 
        based on their respective total value in a secondary column (e.g. total Ratings).
    """

    countCol: str
    """
        The secondary column. If `== TopColumnValuesAnswerGenerator.NO_COUNT_COLUMN`,
        just count the total rows for each primary column value (`pd.Dataframe.value_counts()`)
    """

    extractor: NotRequired[
        Optional[
            Callable[[str], str]
        ]
    ]
    """"
        If specified, it is used as custom logic when extracting
        the primary column values (e.g. Author names).
    """

    counter: NotRequired[
        Optional[
            Callable[["defaultdict[str, int]", str, Optional[Any]], None]
        ]
    ]
    """ If specified, it is used as custom logic when counting the total values. """

    useExtracted: NotRequired[bool]
    """
        If specified (default: `False`), the generator will use the primary column values that have already been extracted.
        This is handy in case there are consecutive questions regarding the same primary column.
    """
