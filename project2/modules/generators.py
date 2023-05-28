from typing import Iterable, List, TypeVar, Generic, Optional

from abc import ABC, abstractmethod
from collections import defaultdict

import pandas as pd

from IPython.display import display

from .questions import AnswerGeneratorQuestion, TopColumnValuesQuestion
from .utils import printMd

QT = TypeVar("QT", bound=AnswerGeneratorQuestion)
class AnswerGenerator(ABC, Generic[QT]):
    """
        Used to generate `DataFrame` answers on Dataset-related questions.
    """

    def __init__(self) -> None:
        self.questions: List[QT] = []

    def registerQuestions(self, questions: Iterable[QT]):
        self.questions = list(questions)
        return self

    def registerQuestion(self, q: QT):
        self.questions.append(q)
        return self

    def clearQuestions(self):
        self.questions.clear()
        return self
    
    @abstractmethod
    def createDataFrameAnswer(self, q: QT, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        pass

    def generateAnswer(self, df: pd.DataFrame, q: QT):
        """
            Generates the `pd.Dataframe` answer for question `q`, using data in `df`.

            This method applies any data filtering and/or answer size restrictions specified in `q`,
            essentially acting as a wrapper for `createDataFrameAnswer`,
            which is expected to be implemented by child classes.
        """

        q['dataFilter'] = q.get('dataFilter')
        q['limit'] = int(q.get('limit') or 0)
        
        filteredDf = q['dataFilter'](df) if q['dataFilter'] else df
        answer = self.createDataFrameAnswer(q, filteredDf)
        limit = q['limit'] if q['limit'] else answer.shape[0]
        return answer.head(n=limit)

    def displayAnswers(self, df: pd.DataFrame):
        """
            Displays the answer for each question provided to the
            `AnswerGenerator`, using data in `df`.

            The answer includes an inline markdown string,
            as well as the results `DataFrame`.
        """
        
        for q in self.questions:
            printMd(f'**{q.get("qnum")}**. {q.get("qtitle")}:')
            display(self.generateAnswer(df, q))

class TopColumnValuesAnswerGenerator(AnswerGenerator[TopColumnValuesQuestion]):
    """
        Used to generate `DataFrame` answers on questions regarding
        the top values of a primary column (e.g. top Authors)
        based on their respective total value in a secondary column (e.g. total Ratings).
    """

    COUNT_ROWS = ''
    
    def __init__(self) -> None:
        super().__init__()
        self.__data__: "pd.Series[str]"

    def __extractColumnValues__(self, df: pd.DataFrame, q: TopColumnValuesQuestion):
        q['extractor'] = q.get('extractor')
        if q['extractor']:
            self.__data__ =  df[q['col']].apply(q['extractor'])
        else:
            self.__data__ = df[q['col']]

    def createDataFrameAnswer(self, q: TopColumnValuesQuestion, df: pd.DataFrame):
        q['useExtracted'] = (q.get('useExtracted') or False)
        if not q['useExtracted']:
            self.__extractColumnValues__(df, q)

        valueCount: defaultdict[str, int] = defaultdict(lambda: 0)

        q['counter'] = q.get('counter')
        if q['countCol'] == self.COUNT_ROWS:

            if q['counter']:
                for v in self.__data__:
                    q['counter'](valueCount, v, None)
                
                ans = pd.DataFrame(
                    valueCount.items(),
                    columns=[q['col'], 'Books']
                ).sort_values(by='Books', ascending=False)
                ans.reset_index(inplace=True, drop=True)

            else:
                ans = self.__data__.value_counts().sort_values(ascending=False).to_frame().reset_index()
                ans.columns = [q['col'], 'Books']            
        else:
            col = q['countCol']
            
            for key, colValue in zip(self.__data__, df[col]):
                if q['counter']:
                    q['counter'](valueCount, key, colValue)
                else:
                    valueCount[key] += colValue

            countColStr = f'Total {q["countCol"]}'
            ans = pd.DataFrame(
                    valueCount.items(),
                    columns=[q['col'], countColStr]
                ).sort_values(by=countColStr, ascending=False)
            ans.reset_index(inplace=True, drop=True)
        
        ans.index += 1
        return ans
    