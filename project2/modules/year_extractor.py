from datetime import datetime
import re

class YearExtractor:

    UNKNOWN_YEAR = -1
    CURRENT_CENT = (datetime.now().year // 100) * 100
    CURRENT_CENT_YEAR = (datetime.now().year % 100)

    @staticmethod
    def sanitizeYear(y: int):
        if y // 1000 > 0:
            return y
        if YearExtractor.CURRENT_CENT_YEAR > y:
            return YearExtractor.CURRENT_CENT + y
        else:
            return YearExtractor.CURRENT_CENT - 100 + y

    FULL_YEAR_GROUP = 'yyyy'
    SMALL_YEAR_GROUP = 'yy'

    # This will match:
    # - Either a full year string (like '1999') and store it in 'fullYear' group
    # - Or a date MM/DD/YY and store only 'YY' in 'fullYear' group
    YEAR_RE = (
        fr"(?P<{FULL_YEAR_GROUP}>[1-9][0-9]{{3}})"
        fr"|"
        fr"(([0-9]{{2}}[/]){{2}}(?P<{SMALL_YEAR_GROUP}>[0-9]{{2}}))"
    )

    # Same pattern but without any grouping
    YEAR_RE_NO_GROUP = (
        r"(([1-9][0-9]{3})|(([0-9]{2}[/]){2}[0-9]{2}))"
    )

    @staticmethod
    def extractYear(s: str):
        match = re.search(YearExtractor.YEAR_RE, s)
        if match:
            if match.group(YearExtractor.FULL_YEAR_GROUP):
                return int(match.group(YearExtractor.FULL_YEAR_GROUP))
            else:
                return YearExtractor.sanitizeYear(int(match.group(YearExtractor.SMALL_YEAR_GROUP)))
            
        return YearExtractor.UNKNOWN_YEAR
