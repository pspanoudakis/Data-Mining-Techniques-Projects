from IPython.display import display, Markdown

def printMd(s: str):
    """ Displays `s` as Markdown text in the output cell. """
    display(Markdown(s))
