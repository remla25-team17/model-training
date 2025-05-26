from pylint.checkers import BaseChecker


class UnnecessaryIterationChecker(BaseChecker):
    """
        Pylint checker that detects and warns about unnecessary use of iteration
        ('for' and 'while' loops), encouraging vectorized solutions.

        This checker is particularly relevant in Machine Learning (ML) contexts
        where iterating over data can be inefficient compared to vectorized
        operations provided by libraries like NumPy or Pandas.
        """
    

    name = "unnecessary-iteration-checker"
    msgs = {
        "W5001": (
            "Unnecessary use of iteration detected, use vectorized solution instead",
            "unnecessary-iteration-used",
            "It is better to adopt a vectorized solution instead of iterating over data in ML applications.",
        ),
    }

    def visit_for(self, node):
        """
        Called when a 'for' loop node is visited in the Abstract Syntax Tree (AST).

        Reports a warning for the presence of unnecessary iteration, indicating
        a potential opportunity for vectorized code.

        :param node: The 'for' loop AST node.
        :type node: astroid.For
        """
        self.add_message("unnecessary-iteration-used", node=node)

    def visit_while(self, node):
        """
        Called when a 'while' loop node is visited in the Abstract Syntax Tree (AST).

        Reports a warning for the presence of unnecessary iteration, indicating
        a potential opportunity for vectorized code.

        :param node: The 'while' loop AST node.
        :type node: astroid.While
        """
        self.add_message("unnecessary-iteration-used", node=node)

def register(linter):
    """
    Registers the UnnecessaryIterationChecker with the Pylint linter.

    This function is the entry point for Pylint to discover and load
    custom checkers.

    :param linter: The Pylint Linter object.
    """
    linter.register_checker(UnnecessaryIterationChecker(linter))