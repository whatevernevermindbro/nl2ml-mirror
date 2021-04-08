import ast


def get_docstrings_from_code(codeblock: str) -> list:
    docstrings = []
    tree = ast.parse(codeblock)
    for child in ast.iter_child_nodes(tree):
        try:
            docstrings.append(ast.get_docstring(child))
        except:
            pass
    return docstrings

