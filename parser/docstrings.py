import ast

def get_docstrings_from_code(codeblock: str) -> str:
    docstrings = []
    tree = ast.parse(codeblock)
    for child in ast.iter_child_nodes(tree):
        try:
            docstrings.append(ast.get_docstring(child))
        except:
            pass
    return docstrings


def bad_way_to_do_stuff(func, *args):
    try:
        return func(*args)
    except:
        return -1