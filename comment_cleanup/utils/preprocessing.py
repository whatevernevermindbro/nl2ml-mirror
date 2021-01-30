import re


def all_occur(text, substr):
    return [m.start() for m in re.finditer(substr, text)]


def greater_than_in(value, l, if_none):
    try:
        return next(y[1] for y in enumerate(l) if y[1] > value)
    except:
        return if_none


def max_less(value ,l, if_none):
    try:
        return max([x for x in l if x < value])
    except:
        return if_none


# проверка строки на наличие комментария
def is_comment(text):
    return True


def change_position(code, code_new, ar):
    dif = len(code) - len(code_new)
    ar = [pos - dif for pos in ar]
    return ar


def trim_symbols(code_chunk):
    code = code_chunk["code_block"]

    pattern1 = r"(#\s*)+"
    pattern2 = r"[#]+"
    
    p1 = re.compile(pattern1)
    p2 = re.compile(pattern2)
    
    code = p2.sub('#', p1.sub('#', code))
    
    # исправляем неправильно заданные многострочные комментарии
    code = code.replace("''''", "'''")
    code = code.replace('""""', '"""')
    
    # убираем последний символ - ','
    code = code[:-1]
    
    # для случаев '''code_chunk''' убираем скобочки
    #if code.strip()[:3] == code.strip()[-3:]:
    #    code = code.strip()[3:-3]
    
    code_chunk["code_block"] = code
    return code_chunk


def single_lines(code_chunk):
    code = code_chunk['code_block']
    
    singles = all_occur(code, '#')
    newlines = all_occur(code, "\n")
    code_new = ""
    
    while len(singles) != 0:
        codelen = len(code)
        comment_start = singles.pop(0)
        
        comment_end = greater_than_in(comment_start, newlines, codelen)
        comment = code[comment_start:comment_end]
        
        prev_newline = max_less(comment_start, newlines, 0)
        before = code[prev_newline:comment_start]
        before_indent = before[:(len(before) - len(before.lstrip()))]
        before = before.strip()

        comment = code[comment_start:comment_end]
        
        # перемещаем однострочные комментарии, перед которыми есть код
        if (before != ""):
            # проверяем, чтобы # не содержался внутри строки
            # проверяем, чтобы # не был закоментирован
            if ((before.count("'") % 2 != 1) and
                (before.count('"') % 2 != 1) and
                (code[:comment_start].count("'''") %  2 != 1)):
                
                code_new = code[:prev_newline] + "\n'''\n" + comment[1:] + "\n'''" + \
                           before_indent + before + code[comment_end:]
                
                singles = change_position(code, code_new, singles)
                newlines = change_position(code, code_new, newlines)
                
                code = code_new
    
        # превращаем однострочные комментарии в начале строки в многострочные
        else:
            # проверяем, чтобы # не был закоментирован
            if (code[:comment_start].count("'''") %  2 != 1):
                code_new = code[:prev_newline] + "\n'''\n" + comment[1:] + "\n'''" + \
                               before_indent + before + code[comment_end:]
                singles = change_position(code, code_new, singles)
                newlines = change_position(code, code_new, newlines)
                code = code_new

    code_chunk['code_block'] = code
    return code_chunk


def multiple_lines(code_chunk):
    code = code_chunk['code_block']
    
    code_new = ""
    
    multiples = all_occur(code, "'''")
    
    code_chunk['code_block'] = code
    return code_chunk

def extract_comments(code_chunk):
    code = code_chunk['code_block']

    # добавляем комментарии в отдельный столбец
    comments = []
    multiples = all_occur(code, "'''")
    
    while len(multiples) > 0:
        comment_start = multiples.pop(0)
        comment_end = multiples.pop(0)
        comment = code[comment_start + 3:comment_end]
        
        #comment = comment.replace("\t", "    ")

        #code_new = code[:comment_start] + code[comment_end + 3:]
        #multiples = change_position(code, code_new, multiples)
        #code = code_new
        
        if (len(comment) > 0) and (is_comment(comment)):
            comments.append((comment_start, comment))
    
    code_chunk['code_block'] = code
    code_chunk['comments'] = comments
    
    return code_chunk
