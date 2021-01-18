import keyword


def _digits_portion(text):
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    count = 0
    for d in digits:
        count += text.count(d)
    return count / len(text)


def _keyword_count(text):
    count = 0
    words = text.split()
    for keyword in keyword.kwlist:
        count += words.count(keyword)
    return count


def _word_count(text):
    return len(text.split())


def _escape_sequence_count(text):
    seqs = ['\n','\t']
    count = 0
    for seq in seqs:
        count += text.count(seq)
    return count


def _can_compile(text):
    try:
        compile(text, "bogusfile.py", "exec")
        return 1
    except Exception as e:
        return 0


def _brackets_count(text):
    symbols = ['(', ')', '[', ']', '{', '}']
    count = 0
    for symbol in symbols:
        count += text.count(symbol)
    return count


def _special_symbols_count(text):
    symbols = ['!', '@', '$', '%', '^', '&', '*', '-', '+', '~', '/', '|', '\\']
    count = 0
    for symbol in symbols:
        count += text.count(symbol)
    return count


def _uppercase_portion(text):
    count = 0
    for symbol in text:
        if symbol.isupper():
            count += 1
    return count / len(text)


def _lowercase_portion(text):
    count = 0
    for symbol in text:
        if symbol.islower():
            count += 1
    return count / len(text)
