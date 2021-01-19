import keyword


# Non-vectorized functions
def digits_portion(text):
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    count = 0
    for d in digits:
        count += text.count(d)
    return count / len(text)


def keyword_count(text):
    count = 0
    words = text.split()
    for word in keyword.kwlist:
        count += words.count(word)
    return count


def word_count(text):
    return len(text.split())


def escape_sequence_count(text):
    seqs = ['\n','\t']
    count = 0
    for seq in seqs:
        count += text.count(seq)
    return count


def can_compile(text):
    try:
        compile(text, "bogusfile.py", "exec")
        return 1
    except Exception as e:
        return 0


def brackets_count(text):
    symbols = ['(', ')', '[', ']', '{', '}']
    count = 0
    for symbol in symbols:
        count += text.count(symbol)
    return count


def special_symbols_count(text):
    symbols = ['!', '@', '$', '%', '^', '&', '*', '-', '+', '~', '/', '|', '\\', ':']
    count = 0
    for symbol in symbols:
        count += text.count(symbol)
    return count


def uppercase_portion(text):
    count = 0
    for symbol in text:
        if symbol.isupper():
            count += 1
    return count / len(text)


def lowercase_portion(text):
    count = 0
    for symbol in text:
        if symbol.islower():
            count += 1
    return count / len(text)


def explanation_words_count(text):
    wordlist = ["count", "plot", "calculate", "build", "create"]
    count = 0
    for word in wordlist:
        count += text.count(word)
    return count

    
def preprocess_comments(comments_df):
    prep_funcs = [
        digits_portion,
        len,
        keyword_count,
        word_count,
        escape_sequence_count,
        can_compile,
        special_symbols_count,
        brackets_count,
        uppercase_portion,
        lowercase_portion,
        explanation_words_count,
    ]
    
    for feature_func in prep_funcs:
        feature_name = feature_func.__name__
        comments_df[feature_name] = comments_df["comment"].apply(feature_func)
    
    return comments_df

