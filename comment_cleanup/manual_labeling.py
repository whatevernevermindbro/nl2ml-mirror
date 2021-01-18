import curses

import numpy as np
import pandas as pd

import utils.preprocessing as preprocessing


LABELER_COUNT = 2
LABELER_ID = 0


def preprocess(code_blocks):
    prep_pipeline = [
        preprocessing.trim_symbols,
        preprocessing.single_lines,
        preprocessing.multiple_lines,
        preprocessing.extract_comments,
    ]
    
    for prep_func in prep_pipeline:
        code_blocks = code_blocks.apply(prep_func, axis=1)
    
    comments = []
    for block_comments in code_blocks["comments"]:
        comments.extend(block_comments)
    return comments
    

def load_comments():
    all_blocks = pd.read_csv("../data/code_blocks_clean.csv")
    all_blocks = all_blocks["code_block"].to_frame()

    comment_blocks_idx = (
        all_blocks["code_block"].str.contains("#") | 
        (all_blocks["code_block"].str.contains("'''") & 
         (all_blocks["code_block"].str.count("'''") % 2 == 0)) |
        (all_blocks["code_block"].str.contains('"""') & 
         (all_blocks["code_block"].str.count('"""') % 2 == 0))
    )
    
    return preprocess(all_blocks[comment_blocks_idx].reset_index())

    
def main(stdscr):
    break_line = "_" * 20

    comments = load_comments()

    comment_count = len(comments)
    comments_to_label = comment_count // LABELER_COUNT
    if LABELER_ID <= comment_count % LABELER_COUNT:
        comments_to_label += 1
    
    is_meaningful = np.full(comment_count, -1)

    is_session_over = False
    labeled_count = 0
    stdscr.clear()
    for comment_idx in range(LABELER_ID, comment_count, LABELER_COUNT):
        
        comment = comments[comment_idx]
        
        stdscr.erase()

        stdscr.addstr(f"Current block: {comment_idx}\n")
        stdscr.addstr(f"Labeled {labeled_count} of {comments_to_label}\n")
        stdscr.addstr(break_line + "\n")
        stdscr.addstr(comment[1])
        stdscr.addstr("\n" + break_line + "\n")
        stdscr.addstr("Is this a good comment? (y/n)\n")

        stdscr.refresh()
        label = stdscr.getkey()
        if label == "y":
            is_meaningful[comment_idx] = 1
        elif label == "n":
            is_meaningful[comment_idx] = 0
        else:
            is_session_over = True
            break
        
        labeled_count += 1
        
    np.save(f"labeled_comments_partition{LABELER_ID}.npy", is_meaningful)
    
    

curses.wrapper(main)
