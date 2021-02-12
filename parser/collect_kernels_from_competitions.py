from io import StringIO
import subprocess

import pandas as pd

COMPETITION_REFS_FILE = "competitions_ref.csv"
comp_refs = None

for i in range(11):
    result = subprocess.run(
        args=["kaggle", "competitions", "list", "--csv", "-p", str(i + 1), "--sort-by", "numberOfTeams"],
        capture_output=True,
        encoding="utf-8"
    )

    if result.returncode != 0:
        print("Failed!")
        exit(0)

    competitions = pd.read_csv(StringIO(result.stdout), header=0)

    if i == 0:
        comp_refs = competitions
    else:
        comp_refs = pd.concat([comp_refs, competitions])

    for ref in competitions.ref:
        subprocess.run(["./competition_kernels.sh", ref])

comp_refs = comp_refs.drop(columns=["userHasEntered", "deadline", "category", "reward", "teamCount"])
comp_refs.to_csv(COMPETITION_REFS_FILE)
