from io import StringIO
import subprocess

import pandas as pd


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

    for ref in competitions.ref:
        subprocess.run(["./competition_kernels.sh", ref])
