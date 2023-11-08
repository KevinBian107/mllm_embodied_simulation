from pathlib import Path
from util import generate
import pandas as pd

PROMPT = list(pd.read_csv("affordance.csv").get("distinguishing_word"))
print(len(PROMPT))

for i in PROMPT:
    print(i + " is generated")
    generate(i)