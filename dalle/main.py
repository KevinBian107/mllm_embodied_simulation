from pathlib import Path
from util import generate
import pandas as pd

PROMPT = list(pd.read_csv("affordance.csv").get("distinguishing_word"))

for i in PROMPT[:2:]:
    print(i + " is generated")
    generate(i)