from util import generate
import pandas as pd

raw = pd.read_csv("affordance.csv").get("distinguishing_word")
PROMPT = list(raw)
print(len(PROMPT))

for i in PROMPT[:54:]:
    print(i + " is generated")
    generate(i)