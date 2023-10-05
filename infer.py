from transformers import pipeline
from pathlib import Path
import time

here = Path(__file__).parent

classifier = pipeline("ner", model= here / "./model", aggregation_strategy="first")

clock = time.time()
print(classifier("half a cup of chopped onion"))
print("Time elapsed: " + str(time.time() - clock))