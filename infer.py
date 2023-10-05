from transformers import pipeline
from pathlib import Path
import time

here = Path(__file__).parent

classifier = pipeline("ner", model= here / "./model/checkpoint-548", aggregation_strategy="first")

clock = time.time()
print(classifier("1 large wakandan almond"))
print("Time elapsed: " + str(time.time() - clock))