import os.path
import csv
from collections import namedtuple

KeywordCustom = namedtuple("Keyword", ['name', 'category'])


class KeywordsDatabase:
    def __init__(self, filename):
        self.cats = set()
        self.keywords = []

        if not os.path.exists(filename):
            return

        with open(filename, "r", newline='', encoding="utf8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                keywordCustom = KeywordCustom(row["keyword"], row["category"])
                self.keywords.append(keywordCustom)
                self.cats.add(keywordCustom.category)

    def categories(self):
        return sorted(self.cats)
