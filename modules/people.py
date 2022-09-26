import os.path
import csv
from collections import namedtuple

People = namedtuple("People", ['name', 'category'])


class PeopleDatabase:
    def __init__(self, filename):
        self.cats = set()
        self.people = []

        if not os.path.exists(filename):
            return

        with open(filename, "r", newline='', encoding="utf8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                person = People(row["person"], row["category"])
                self.people.append(person)
                self.cats.add(person.category)

    def categories(self):
        return sorted(self.cats)
