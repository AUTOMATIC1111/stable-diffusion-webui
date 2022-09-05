import os.path
import csv
from collections import namedtuple

Artist = namedtuple("Artist", ['name', 'weight', 'category'])


class ArtistsDatabase:
    def __init__(self, filename):
        self.cats = set()
        self.artists = []

        if not os.path.exists(filename):
            return

        with open(filename, "r", newline='', encoding="utf8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                artist = Artist(row["artist"], float(row["score"]), row["category"])
                self.artists.append(artist)
                self.cats.add(artist.category)

    def categories(self):
        return sorted(self.cats)
