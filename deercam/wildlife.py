"""Animals other than deer, by name. The AI says what a non-deer animal is in its own words ('wild turkey',
'feral hog', 'bird, likely a grackle'); this sorts those words into a fixed list the app can filter on.
A person's correction (detections.animal) wins over the AI's words."""
import re

# key: (label, plural, words that mean it). Checked in order, so 'turkey vulture' is a bird before it's a
# turkey and 'bobcat' is a bobcat before it's a cat.
KINDS = {
    "bird":      ("Bird", "Birds", ["vulture", "buzzard", "crow", "raven", "grackle", "blackbird", "cowbird", "starling",
                                    "hawk", "eagle", "owl", "jay", "cardinal", "dove", "pigeon", "quail", "heron",
                                    "egret", "bittern", "peacock", "peafowl", "guinea", "roadrunner", "woodpecker",
                                    "robin", "sparrow", "magpie", "goose", "duck", "catbird", "bird"]),
    "turkey":    ("Turkey", "Turkeys", ["turkey"]),
    "hog":       ("Hog", "Hogs", ["hog", "pig", "boar", "swine", "piglet", "javelina", "peccary"]),
    "coyote":    ("Coyote", "Coyotes", ["coyote"]),
    "bobcat":    ("Bobcat", "Bobcats", ["bobcat", "lynx", "mountain lion", "cougar", "puma"]),
    "fox":       ("Fox", "Foxes", ["fox", "foxes"]),
    "dog":       ("Dog", "Dogs", ["dog", "wolf"]),
    "cat":       ("Cat", "Cats", ["cat"]),
    "raccoon":   ("Raccoon", "Raccoons", ["raccoon"]),
    "opossum":   ("Opossum", "Opossums", ["opossum", "possum"]),
    "armadillo": ("Armadillo", "Armadillos", ["armadillo"]),
    "skunk":     ("Skunk", "Skunks", ["skunk"]),
    "rabbit":    ("Rabbit", "Rabbits", ["rabbit", "hare", "cottontail", "jackrabbit"]),
    "squirrel":  ("Squirrel", "Squirrels", ["squirrel"]),
    "cattle":    ("Cattle", "Cattle", ["cow", "cattle", "calf", "calves", "bull", "steer", "heifer"]),
    "horse":     ("Horse", "Horses", ["horse", "donkey", "mule"]),
    "other":     ("Other animal", "Other animals", []),
}
_WORDS = [(k, re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")s?\b", re.I)) for k, (_, _, words) in KINDS.items() if words]


def classify(text: str | None) -> str:
    """The AI's words for an animal -> a key of KINDS ('other' when nothing matches)."""
    for key, rx in _WORDS:
        if text and rx.search(text):
            return key
    return "other"


def listing() -> list[dict]:
    return [{"key": k, "label": label, "plural": plural} for k, (label, plural, _) in KINDS.items()]
