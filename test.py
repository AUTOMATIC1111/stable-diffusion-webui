import re

re_attention = re.compile(r"""
\(|\[|\\\(|\\\[|\\|\\\\|
:([+-]?[.\d]+)|
\)|\]|\\\)|\\\]|
[^\(\)\[\]:]+|
:
""", re.X)

texts = ["car:2.0", "(car:1.1)", "((car:1.2))", "[car:0.9]", "[[car:0.8]]"]
for text in texts:
    for m in re_attention.finditer(text):
        print(text, '0:', m.group(0), '1:', m.group(1))
