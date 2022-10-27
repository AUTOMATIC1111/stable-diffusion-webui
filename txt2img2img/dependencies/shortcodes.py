import re

__version__ = "5.4.0"


# Globally-registered handler functions indexed by keyword.
global_keywords = {}


# The set of all end-words for globally-registered block-scoped shortcodes.
global_endwords = set()


# Decorator function for globally registering shortcode handlers.
def register(keyword, endword=None):

    def register_function(func):
        global_keywords[keyword] = (func, endword)
        if endword:
            global_endwords.add(endword)
        return func

    return register_function


# ------------------- #
#  Exception Classes  #
# ------------------- #


# Base class for all exceptions raised by the library.
class ShortcodeError(Exception):
    pass


# Raised if the parser detects invalid shortcode syntax.
class ShortcodeSyntaxError(ShortcodeError):
    pass


# Raised if a handler function throws an error.
class ShortcodeRenderingError(ShortcodeError):
    pass


# ----------- #
#  AST Nodes  #
# ----------- #


# Input text is parsed into a tree of Node instances.
class Node:

    def __init__(self):
        self.children = []

    def render(self, context):
        return ''.join(child.render(context) for child in self.children)


# Represents ordinary text not enclosed in tag delimiters.
class Text(Node):

    def __init__(self, text):
        self.text = text

    def render(self, context):
        return self.text


# Base class for atomic and block-scoped shortcodes.
class Shortcode(Node):

    # Regex for parsing the shortcode's arguments.
    re_args = re.compile(r"""
        (?:([^\s'"=]+)=)?
        (
            "((?:[^\\"]|\\.)*)"
            |
            '((?:[^\\']|\\.)*)'
        )
        |
        ([^\s'"=]+)=(\S+)
        |
        (\S+)
    """, re.VERBOSE)

    def __init__(self, token, handler_function):
        self.token = token
        self.handler = handler_function
        self.pargs, self.kwargs = self.parse_args(token.text[len(token.keyword):])
        self.children = []

    def parse_args(self, argstring):
        pargs, kwargs = [], {}
        for match in self.re_args.finditer(argstring):
            if match.group(2) or match.group(5):
                key = match.group(1) or match.group(5)
                value = match.group(3) or match.group(4) or match.group(6)
                if key:
                    kwargs[key] = value
                else:
                    pargs.append(value)
            else:
                pargs.append(match.group(7))
        return pargs, kwargs


# An atomic shortcode is a shortcode with no closing tag.
class AtomicShortcode(Shortcode):

    # If the shortcode handler raises an exception we intercept it and wrap it
    # in a ShortcodeRenderingError. The original exception will still be
    # available via the exception's __cause__ attribute.
    def render(self, context):
        try:
            return str(self.handler(self.pargs, self.kwargs, context))
        except Exception as ex:
            msg = f"An exception was raised while rendering the "
            msg += f"'{self.token.keyword}' shortcode in line {self.token.line_number}."
            raise ShortcodeRenderingError(msg) from ex


# A block-scoped shortcode is a shortcode with a closing tag.
class BlockShortcode(Shortcode):

    # If the shortcode handler raises an exception we intercept it and wrap it
    # in a ShortcodeRenderingError. The original exception will still be
    # available via the exception's __cause__ attribute.
    def render(self, context):
        content = ''.join(child.render(context) for child in self.children)
        try:
            return str(self.handler(self.pargs, self.kwargs, context, content))
        except Exception as ex:
            msg = f"An exception was raised while rendering the "
            msg += f"'{self.token.keyword}' shortcode in line {self.token.line_number}."
            raise ShortcodeRenderingError(msg) from ex


# -------- #
#  Parser  #
# -------- #


# A Parser instance parses input text and renders shortcodes. A single Parser
# instance can parse an unlimited number of input strings. Note that the parse()
# method accepts an optional arbitrary context object which it passes on to each
# shortcode's handler function.
#
# If the `inherit_globals` parameter is true, the parser will inherit a copy of
# the set of globally-registered shortcodes at the moment of instantiation.
#
# If `ignore_unknown` is true, unknown shortcodes are ignored. If this parameter
# is false (the default), unknown shortcodes cause an error.
class Parser:

    def __init__(self, start='[%', end='%]', esc='\\', inherit_globals=True, ignore_unknown=False):
        self.start = start
        self.end = end
        self.esc_start = esc + start
        self.keywords = global_keywords.copy() if inherit_globals else {}
        self.endwords = global_endwords.copy() if inherit_globals else set()
        self.ignore_unknown = ignore_unknown

    def register(self, func, keyword, endword=None):
        self.keywords[keyword] = (func, endword)
        if endword:
            self.endwords.add(endword)

    def parse(self, text, context=None):
        if not self.start in text:
            return text

        stack  = [Node()]
        expecting = []

        lexer = Lexer(text, self.start, self.end, self.esc_start)
        for token in lexer.tokenize():
            if token.type == "TEXT":
                stack[-1].children.append(Text(token.text))
            elif token.keyword in self.keywords:
                handler, endword = self.keywords[token.keyword]
                if endword:
                    node = BlockShortcode(token, handler)
                    stack[-1].children.append(node)
                    stack.append(node)
                    expecting.append(endword)
                else:
                    node = AtomicShortcode(token, handler)
                    stack[-1].children.append(node)
            elif token.keyword in self.endwords:
                if len(expecting) == 0:
                    msg = f"Unexpected '{token.keyword}' tag in line {token.line_number}."
                    raise ShortcodeSyntaxError(msg)
                elif token.keyword == expecting[-1]:
                    stack.pop()
                    expecting.pop()
                else:
                    msg = f"Unexpected '{token.keyword}' tag in line {token.line_number}. "
                    msg += f"The shortcode parser was expecting a closing '{expecting[-1]}' tag."
                    raise ShortcodeSyntaxError(msg)
            elif token.keyword == '':
                msg = f"Empty shortcode tag in line {token.line_number}."
                raise ShortcodeSyntaxError(msg)
            elif self.ignore_unknown:
                stack[-1].children.append(Text(token.raw_text))
            else:
                msg = f"Unrecognised shortcode tag '{token.keyword}' "
                msg += f"in line {token.line_number}."
                raise ShortcodeSyntaxError(msg)

        if expecting:
            token = stack[-1].token
            msg = f"Unexpected end of document. The shortcode parser was "
            msg += f"expecting a closing '{expecting[-1]}' tag to close the "
            msg += f"'{token.keyword}' tag opened in line {token.line_number}."
            raise ShortcodeSyntaxError(msg)

        return stack.pop().render(context)


# ------- #
#  Lexer  #
# ------- #


class Token:

    def __init__(self, token_type, token_text, raw_text, line_number):
        words = token_text.split()
        self.keyword = words[0] if words else ''
        self.type = token_type
        self.text = token_text
        self.raw_text = raw_text
        self.line_number = line_number

    def __str__(self):
        return f"({self.type}, {repr(self.text)}, {self.line_number})"


class Lexer:

    def __init__(self, text, start, end, esc_start):
        self.text = text
        self.start = start
        self.end = end
        self.esc_start = esc_start
        self.tokens = []
        self.index = 0
        self.line_number = 1

    def match(self, target):
        if self.text.startswith(target, self.index):
            return True
        return False

    def advance(self):
        if self.text[self.index] == '\n':
            self.line_number += 1
        self.index += 1

    def tokenize(self):
        while self.index < len(self.text):
            if self.match(self.esc_start):
                self.read_escaped_tag_delimiter()
            elif self.match(self.start):
                self.read_tag()
            else:
                self.read_text()
        return self.tokens

    def read_escaped_tag_delimiter(self):
        self.index += len(self.esc_start)
        self.tokens.append(Token("TEXT", self.start, self.esc_start, self.line_number))

    def read_tag(self):
        self.index += len(self.start)
        start_index = self.index
        start_line_number = self.line_number
        while self.index < len(self.text):
            if self.match(self.end):
                text = self.text[start_index:self.index].strip()
                raw_text = self.text[start_index-len(self.start):self.index+len(self.end)]
                self.tokens.append(Token("TAG", text, raw_text, start_line_number))
                self.index += len(self.end)
                return
            self.advance()
        msg = f"Unclosed shortcode tag. The tag was opened in line {start_line_number}."
        raise ShortcodeSyntaxError(msg)

    def read_text(self):
        start_index = self.index
        start_line_number = self.line_number
        while self.index < len(self.text):
            if self.match(self.esc_start) or self.match(self.start):
                break
            self.advance()
        text = self.text[start_index:self.index]
        self.tokens.append(Token("TEXT", text, text, start_line_number))
