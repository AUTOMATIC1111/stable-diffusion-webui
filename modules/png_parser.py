import re

class PngParser:
    re_top_level = None
    re_extra_newline = None
    re_parameters = None

    def __init__(self, pnginfo_string):
        PngParser.init_re()

        self.valid = self.parse_pnginfo(pnginfo_string)

    def parse_pnginfo(self, pnginfo_string):
        try:
            # separate positive, negative, and parameters
            m = PngParser.re_top_level.search(pnginfo_string)
            if m is None:
                return False

            self.positive = m.group(1)
            self.negative = m.group(2)
            self.parameters = m.group(3)
            self.extra = None
            self.settings = None

            # parse extra parameters (if they exist) by a newline outside of quotes
            m = PngParser.re_extra_newline.search(self.parameters)
            if m is not None:
                s = m.span()
                self.extra = self.parameters[s[1]:]
                self.parameters = self.parameters[:s[0]]

            # parse standard parameters
            self.settings = PngParser.re_parameters.findall(self.parameters)
            if self.settings is None:
                return False
        except Exception:
            return False

        return True

    @classmethod
    def init_re(cls):
        if cls.re_top_level is None:
            cls.re_top_level = re.compile(r'^(?P<positive>(?:.|\n)*)\nNegative prompt: (?P<negative>(?:.|\n)*)\n(?=Steps: )(?P<parameters>(?:.|\n)*)$')
            cls.re_extra_newline = re.compile(r'\n(?=(?:[^"]*"[^"]*")*[^"]*$)')
            cls.re_parameters = re.compile(r'\s*(?P<param>[^:,]+):\s*(?P<quote>")?(?P<value>(?(2)(?:.)*?(?:(?<!\\)")|.*?))(?:\s*,\s*|$)')
