# https://github.com/openstack/rally/blob/master/rally/common/plugin/info.py
# Copyright 2015: Mirantis Inc.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import re
import sys

__all__ = ['parse_docstring']


FIELDS = 'param|val' # supported fields
PARAM_OR_RETURN_REGEX = re.compile(f":(?:{FIELDS}|return)")
RETURN_REGEX = re.compile(":return: (?P<doc>.*)", re.S)
NEW_REGEX = re.compile(f":(?P<field>{FIELDS}) (?P<name>[\*\w]+): (?P<doc>.*?)"
                         f"(?:(?=:(?:{FIELDS}|return|raises))|\Z)", re.S)

def trim(docstring):
    """trim function from PEP-257"""
    if not docstring:
        return ""
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)

    # Current code/unittests expects a line return at
    # end of multiline docstrings
    # workaround expected behavior from unittests
    if "\n" in docstring:
        trimmed.append("")

    # Return a single string:
    return "\n".join(trimmed)


def reindent(string):
    return "\n".join(l.strip() for l in string.strip().split("\n"))


def parse_docstring(docstring):
    """Parse the docstring into its components.

    :return: a dictionary of form
              {
                  "short_description": ...,
                  "long_description": ...,
                  "params": [{"name": ..., "doc": ...}, ...],
                  "vals": [{"name": ..., "doc": ...}, ...],
                  "return": ...
              }
    """

    short_description = long_description = return_str = ""
    args = []

    if docstring:
        docstring = trim(docstring.lstrip("\n"))

        lines = docstring.split("\n", 1)
        short_description = lines[0]

        if len(lines) > 1:
            long_description = lines[1].strip()

            params_return_desc = None

            match = PARAM_OR_RETURN_REGEX.search(long_description)
            if match:
                long_desc_end = match.start()
                params_return_desc = long_description[long_desc_end:].strip()
                long_description = long_description[:long_desc_end].rstrip()

            if params_return_desc:
                args = [
                    {"name": name, "doc": trim(doc), "field": field}
                    for field, name, doc in NEW_REGEX.findall(params_return_desc)
                ]
                match = RETURN_REGEX.search(params_return_desc)
                if match:
                    return_str = reindent(match.group("doc"))
    comments = {p['name']: p['doc'] for p in args}
    return {
        "short_description": short_description,
        "long_description": long_description,
        "args": args,
        "comments": comments,
        "return": return_str
    }


class InfoMixin(object):

    @classmethod
    def _get_doc(cls):
        """Return documentary of class

        By default it returns docstring of class, but it can be overridden
        for example for cases like merging own docstring with parent
        """
        return cls.__doc__

    @classmethod
    def get_info(cls):
        doc = parse_docstring(cls._get_doc())

        return {
            "name": cls.get_name(),
            "platform": cls.get_platform(),
            "module": cls.__module__,
            "title": doc["short_description"],
            "description": doc["long_description"],
            "parameters": doc["params"],
            "schema": getattr(cls, "CONFIG_SCHEMA", None),
            "return": doc["return"]
        }
