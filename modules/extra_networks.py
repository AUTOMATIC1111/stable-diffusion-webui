import re
from collections import defaultdict, OrderedDict
from typing import Dict, List

from modules import errors
from modules.processing import StableDiffusionProcessing

extra_network_registry = {}


def initialize():
    extra_network_registry.clear()


def register_extra_network(extra_network):
    extra_network_registry[extra_network.name] = extra_network


class ExtraNetworkParams:
    def __init__(self, items: List[str] | None = None):
        self.items = items or []


class ExtraNetwork:
    def __init__(self, name: str):
        self.name = name

    def activate(self, p: StableDiffusionProcessing, params_list: List[ExtraNetworkParams]):
        """
        Called by processing on every run. Whatever the extra network is meant to do should be activated here.
        Passes arguments related to this extra network in params_list.
        User passes arguments by specifying this in his prompt:

        <name:arg1:arg2:arg3>

        Where name matches the name of this ExtraNetwork object, and arg1:arg2:arg3 are any natural number of text arguments
        separated by colon.

        Even if the user does not mention this ExtraNetwork in his prompt, the call will stil be made, with empty params_list -
        in this case, all effects of this extra networks should be disabled.

        Can be called multiple times before deactivate() - each new call should override the previous call completely.

        For example, if this ExtraNetwork's name is 'hypernet' and user's prompt is:

        > "1girl, <hypernet:agm:1.1> <extrasupernet:master:12:13:14> <hypernet:ray>"

        params_list will be:

        [
            ExtraNetworkParams(items=["agm", "1.1"]),
            ExtraNetworkParams(items=["ray"])
        ]

        """
        raise NotImplementedError

    def deactivate(self, p: StableDiffusionProcessing):
        """
        Called at the end of processing for housekeeping. No need to do anything here.
        """

        raise NotImplementedError

    def infotext_params(self, p: StableDiffusionProcessing, params_list: List[ExtraNetworkParams]):
        """
        Returns a dict of extra infotext parameters, to be added inside the
        "Extra networks" infotext field, for example model hashes
        """
        return {}


def activate(p: StableDiffusionProcessing, extra_network_data: Dict[str, List[ExtraNetworkParams]]):
    """call activate for extra networks in extra_network_data in specified order, then call
    activate for all remaining registered networks with an empty argument list"""

    for extra_network_name, extra_network_args in extra_network_data.items():
        extra_network = extra_network_registry.get(extra_network_name, None)
        if extra_network is None:
            print(f"Skipping unknown extra network: {extra_network_name}")
            continue

        try:
            extra_network.activate(p, extra_network_args)
        except Exception as e:
            errors.display(e, f"activating extra network {extra_network_name} with arguments {extra_network_args}")

    for extra_network_name, extra_network in extra_network_registry.items():
        args = extra_network_data.get(extra_network_name, None)
        if args is not None:
            continue

        try:
            extra_network.activate(p, [])
        except Exception as e:
            errors.display(e, f"activating extra network {extra_network_name}")


def deactivate(p: StableDiffusionProcessing, extra_network_data: Dict[str, List[ExtraNetworkParams]]):
    """call deactivate for extra networks in extra_network_data in specified order, then call
    deactivate for all remaining registered networks"""

    for extra_network_name, extra_network_args in extra_network_data.items():
        extra_network = extra_network_registry.get(extra_network_name, None)
        if extra_network is None:
            continue

        try:
            extra_network.deactivate(p)
        except Exception as e:
            errors.display(e, f"deactivating extra network {extra_network_name}")

    for extra_network_name, extra_network in extra_network_registry.items():
        args = extra_network_data.get(extra_network_name, None)
        if args is not None:
            continue

        try:
            extra_network.deactivate(p)
        except Exception as e:
            errors.display(e, f"deactivating unmentioned extra network {extra_network_name}")


def get_infotext_params(p: StableDiffusionProcessing, extra_network_data: Dict[str, List[ExtraNetworkParams]]):
    """
    returns the infotext for extra networks
    example output:
    Extra networks: "lora|claude-monet: abcd1234, lora|vincent-van-gogh: 4321dcba"
    """
    params = OrderedDict()

    for extra_network_name, extra_network_args in extra_network_data.items():
        extra_network = extra_network_registry.get(extra_network_name, None)
        if extra_network is None:
            continue

        try:
            found = extra_network.infotext_params(p, extra_network_args)
            if found:
                params.update(found)
        except Exception as e:
            errors.display(e, f"getting extra network params {extra_network_name}")

    for extra_network_name, extra_network in extra_network_registry.items():
        args = extra_network_data.get(extra_network_name, None)
        if args is not None:
            continue

        try:
            found = extra_network.infotext_params(p, [])
            if found:
                params.update(found)
        except Exception as e:
            errors.display(e, f"getting unmentioned extra network params {extra_network_name}")

    result = []
    for k, v in params.items():
        result.append(f"{k}: {v}".replace('"', '\\"'))

    return ", ".join(result)


re_extra_net = re.compile(r"<(\w+):([^>]+)>")


def parse_prompt(prompt):
    res = defaultdict(list)

    def found(m):
        name = m.group(1)
        args = m.group(2)

        res[name].append(ExtraNetworkParams(items=args.split(":")))

        return ""

    prompt = re.sub(re_extra_net, found, prompt)

    return prompt, res


def parse_prompts(prompts):
    res = []
    extra_data = None

    for prompt in prompts:
        updated_prompt, parsed_extra_data = parse_prompt(prompt)

        if extra_data is None:
            extra_data = parsed_extra_data

        res.append(updated_prompt)

    return res, extra_data

