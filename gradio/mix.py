"""
Ways to transform interfaces to produce new interfaces
"""
import asyncio
import warnings

from gradio_client.documentation import document, set_documentation_group

import gradio

set_documentation_group("mix_interface")


@document()
class Parallel(gradio.Interface):
    """
    Creates a new Interface consisting of multiple Interfaces in parallel (comparing their outputs).
    The Interfaces to put in Parallel must share the same input components (but can have different output components).

    Demos: interface_parallel, interface_parallel_load
    Guides: advanced-interface-features
    """

    def __init__(self, *interfaces: gradio.Interface, **options):
        """
        Parameters:
            interfaces: any number of Interface objects that are to be compared in parallel
            options: additional kwargs that are passed into the new Interface object to customize it
        Returns:
            an Interface object comparing the given models
        """
        outputs = []

        for interface in interfaces:
            if not (isinstance(interface, gradio.Interface)):
                warnings.warn(
                    "Parallel requires all inputs to be of type Interface. "
                    "May not work as expected."
                )
            outputs.extend(interface.output_components)

        async def parallel_fn(*args):
            return_values_with_durations = await asyncio.gather(
                *[interface.call_function(0, list(args)) for interface in interfaces]
            )
            return_values = [rv["prediction"] for rv in return_values_with_durations]
            combined_list = []
            for interface, return_value in zip(interfaces, return_values):
                if len(interface.output_components) == 1:
                    combined_list.append(return_value)
                else:
                    combined_list.extend(return_value)
            if len(outputs) == 1:
                return combined_list[0]
            return combined_list

        parallel_fn.__name__ = " | ".join([io.__name__ for io in interfaces])

        kwargs = {
            "fn": parallel_fn,
            "inputs": interfaces[0].input_components,
            "outputs": outputs,
        }
        kwargs.update(options)
        super().__init__(**kwargs)


@document()
class Series(gradio.Interface):
    """
    Creates a new Interface from multiple Interfaces in series (the output of one is fed as the input to the next,
    and so the input and output components must agree between the interfaces).

    Demos: interface_series, interface_series_load
    Guides: advanced-interface-features
    """

    def __init__(self, *interfaces: gradio.Interface, **options):
        """
        Parameters:
            interfaces: any number of Interface objects that are to be connected in series
            options: additional kwargs that are passed into the new Interface object to customize it
        Returns:
            an Interface object connecting the given models
        """

        async def connected_fn(*data):
            for idx, interface in enumerate(interfaces):
                # skip preprocessing for first interface since the Series interface will include it
                if idx > 0 and not (interface.api_mode):
                    data = [
                        input_component.preprocess(data[i])
                        for i, input_component in enumerate(interface.input_components)
                    ]

                # run all of predictions sequentially
                data = (await interface.call_function(0, list(data)))["prediction"]
                if len(interface.output_components) == 1:
                    data = [data]

                # skip postprocessing for final interface since the Series interface will include it
                if idx < len(interfaces) - 1 and not (interface.api_mode):
                    data = [
                        output_component.postprocess(data[i])
                        for i, output_component in enumerate(
                            interface.output_components
                        )
                    ]

            if len(interface.output_components) == 1:  # type: ignore
                return data[0]
            return data

        for interface in interfaces:
            if not (isinstance(interface, gradio.Interface)):
                warnings.warn(
                    "Series requires all inputs to be of type Interface. May "
                    "not work as expected."
                )
        connected_fn.__name__ = " => ".join([io.__name__ for io in interfaces])

        kwargs = {
            "fn": connected_fn,
            "inputs": interfaces[0].input_components,
            "outputs": interfaces[-1].output_components,
            "_api_mode": interfaces[0].api_mode,  # TODO: set api_mode per-interface
        }
        kwargs.update(options)
        super().__init__(**kwargs)
