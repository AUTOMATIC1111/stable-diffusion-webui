"""Contains classes and methods related to interpretation for components in Gradio."""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

from gradio import components, utils

if TYPE_CHECKING:  # Only import for type checking (is False at runtime).
    from gradio import Interface


class Interpretable(ABC):
    def __init__(self) -> None:
        self.set_interpret_parameters()

    def set_interpret_parameters(self):
        """
        Set any parameters for interpretation. Properties can be set here to be
        used in get_interpretation_neighbors and get_interpretation_scores.
        """
        pass

    def get_interpretation_scores(
        self, x: Any, neighbors: List[Any] | None, scores: List[float], **kwargs
    ) -> List:
        """
        Arrange the output values from the neighbors into interpretation scores for the interface to render.
        Parameters:
            x: Input to interface
            neighbors: Neighboring values to input x used for interpretation.
            scores: Output value corresponding to each neighbor in neighbors
        Returns:
            Arrangement of interpretation scores for interfaces to render.
        """
        return scores


class TokenInterpretable(Interpretable, ABC):
    @abstractmethod
    def tokenize(self, x: Any) -> Tuple[List, List, None]:
        """
        Interprets an input data point x by splitting it into a list of tokens (e.g
        a string into words or an image into super-pixels).
        """
        return [], [], None

    @abstractmethod
    def get_masked_inputs(self, tokens: List, binary_mask_matrix: List[List]) -> List:
        return []


class NeighborInterpretable(Interpretable, ABC):
    @abstractmethod
    def get_interpretation_neighbors(self, x: Any) -> Tuple[List, Dict]:
        """
        Generates values similar to input to be used to interpret the significance of the input in the final output.
        Parameters:
            x: Input to interface
        Returns: (neighbor_values, interpret_kwargs, interpret_by_removal)
            neighbor_values: Neighboring values to input x to compute for interpretation
            interpret_kwargs: Keyword arguments to be passed to get_interpretation_scores
        """
        return [], {}


async def run_interpret(interface: Interface, raw_input: List):
    """
    Runs the interpretation command for the machine learning model. Handles both the "default" out-of-the-box
    interpretation for a certain set of UI component types, as well as the custom interpretation case.
    Parameters:
    raw_input: a list of raw inputs to apply the interpretation(s) on.
    """
    if isinstance(interface.interpretation, list):  # Either "default" or "shap"
        processed_input = [
            input_component.preprocess(raw_input[i])
            for i, input_component in enumerate(interface.input_components)
        ]
        original_output = await interface.call_function(0, processed_input)
        original_output = original_output["prediction"]

        if len(interface.output_components) == 1:
            original_output = [original_output]

        scores, alternative_outputs = [], []

        for i, (x, interp) in enumerate(zip(raw_input, interface.interpretation)):
            if interp == "default":
                input_component = interface.input_components[i]
                neighbor_raw_input = list(raw_input)
                if isinstance(input_component, TokenInterpretable):
                    tokens, neighbor_values, masks = input_component.tokenize(x)
                    interface_scores = []
                    alternative_output = []
                    for neighbor_input in neighbor_values:
                        neighbor_raw_input[i] = neighbor_input
                        processed_neighbor_input = [
                            input_component.preprocess(neighbor_raw_input[i])
                            for i, input_component in enumerate(
                                interface.input_components
                            )
                        ]

                        neighbor_output = await interface.call_function(
                            0, processed_neighbor_input
                        )
                        neighbor_output = neighbor_output["prediction"]
                        if len(interface.output_components) == 1:
                            neighbor_output = [neighbor_output]
                        processed_neighbor_output = [
                            output_component.postprocess(neighbor_output[i])
                            for i, output_component in enumerate(
                                interface.output_components
                            )
                        ]

                        alternative_output.append(processed_neighbor_output)
                        interface_scores.append(
                            quantify_difference_in_label(
                                interface, original_output, neighbor_output
                            )
                        )
                    alternative_outputs.append(alternative_output)
                    scores.append(
                        input_component.get_interpretation_scores(
                            raw_input[i],
                            neighbor_values,
                            interface_scores,
                            masks=masks,
                            tokens=tokens,
                        )
                    )
                elif isinstance(input_component, NeighborInterpretable):
                    (
                        neighbor_values,
                        interpret_kwargs,
                    ) = input_component.get_interpretation_neighbors(x)
                    interface_scores = []
                    alternative_output = []
                    for neighbor_input in neighbor_values:
                        neighbor_raw_input[i] = neighbor_input
                        processed_neighbor_input = [
                            input_component.preprocess(neighbor_raw_input[i])
                            for i, input_component in enumerate(
                                interface.input_components
                            )
                        ]
                        neighbor_output = await interface.call_function(
                            0, processed_neighbor_input
                        )
                        neighbor_output = neighbor_output["prediction"]
                        if len(interface.output_components) == 1:
                            neighbor_output = [neighbor_output]
                        processed_neighbor_output = [
                            output_component.postprocess(neighbor_output[i])
                            for i, output_component in enumerate(
                                interface.output_components
                            )
                        ]

                        alternative_output.append(processed_neighbor_output)
                        interface_scores.append(
                            quantify_difference_in_label(
                                interface, original_output, neighbor_output
                            )
                        )
                    alternative_outputs.append(alternative_output)
                    interface_scores = [-score for score in interface_scores]
                    scores.append(
                        input_component.get_interpretation_scores(
                            raw_input[i],
                            neighbor_values,
                            interface_scores,
                            **interpret_kwargs,
                        )
                    )
                else:
                    raise ValueError(
                        f"Component {input_component} does not support interpretation"
                    )
            elif interp == "shap" or interp == "shapley":
                try:
                    import shap  # type: ignore
                except (ImportError, ModuleNotFoundError):
                    raise ValueError(
                        "The package `shap` is required for this interpretation method. Try: `pip install shap`"
                    )
                input_component = interface.input_components[i]
                if not isinstance(input_component, TokenInterpretable):
                    raise ValueError(
                        "Input component {} does not support `shap` interpretation".format(
                            input_component
                        )
                    )

                tokens, _, masks = input_component.tokenize(x)

                # construct a masked version of the input
                def get_masked_prediction(binary_mask):
                    assert isinstance(input_component, TokenInterpretable)
                    masked_xs = input_component.get_masked_inputs(tokens, binary_mask)
                    preds = []
                    for masked_x in masked_xs:
                        processed_masked_input = copy.deepcopy(processed_input)
                        processed_masked_input[i] = input_component.preprocess(masked_x)
                        new_output = utils.synchronize_async(
                            interface.call_function, 0, processed_masked_input
                        )
                        new_output = new_output["prediction"]
                        if len(interface.output_components) == 1:
                            new_output = [new_output]
                        pred = get_regression_or_classification_value(
                            interface, original_output, new_output
                        )
                        preds.append(pred)
                    return np.array(preds)

                num_total_segments = len(tokens)
                explainer = shap.KernelExplainer(
                    get_masked_prediction, np.zeros((1, num_total_segments))
                )
                shap_values = explainer.shap_values(
                    np.ones((1, num_total_segments)),
                    nsamples=int(interface.num_shap * num_total_segments),
                    silent=True,
                )
                assert shap_values is not None, "SHAP values could not be calculated"
                scores.append(
                    input_component.get_interpretation_scores(
                        raw_input[i],
                        None,
                        shap_values[0].tolist(),
                        masks=masks,
                        tokens=tokens,
                    )
                )
                alternative_outputs.append([])
            elif interp is None:
                scores.append(None)
                alternative_outputs.append([])
            else:
                raise ValueError("Unknown intepretation method: {}".format(interp))
        return scores, alternative_outputs
    elif interface.interpretation:  # custom interpretation function
        processed_input = [
            input_component.preprocess(raw_input[i])
            for i, input_component in enumerate(interface.input_components)
        ]
        interpreter = interface.interpretation
        interpretation = interpreter(*processed_input)
        if len(raw_input) == 1:
            interpretation = [interpretation]
        return interpretation, []
    else:
        raise ValueError("No interpretation method specified.")


def diff(original: Any, perturbed: Any) -> int | float:
    try:  # try computing numerical difference
        score = float(original) - float(perturbed)
    except ValueError:  # otherwise, look at strict difference in label
        score = int(not (original == perturbed))
    return score


def quantify_difference_in_label(
    interface: Interface, original_output: List, perturbed_output: List
) -> int | float:
    output_component = interface.output_components[0]
    post_original_output = output_component.postprocess(original_output[0])
    post_perturbed_output = output_component.postprocess(perturbed_output[0])

    if isinstance(output_component, components.Label):
        original_label = post_original_output["label"]
        perturbed_label = post_perturbed_output["label"]

        # Handle different return types of Label interface
        if "confidences" in post_original_output:
            original_confidence = original_output[0][original_label]
            perturbed_confidence = perturbed_output[0][original_label]
            score = original_confidence - perturbed_confidence
        else:
            score = diff(original_label, perturbed_label)
        return score

    elif isinstance(output_component, components.Number):
        score = diff(post_original_output, post_perturbed_output)
        return score

    else:
        raise ValueError(
            "This interpretation method doesn't support the Output component: {}".format(
                output_component
            )
        )


def get_regression_or_classification_value(
    interface: Interface, original_output: List, perturbed_output: List
) -> int | float:
    """Used to combine regression/classification for Shap interpretation method."""
    output_component = interface.output_components[0]
    post_original_output = output_component.postprocess(original_output[0])
    post_perturbed_output = output_component.postprocess(perturbed_output[0])

    if isinstance(output_component, components.Label):
        original_label = post_original_output["label"]
        perturbed_label = post_perturbed_output["label"]

        # Handle different return types of Label interface
        if "confidences" in post_original_output:
            if math.isnan(perturbed_output[0][original_label]):
                return 0
            return perturbed_output[0][original_label]
        else:
            score = diff(
                perturbed_label, original_label
            )  # Intentionally inverted order of arguments.
        return score

    else:
        raise ValueError(
            "This interpretation method doesn't support the Output component: {}".format(
                output_component
            )
        )
