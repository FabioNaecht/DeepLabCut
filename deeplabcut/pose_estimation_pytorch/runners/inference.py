#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Generic, Iterable

import numpy as np
import torch
import torch.nn as nn
from time import perf_counter_ns

from deeplabcut.pose_estimation_pytorch.data.postprocessor import Postprocessor
from deeplabcut.pose_estimation_pytorch.data.preprocessor import Preprocessor
from deeplabcut.pose_estimation_pytorch.models.detectors import BaseDetector
from deeplabcut.pose_estimation_pytorch.models.model import PoseModel
from deeplabcut.pose_estimation_pytorch.runners.base import ModelType, Runner
from deeplabcut.pose_estimation_pytorch.task import Task


class InferenceRunner(Runner, Generic[ModelType], metaclass=ABCMeta):
    """Base class for inference runners

    A runner takes a model and runs actions on it, such as training or inference
    """

    def __init__(
        self,
        model: ModelType,
        batch_size: int = 1,
        device: str = "cpu",
        snapshot_path: str | Path | None = None,
        preprocessor: Preprocessor | None = None,
        postprocessor: Postprocessor | None = None,
    ):
        """
        Args:
            model: the model to run actions on
            device: the device to use (e.g. {'cpu', 'cuda:0', 'mps'})
            snapshot_path: if defined, the path of a snapshot from which to load pretrained weights
            preprocessor: the preprocessor to use on images before inference
            postprocessor: the postprocessor to use on images after inference
        """
        super().__init__(model=model, device=device, snapshot_path=snapshot_path)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer; is {batch_size}")

        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        if self.snapshot_path is not None and self.snapshot_path != "":
            self.load_snapshot(self.snapshot_path, self.device, self.model)

        self._batch: torch.Tensor | None = None
        self._contexts: list[dict] = []
        self._image_batch_sizes: list[int] = []
        self._predictions: list = []

    @abstractmethod
    def predict(self, inputs: torch.Tensor) -> list[dict[str, dict[str, np.ndarray]]]:
        """Makes predictions from a model input and output

        Args:
            the inputs to the model, of shape (batch_size, ...)

        Returns:
            the predictions for each of the 'batch_size' inputs
        """

    @torch.no_grad()
    def inference(
        self,
        images: Iterable[str | np.ndarray]
        | Iterable[tuple[str | np.ndarray, dict[str, Any]]],
    ) -> list[dict[str, np.ndarray]]:
        """Run model inference on the given dataset

        TODO: Add an option to also return head outputs (such as heatmaps)? Can be
         super useful for debugging

        Args:
            images: the images to run inference on, optionally with context

        Returns:
            a dict containing head predictions for each image
            [
                {
                    "bodypart": {"poses": np.array},
                    "unique_bodypart": {"poses": np.array},
                }
            ]
        """
        self.model.to(self.device)
        self.model.eval()

        results = []
        for data in images:
            self._prepare_inputs(data)
            self._process_full_batches()
            results += self._extract_results()

        # Process the last batch even if not full
        if self._inputs_waiting_for_processing():
            self._process_batch()
            results += self._extract_results()

        return results

    @torch.no_grad()
    def inference_shrunk(self, images: Iterable[str | np.ndarray] | Iterable[tuple[str | np.ndarray, dict[str, Any]]]) -> list[dict[str, np.ndarray]]:
        """ model inference - deleted all unnecessary code for testing!! """
        self.model.to(self.device)
        self.model.eval()

        results = []
        for data in images:
            self._prepare_inputs(data)
            self._process_full_batches()
            results += self._extract_results()

        return results

    @torch.no_grad()
    def inference_single_image2(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """ Run inference on a single image """
        # Move model to device and set it to evaluation mode
        # self.device = "mps"
        self.model.to(self.device)
        self.model.eval()
        print('\n')

        # Convert the input image to a tensor and cast it to float
        start_timer_process = perf_counter_ns() / 1_000_000
        input_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Convert to [1, 3, height, width] => torch.Size([1, 3, 200, 200])
        input_tensor = input_tensor.to(self.device)
        end_timer_process = perf_counter_ns() / 1_000_000
        print(f"Time to process image: {end_timer_process - start_timer_process} ms")

        # Run the model forward pass
        start_timer_inference = perf_counter_ns() / 1_000_000
        outputs = self.model.forward(input_tensor) # forward method not necessary...
        end_timer_inference = perf_counter_ns() / 1_000_000
        print(f"Time to run inference: {end_timer_inference - start_timer_inference} ms")

        # Get predictions from the model outputs
        start_timer_get_predictions = perf_counter_ns() / 1_000_000
        predictions = self.model.get_predictions(outputs)
        end_timer_get_predictions = perf_counter_ns() / 1_000_000
        print(f"Time to get predictions: {end_timer_get_predictions - start_timer_get_predictions} ms")
        print('\n')

        return predictions

    @torch.no_grad()
    def inference_single_image(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """ Run inference on a single image """
        # Move model to device and set it to evaluation mode
        self.model.to(self.device)
        self.model.eval()

        # Convert the input image to a tensor and cast it to float
        input_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Convert to [1, 3, height, width] => torch.Size([1, 3, 200, 200])
        print(f"input_tensor shape: {input_tensor.shape}")  # Inspect shape for further debugging
        input_tensor = input_tensor.to(self.device)

        # Run the model forward pass
        outputs = self.model(input_tensor)
        print(f"output type: {type(outputs)}")  # Inspect type for further debugging
        print(f"outputs: {outputs}")  # Inspect outputs for further debugging
        print(f"tensorshape of outputs heatmap: {outputs['bodypart']['heatmap'].shape}")  # torch.Size([1, 1, 27, 27])
        print(f"tensorshape of outputs locref: {outputs['bodypart']['locref'].shape}")  # torch.Size([1, 2, 27, 27])

        import matplotlib.pyplot as plt
        # import torch
        heatmap = outputs['bodypart']['heatmap'][0, 0]  # Selects the first heatmap (adjust as needed)
        locref = outputs['bodypart']['locref'][0]  # Selects the first locref (adjust as needed)
        print(f"Max value in heatmap: {heatmap.max().item()}")
        print(f"Min value in heatmap: {heatmap.min().item()}")
        print(f"Mean value in heatmap: {heatmap.mean().item()}")
        # coordinate of max value
        max_value = heatmap.max().item()
        max_index = heatmap.argmax()
        max_index = torch.unravel_index(max_index, heatmap.shape)
        print(f"Max value in heatmap: {max_value} at index: {max_index}")

        plt.imshow(heatmap.cpu().numpy(), cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title("Heatmap for Bodypart")
        plt.show()

        import cv2
        import numpy as np

        # Assuming `image` is your original RGB image (200, 200, 3)
        # and `heatmap` is the torch tensor heatmap (1, 1, 27, 27)

        # Step 1: Convert heatmap tensor to numpy and resize to match image shape
        heatmap_np = heatmap.squeeze().cpu().numpy()  # Convert to 2D numpy array (27, 27)

        # Resize heatmap to match image dimensions (200, 200)
        heatmap_resized = cv2.resize(heatmap_np, (image.shape[1], image.shape[0]))

        # Step 2: Normalize heatmap to range [0, 255] for visualization
        heatmap_resized = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Step 3: Convert single-channel heatmap to 3 channels to match the image shape
        heatmap_color = cv2.applyColorMap(heatmap_resized,
                                          cv2.COLORMAP_JET)  # Apply a color map for better visualization

        # Step 4: Overlay the heatmap onto the image
        overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)

        # Display or save the overlay image
        cv2.imshow("Overlay", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Find the index of the maximum value along each dimension separately
        max_index = torch.argmax(heatmap.view(-1))  # Flatten the heatmap and get the index of the max value
        max_y, max_x = divmod(max_index.item(), heatmap.shape[-1])  # Convert flat index to 2D (y, x) coordinates

        # Step 2: Retrieve x and y offsets from locref at coarse location
        x_refinement = locref[0, 0, max_y, max_x].item()  # x offset
        y_refinement = locref[0, 1, max_y, max_x].item()  # y offset

        # Step 3: Apply refinement to improve coordinate prediction
        # Convert the coarse (max_x, max_y) location to match original image scale, then add refinements.
        coarse_x = max_x * (200 / 27)  # Scale x position to original image
        coarse_y = max_y * (200 / 27)  # Scale y position to original image

        # Final refined coordinates
        final_x = coarse_x + x_refinement
        final_y = coarse_y + y_refinement

        print(f"Refined landmark coordinates: (x={final_x}, y={final_y})")

        # for key in outputs.keys():
        #     print(f"key: {key}")
        #     print(f"outputs[key]: {outputs[key]}")
        #     # print(f"outputs[key].shape: {outputs[key].shape}")
        #
        # # Post-process outputs if needed (optional)
        # print(f"used postprocessor: {self.postprocessor}")  # Inspect postprocessor for further debugging
        # if self.postprocessor is not None:
        #     outputs, _ = self.postprocessor(outputs, {})
        #
        # print(f"outputs after postprocessing: {outputs}")  # Inspect outputs for further debugging
        # # Convert outputs to numpy format if they need to be returned as np.ndarray
        # predictions = {key: output.cpu().numpy() for key, output in outputs.items()}

        return predictions

    @torch.no_grad()
    def inference_single_image1(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Run inference on a single image."""
        # Move model to device and set it to evaluation mode
        self.model.to(self.device)
        self.model.eval()

        # Convert the input image to a tensor and format it
        input_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Run the model forward pass
        outputs = self.model(input_tensor)
        print("Shape of outputs:", outputs.shape)  # Inspect shape for further debugging

        # Construct predictions dictionary to match postprocessor's expected format
        self._predictions = [
            {
                "bodypart": {
                    "x": outputs[0, i, 0].item(),  # Assuming outputs is [1, num_parts, 2]
                    "y": outputs[0, i, 1].item()
                }
            } for i in range(outputs.shape[1])  # Iterating over parts
        ]
        self._image_batch_sizes = [1]
        self._contexts = [{}]  # Placeholder for context

        # Post-process predictions
        results = []
        if self.postprocessor is not None:
            while (
                    len(self._image_batch_sizes) > 0
                    and len(self._predictions) >= self._image_batch_sizes[0]
            ):
                num_predictions = self._image_batch_sizes[0]
                image_predictions = self._predictions[:num_predictions]
                context = self._contexts[0]

                # Pass through postprocessor
                image_predictions, _ = self.postprocessor(image_predictions, context)

                # Clean up as in _extract_results
                self._contexts = self._contexts[1:]
                self._image_batch_sizes = self._image_batch_sizes[1:]
                self._predictions = self._predictions[num_predictions:]
                results.append(image_predictions)

        return results[0] if results else {}

    def _prepare_inputs(
        self, data: str | np.ndarray | tuple[str | np.ndarray, dict],
    ) -> None:
        """
        Prepares inputs for an image and adds them to the data ready to be processed
        """
        if isinstance(data, (str, np.ndarray)):
            inputs, context = data, {}
        else:
            inputs, context = data

        if self.preprocessor is not None:
            inputs, context = self.preprocessor(inputs, context)
        else:
            inputs = torch.as_tensor(inputs)

        print(f"inputs shape (prepare inputs): {inputs.shape}") # torch.Size([1, 3, 200, 200])

        self._contexts.append(context)
        self._image_batch_sizes.append(len(inputs))

        # skip when there are no inputs for an image
        if len(inputs) == 0:
            return

        if self._batch is None:
            self._batch = inputs
        else:
            self._batch = torch.cat([self._batch, inputs], dim=0)
            print(f"batch shape: {self._batch.shape}")  # batch shape: torch.Size([k, 3, 200, 200])

    def _process_full_batches(self) -> None:
        """Processes prepared inputs in batches of the desired batch size."""
        print(f" batch size: {self.batch_size}")
        while self._batch is not None and len(self._batch) >= self.batch_size:
            self._process_batch()

    def _extract_results(self) -> list:
        """Obtains results that were obtained from processing a batch."""
        results = []
        while (
            len(self._image_batch_sizes) > 0
            and len(self._predictions) >= self._image_batch_sizes[0]
        ):
            num_predictions = self._image_batch_sizes[0]
            image_predictions = self._predictions[:num_predictions]
            print(f"image_predictions0 (_extract_results): {image_predictions}")
            context = self._contexts[0]

            if self.postprocessor is not None:
                # TODO: Should we return context?
                # TODO: typing update - the post-processor can remove a dict level
                image_predictions, _ = self.postprocessor(image_predictions, context)
                print(f"image_predictions1 (_extract_results): {image_predictions}")

            self._contexts = self._contexts[1:]
            self._image_batch_sizes = self._image_batch_sizes[1:]
            self._predictions = self._predictions[num_predictions:]
            results.append(image_predictions)

        return results

    def _process_batch(self) -> None:
        """
        Processes a batch. There must be inputs waiting to be processed before this is
        called, otherwise this method will raise an error.
        """
        batch = self._batch[:self.batch_size]
        print(f"batch shape (_process batch): {batch.shape}")
        self._predictions += self.predict(batch)
        print(f"predictions (_process batch): {self._predictions}")

        # remove processed inputs from batch
        if len(self._batch) <= self.batch_size:
            self._batch = None
        else:
            self._batch = self._batch[self.batch_size:]

    def _inputs_waiting_for_processing(self) -> bool:
        """Returns: Whether there are inputs which have not yet been processed"""
        return self._batch is not None and len(self._batch) > 0


class PoseInferenceRunner(InferenceRunner[PoseModel]):
    """Runner for pose estimation inference"""

    def __init__(self, model: PoseModel, **kwargs):
        super().__init__(model, **kwargs)

    def predict(self, inputs: torch.Tensor) -> list[dict[str, dict[str, np.ndarray]]]:
        """Makes predictions from a model input and output

        Args:
            the inputs to the model, of shape (batch_size, ...)

        Returns:
            predictions for each of the 'batch_size' inputs, made by each head, e.g.
            [
                {
                    "bodypart": {"poses": np.ndarray},
                    "unique_bodypart": {"poses": np.ndarray},
                }
            ]
        """
        print("PoseInferenceRunner(InferenceRunner[PoseModel])")
        outputs = self.model(inputs.to(self.device))
        raw_predictions = self.model.get_predictions(outputs)
        predictions = [
            {
                head: {
                    pred_name: pred[b].cpu().numpy()
                    for pred_name, pred in head_outputs.items()
                }
                for head, head_outputs in raw_predictions.items()
            }
            for b in range(len(inputs))
        ]
        return predictions


class DetectorInferenceRunner(InferenceRunner[BaseDetector]):
    """Runner for object detection inference"""

    def __init__(self, model: BaseDetector, **kwargs):
        """
        Args:
            model: The detector to use for inference.
            **kwargs: Inference runner kwargs.
        """
        super().__init__(model, **kwargs)

    def predict(self, inputs: torch.Tensor) -> list[dict[str, dict[str, np.ndarray]]]:
        """Makes predictions from a model input and output

        Args:
            the inputs to the model, of shape (batch_size, ...)

        Returns:
            predictions for each of the 'batch_size' inputs, made by each head, e.g.
            [
                {
                    "bodypart": {"poses": np.ndarray},
                    "unique_bodypart": "poses": np.ndarray},
            ]
        """
        _, raw_predictions = self.model(inputs.to(self.device))
        predictions = [
            {
                "detection": {
                    "bboxes": item["boxes"]
                    .cpu()
                    .numpy()
                    .reshape(-1, 4),
                    "scores": item["scores"]
                    .cpu()
                    .numpy()
                    .reshape(-1),
                }
            }
            for item in raw_predictions
        ]
        return predictions


def build_inference_runner(
    task: Task,
    model: nn.Module,
    device: str,
    snapshot_path: str | Path,
    batch_size: int = 1,
    preprocessor: Preprocessor | None = None,
    postprocessor: Postprocessor | None = None,
) -> InferenceRunner:
    """
    Build a runner object according to a pytorch configuration file

    Args:
        task: the inference task to run
        model: the model to run
        device: the device to use (e.g. {'cpu', 'cuda:0', 'mps'})
        snapshot_path: the snapshot from which to load the weights
        batch_size: the batch size to use to run inference
        preprocessor: the preprocessor to use on images before inference
        postprocessor: the postprocessor to use on images after inference

    Returns:
        the inference runner
    """
    print("build_inference_runner")
    kwargs = dict(
        model=model,
        device=device,
        snapshot_path=snapshot_path,
        batch_size=batch_size,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
    if task == Task.DETECT:
        return DetectorInferenceRunner(**kwargs)

    return PoseInferenceRunner(**kwargs)
