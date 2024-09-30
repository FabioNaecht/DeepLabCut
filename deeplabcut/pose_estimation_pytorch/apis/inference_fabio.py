import os
import tqdm
import json
import torch
import logging
import yaml
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
import ruamel.yaml
from deeplabcut.core.engine import Engine
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.pose_estimation_pytorch.runners.snapshots import (
    Snapshot,
    TorchSnapshotManager,
)
import deeplabcut.utils.auxiliaryfunctions as auxiliaryfunctions


def analyze_images(
    config: str | Path,
    images: str | Path | list[str] | list[Path],
    output_dir: str | Path,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    snapshot_index: int | None = None,
    detector_snapshot_index: int | None = None,
    modelprefix: str = "",
    device: str | None = None,
    max_individuals: int | None = None,
    progress_bar: bool = True,
) -> dict[str, dict]:
    """Runs analysis on images using a pose model.

    Args:
        config: The project configuration file.
        images: The image(s) to run inference on. Can be the path to an image, the path
            to a directory containing images, or a list of image paths or directories
            containing images.
        output_dir: The directory where the predictions will be stored.
        shuffle: The shuffle for which to run image analysis.
        trainingsetindex: The trainingsetindex for which to run image analysis.
        snapshot_index: The index of the snapshot to use. Loaded from the project
            configuration file if None.
        detector_snapshot_index: For top-down models only. The index of the detector
            snapshot to use. Loaded from the project configuration file if None.
        modelprefix: The model prefix used for the shuffle.
        device: The device to use to run image analysis.
        max_individuals: The maximum number of individuals to detect in each image. Set
            to the number of individuals in the project if None.
        progress_bar: Whether to display a progress bar when running inference.

    Returns:
        A dictionary mapping each image filename to the different types of predictions
        for it (e.g. "bodyparts", "unique_bodyparts", "bboxes", "bbox_scores")
    """
    if not output_dir.is_dir():
        raise ValueError(
            f"The `output_dir` must be a directory - please change: {output_dir}"
        )

    cfg = read_config(config)
    print({cfg})
    train_frac = cfg["TrainingFraction"][trainingsetindex]
    model_folder = Path(cfg["project_path"]) / get_model_folder(
        train_frac,
        shuffle,
        cfg,
        engine=Engine.PYTORCH,
        modelprefix=modelprefix,
    )
    train_folder = model_folder / "train"

    model_cfg_path = train_folder / Engine.PYTORCH.pose_cfg_name
    model_cfg = read_config_as_dict(model_cfg_path)
    pose_task = Task(model_cfg["method"])

    # get the snapshots to analyze images with
    snapshot_index, detector_snapshot_index = parse_snapshot_index_for_analysis(
        cfg, model_cfg, snapshot_index, detector_snapshot_index
    )
    snapshot = get_model_snapshots(snapshot_index, train_folder, pose_task)[0]
    detector_snapshot = None
    if detector_snapshot_index is not None:
        detector_snapshot = get_model_snapshots(
            detector_snapshot_index, train_folder, Task.DETECT
        )[0]

    predictions = analyze_image_folder(
        model_cfg=model_cfg,
        images=images,
        snapshot_path=snapshot.path,
        detector_path=None if detector_snapshot is None else detector_snapshot.path,
        device=device,
        max_individuals=max_individuals,
        progress_bar=progress_bar,
    )

    if len(predictions) == 0:
        logging.info(f"Found no images in {images}")
        return {}

    # FIXME(niels): store as H5
    pred_json = {}
    for image, pred in predictions.items():
        pred_json[image] = dict(bodyparts=pred["bodyparts"].tolist())
        for k in ("unique_bodyparts", "bboxes", "bbox_scores"):
            if k in pred:
                pred_json[image][k] = pred[k].tolist()

    scorer = get_scorer_name(
        cfg,
        shuffle=shuffle,
        train_fraction=train_frac,
        snapshot_uid=get_scorer_uid(snapshot, detector_snapshot),
        modelprefix=modelprefix,
    )
    output_path = output_dir / f"{scorer}_image_predictions.json"
    logging.info(f"Saving predictions to {output_path}")
    with open(output_path, "w") as f:
        json.dump(pred_json, f)

    return predictions

def analyze_image_folder(
    model_cfg: str | Path | dict,
    images: str | Path | list[str] | list[Path],
    snapshot_path: str | Path,
    detector_path: str | Path | None = None,
    device: str | None = None,
    max_individuals: int | None = None,
    progress_bar: bool = True,
) -> dict[str, dict[str, np.ndarray | np.ndarray]]:
    """Runs pose inference on a folder of images

    Args:
        model_cfg: The model config (or its path) used to analyze the images.
        images: The images to analyze. Can either be a directory containing images, or
            a list of paths of images.
        snapshot_path: The path of the snapshot to use to analyze the images.
        detector_path: The path of the detector snapshot to use to analyze the images,
            if a top-down model was used.
        device: The device to use to run image analysis.
        max_individuals: The maximum number of individuals to detect in each image. Set
            to the number of individuals in the project if None.
        progress_bar: Whether to display a progress bar when running inference.

    Returns:
        A dictionary mapping each image filename to the different types of predictions
        for it (e.g. "bodyparts", "unique_bodyparts", "bboxes", "bbox_scores")

    Raises:
        ValueError: if the pose model is a top-down model but no detector path is given
    """
    if not isinstance(model_cfg, dict):
        model_cfg = read_config_as_dict(model_cfg)

    pose_task = Task(model_cfg["method"])
    if pose_task == Task.TOP_DOWN and detector_path is None:
        raise ValueError(
            "A detector path must be specified for image analysis using top-down models"
            f" Please specify the `detector_path` parameter."
        )

    bodyparts = model_cfg["metadata"]["bodyparts"]
    unique_bodyparts = model_cfg["metadata"]["unique_bodyparts"]
    individuals = model_cfg["metadata"]["individuals"]
    if max_individuals is None:
        max_individuals = len(individuals)

    if device is None:
        device = resolve_device(model_cfg)

    pose_runner, detector_runner = get_inference_runners(
        model_config=model_cfg,
        snapshot_path=snapshot_path,
        max_individuals=max_individuals,
        num_bodyparts=len(bodyparts),
        num_unique_bodyparts=len(unique_bodyparts),
        device=device,
        with_identity=False,
        transform=None,
        detector_path=detector_path,
        detector_transform=None,
    )

    image_paths = parse_images_and_image_folders(images)
    pose_inputs = image_paths
    if detector_runner is not None:
        logging.info(f"Running object detection with {detector_path}")

        detector_image_paths = image_paths
        if progress_bar:
            detector_image_paths = tqdm(detector_image_paths)
        bbox_predictions = detector_runner.inference(images=detector_image_paths)
        pose_inputs = list(zip(image_paths, bbox_predictions))

    logging.info(f"Running pose estimation with {detector_path}")

    if progress_bar:
        pose_inputs = tqdm(pose_inputs)

    predictions = pose_runner.inference(pose_inputs)

    return {
        image_path: image_predictions
        for image_path, image_predictions in zip(image_paths, predictions)
    }


def read_config(configname):
    """
    Reads structured config file defining a project.
    """
    ruamelFile = YAML()
    path = Path(configname)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cfg = ruamelFile.load(f)
                curr_dir = str(Path(configname).parent.resolve())

                if cfg.get("engine") is None:
                    cfg["engine"] = Engine.TF.aliases[0]
                    write_config(configname, cfg)

                if cfg.get("detector_snapshotindex") is None:
                    cfg["detector_snapshotindex"] = -1

                if cfg.get("detector_batch_size") is None:
                    cfg["detector_batch_size"] = 1

                if cfg["project_path"] != curr_dir:
                    cfg["project_path"] = curr_dir
                    write_config(configname, cfg)
        except Exception as err:
            if len(err.args) > 2:
                if (
                    err.args[2]
                    == "could not determine a constructor for the tag '!!python/tuple'"
                ):
                    with open(path, "r") as ymlfile:
                        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
                        write_config(configname, cfg)
                else:
                    raise

    else:
        raise FileNotFoundError(
            f"Config file at {path} not found. Please make sure that the file exists and/or that you passed the path of the config file correctly!"
        )
    return cfg

def read_config_as_dict(config_path: str | Path) -> dict:
    """
    Args:
        config_path: the path to the configuration file to load

    Returns:
        The configuration file with pure Python classes
    """
    with open(config_path, "r") as f:
        cfg = YAML(typ='safe', pure=True).load(f)

    return cfg


def write_config(config_path: str | Path, config: dict, overwrite: bool = True) -> None:
    """Writes a pose configuration file to disk

    Args:
        config_path: the path where the config should be saved
        config: the config to save
        overwrite: whether to overwrite the file if it already exists

    Raises:
        FileExistsError if overwrite=True and the file already exists
    """
    if not overwrite and Path(config_path).exists():
        raise FileExistsError(
            f"Cannot write to {config_path} - set overwrite=True to force"
        )

    with open(config_path, "w") as file:
        YAML().dump(config, file)

def get_model_folder(
    trainFraction: float,
    shuffle: int,
    cfg: dict,
    modelprefix: str = "",
    engine: Engine = Engine.TF,
) -> Path:
    """
    Args:
        trainFraction: the training fraction (as defined in the project configuration)
            for which to get the model folder
        shuffle: the index of the shuffle for which to get the model folder
        cfg: the project configuration
        modelprefix: The name of the folder
        engine: The engine for which we want the model folder. Defaults to `tensorflow`
            for backwards compatibility with DeepLabCut 2.X

    Returns:
        the relative path from the project root to the folder containing the model files
        for a shuffle (configuration files, snapshots, training logs, ...)
    """
    proj_id = f"{cfg['Task']}{cfg['date']}"
    return Path(
        modelprefix,
        engine.model_folder_name,
        f"iteration-{cfg['iteration']}",
        f"{proj_id}-trainset{int(trainFraction * 100)}shuffle{shuffle}",
    )

def parse_snapshot_index_for_analysis(
    cfg: dict,
    model_cfg: dict,
    snapshot_index: int | str | None,
    detector_snapshot_index: int | str | None,
) -> tuple[int, int | None]:
    """Gets the index of the snapshots to use for data analysis (e.g. video analysis)

    Args:
        cfg: The project configuration.
        model_cfg: The model configuration.
        snapshot_index: The index of the snapshot to use, if one was given by the user.
        detector_snapshot_index: The index of the detector snapshot to use, if one
            was given by the user.

    Returns:
        snapshot_index: the snapshot index to use for analysis
        detector_snapshot_index: the detector index to use for analysis, or None if no
            detector should be used
    """
    if snapshot_index is None:
        snapshot_index = cfg["snapshotindex"]
    if snapshot_index == "all":
        logging.warning(
            "snapshotindex is set to 'all' (in the config.yaml file or as given to "
            "`analyze_...`). Running data analysis with all snapshots is very "
            "costly! Use the function 'evaluate_network' to choose the best the "
            "snapshot. For now, changing snapshot index to -1. To evaluate another "
            "snapshot, you can change the value in the config file or call "
            "`analyze_videos` or `analyze_images` with your desired snapshot index."
        )
        snapshot_index = -1

    pose_task = Task(model_cfg["method"])
    if pose_task == Task.TOP_DOWN:
        if detector_snapshot_index is None:
            detector_snapshot_index = cfg.get("detector_snapshotindex", -1)

        if detector_snapshot_index == "all":
            logging.warning(
                f"detector_snapshotindex is set to '{detector_snapshot_index}' (in the "
                "config.yaml file or as given to `analyze_...`). Running data analysis "
                "with all snapshots is very costly! Use 'evaluate_network' to choose "
                "the best detector snapshot. For now, changing the detector snapshot "
                "index to -1. To evaluate another detector snapshot, you can change "
                "the value in the config file or call `analyze_videos` or "
                "`analyze_images` with your desired detector snapshot index."
            )
            detector_snapshot_index = -1

    else:
        detector_snapshot_index = None

    return snapshot_index, detector_snapshot_index

def get_model_snapshots(
    index: int | str,
    model_folder: Path,
    task: Task,
) -> list[Snapshot]:
    """
    Args:
        index: Passing an index returns the snapshot with that index (where snapshots
            based on their number of training epochs, and the last snapshot is the
            "best" model based on validation metrics if one exists). Passing "best"
            returns the best snapshot from the training run. Passing "all" returns all
            snapshots.
        model_folder: The path to the folder containing the snapshots
        task: The task for which to return the snapshot

    Returns:
        If index=="all", returns all snapshots. Otherwise, returns a list containing a
        single snapshot, with the desired index.

    Raises:
        ValueError: If the index given is not valid
        ValueError: If index=="best" but there is no saved best model
    """
    snapshot_manager = TorchSnapshotManager(model_folder=model_folder, task=task)
    if isinstance(index, str) and index.lower() == "best":
        best_snapshot = snapshot_manager.best()
        if best_snapshot is None:
            raise ValueError(f"No best snapshot found in {model_folder}")
        snapshots = [best_snapshot]
    elif isinstance(index, str) and index.lower() == "all":
        snapshots = snapshot_manager.snapshots(include_best=True)
    elif isinstance(index, int):
        all_snapshots = snapshot_manager.snapshots(include_best=True)
        if (
            len(all_snapshots) == 0
            or len(all_snapshots) <= index
            or (index < 0 and len(all_snapshots) < -index)
        ):
            names = [s.path.name for s in all_snapshots]
            raise ValueError(
                f"Found {len(all_snapshots)} snapshots in {model_folder} (with names "
                f"{names}) with prefix {snapshot_manager.task.snapshot_prefix}. Could "
                f"not return snapshot with index {index}."
            )

        snapshots = [all_snapshots[index]]
    else:
        raise ValueError(f"Invalid snapshotindex: {index}")

    return snapshots

def get_scorer_name(
    cfg: dict,
    shuffle: int,
    train_fraction: float,
    snapshot_index: int | None = None,
    detector_index: int | None = None,
    snapshot_uid: str | None = None,
    modelprefix: str = "",
) -> str:
    """Get the scorer name for a particular PyTorch DeepLabCut shuffle

    Args:
        cfg: The project configuration.
        shuffle: The index of the shuffle for which to get the scorer
        train_fraction: The training fraction for the shuffle.
        snapshot_index: The index of the snapshot used. If None, the value is loaded
            from the project's config.yaml file.
        detector_index: For top-down models, the index of the detector used. If None,
            the value is loaded from the project's config.yaml file.
        snapshot_uid: If the snapshot_uid is not None, this value will be used instead
            of loading the snapshot and detector with given indices and calling
            utils.get_scorer_uid.
        modelprefix: The model prefix, if one was used.

    Returns:
        the scorer name
    """
    model_dir = Path(cfg["project_path"]) / auxiliaryfunctions.get_model_folder(
        train_fraction,
        shuffle,
        cfg,
        engine=Engine.PYTORCH,
        modelprefix=modelprefix,
    )
    train_dir = model_dir / "train"
    model_cfg = read_config_as_dict(str(train_dir / Engine.PYTORCH.pose_cfg_name))
    net_type = model_cfg["net_type"]
    pose_task = Task(model_cfg["method"])

    if snapshot_uid is None:
        if snapshot_index is None:
            snapshot_index = auxiliaryfunctions.get_snapshot_index_for_scorer(
                "snapshotindex", cfg["snapshotindex"]
            )
        if detector_index is None:
            detector_index = auxiliaryfunctions.get_snapshot_index_for_scorer(
                "detector_snapshotindex", cfg["detector_snapshotindex"]
            )

        snapshot = get_model_snapshots(snapshot_index, train_dir, pose_task)[0]
        detector_snapshot = None
        if detector_index is not None and pose_task == Task.TOP_DOWN:
            detector_snapshot = get_model_snapshots(
                detector_index, train_dir, Task.DETECT
            )[0]

        snapshot_uid = get_scorer_uid(snapshot, detector_snapshot)

    task, date = cfg["Task"], cfg["date"]
    name = "".join([p.capitalize() for p in net_type.split("_")])
    return f"DLC_{name}_{task}{date}shuffle{shuffle}_{snapshot_uid}"

def get_scorer_uid(snapshot: Snapshot, detector_snapshot: Snapshot | None) -> str:
    """
    Args:
        snapshot: the snapshot for which to get the scorer UID
        detector_snapshot: if a top-down model is used with a detector, the detector
            snapshot for which to get the scorer UID

    Returns:
        the uid to use for the scorer
    """
    snapshot_id = f"snapshot_{snapshot.uid()}"
    if detector_snapshot is not None:
        detect_id = detector_snapshot.uid()
        snapshot_id = f"detector_{detect_id}_{snapshot_id}"
    return snapshot_id

def resolve_device(model_config: dict) -> str:
    """Determines which device should be used from the model config

    When the device is set to 'auto':
        If an Nvidia GPU is available, selects the device as cuda:0.
        Selects 'mps' if available (on macOS) and the net type is compatible.
        Otherwise, returns 'cpu'.
    Otherwise, simply returns the selected device

    Args:
        model_config: the configuration for the pose model

    Returns:
        the device on which training should be run
    """
    device = model_config["device"]
    supports_mps = "resnet" in model_config.get("net_type", "resnet")

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif supports_mps and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device

def get_inference_runners(
    model_config: dict,
    snapshot_path: str | Path,
    max_individuals: int,
    num_bodyparts: int,
    num_unique_bodyparts: int,
    batch_size: int = 1,
    device: str | None = None,
    with_identity: bool = False,
    transform: A.BaseCompose | None = None,
    detector_batch_size: int = 1,
    detector_path: str | Path | None = None,
    detector_transform: A.BaseCompose | None = None,
) -> tuple[InferenceRunner, InferenceRunner | None]:
    """Builds the runners for pose estimation

    Args:
        model_config: the pytorch configuration file
        snapshot_path: the path of the snapshot from which to load the weights
        max_individuals: the maximum number of individuals per image
        num_bodyparts: the number of bodyparts predicted by the model
        num_unique_bodyparts: the number of unique_bodyparts predicted by the model
        batch_size: the batch size to use for the pose model.
        with_identity: whether the pose model has an identity head
        device: if defined, overwrites the device selection from the model config
        transform: the transform for pose estimation. if None, uses the transform
            defined in the config.
        detector_batch_size: the batch size to use for the detector
        detector_path: the path to the detector snapshot from which to load weights,
            for top-down models (if a detector runner is needed)
        detector_transform: the transform for object detection. if None, uses the
            transform defined in the config.

    Returns:
        a runner for pose estimation
        a runner for detection, if detector_path is not None
    """
    pose_task = Task(model_config["method"])
    if device is None:
        device = resolve_device(model_config)

    if transform is None:
        transform = build_transforms(model_config["data"]["inference"])

    detector_runner = None
    if pose_task == Task.BOTTOM_UP:
        pose_preprocessor = build_bottom_up_preprocessor(
            color_mode=model_config["data"]["colormode"],
            transform=transform,
        )
        pose_postprocessor = build_bottom_up_postprocessor(
            max_individuals=max_individuals,
            num_bodyparts=num_bodyparts,
            num_unique_bodyparts=num_unique_bodyparts,
            with_identity=with_identity,
        )
    else:
        # FIXME: Cannot run detectors on MPS
        detector_device = device
        if device == "mps":
            detector_device = "cpu"

        pose_preprocessor = build_top_down_preprocessor(
            color_mode=model_config["data"]["colormode"],
            transform=transform,
            cropped_image_size=(256, 256),
        )
        pose_postprocessor = build_top_down_postprocessor(
            max_individuals=max_individuals,
            num_bodyparts=num_bodyparts,
            num_unique_bodyparts=num_unique_bodyparts,
        )

        if detector_path is not None:
            if detector_transform is None:
                detector_transform = build_transforms(
                    model_config["detector"]["data"]["inference"]
                )

            detector_config = model_config["detector"]["model"]
            if "pretrained" in detector_config:
                detector_config["pretrained"] = False

            detector_runner = build_inference_runner(
                task=Task.DETECT,
                model=DETECTORS.build(detector_config),
                device=detector_device,
                snapshot_path=detector_path,
                batch_size=detector_batch_size,
                preprocessor=build_bottom_up_preprocessor(
                    color_mode=model_config["detector"]["data"]["colormode"],
                    transform=detector_transform,
                ),
                postprocessor=build_detector_postprocessor(
                    max_individuals=max_individuals,
                ),
            )

    pose_runner = build_inference_runner(
        task=pose_task,
        model=PoseModel.build(model_config["model"]),
        device=device,
        snapshot_path=snapshot_path,
        batch_size=batch_size,
        preprocessor=pose_preprocessor,
        postprocessor=pose_postprocessor,
    )
    return pose_runner, detector_runner
