from pathlib import Path

from deeplabcut.pose_estimation_pytorch.apis.analyze_videos import video_inference, video_inference_shrunk, inference_single_img
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners

train_dir = Path("/Users/fabionaecht/Documents/PhD/dlc/tensorflow_pytorch/refined/20230424_nn200_tip-sam/dlc-models-pytorch/iteration-1/20230424_nn200_tipApr24-trainset95shuffle1/train/")
pytorch_config_path = train_dir / "pytorch_config.yaml"
snapshot_path = train_dir / "snapshot-200.pt"

# video and inference parameters
video_path = Path("/Users/fabionaecht/Documents/PhD/dlc/tensorflow_pytorch/refined/20230424_nn200_tip-sam/test-videos/2023_04_22_15_00_tracking_trial_control_fabio_image_stack1.avi")
max_num_animals = 1
batch_size = 16

# read model configuration
model_cfg = read_config_as_dict(pytorch_config_path)
bodyparts = model_cfg["metadata"]["bodyparts"]
unique_bodyparts = model_cfg["metadata"]["unique_bodyparts"]
with_identity = model_cfg["metadata"].get("with_identity", False)
print(f"Bodyparts: {bodyparts}")
print(f"Unique bodyparts: {unique_bodyparts}")
print(f"With identity: {with_identity}")

pose_task = Task(model_cfg["method"])
print(f"Pose task: {pose_task}")

pose_runner, detector_runner = get_inference_runners(
    model_config=model_cfg,
    snapshot_path=snapshot_path,
    max_individuals=max_num_animals,
    num_bodyparts=len(bodyparts),
    num_unique_bodyparts=len(unique_bodyparts),
    batch_size=batch_size,
    with_identity=with_identity,
    transform=None,
    detector_transform=None,
)
print(f"Pose runner: {pose_runner}")

# predictions = video_inference(
#     video_path=video_path,
#     task=pose_task,
#     pose_runner=pose_runner,
#     detector_runner=detector_runner,
#     with_identity=False,
# )

# predictions = video_inference_shrunk(
#     video_path=video_path,
#     pose_runner=pose_runner,
# )

# predictions = inference_single_img( pose_runner=pose_runner)

# load images from video using iio.imread
import imageio as iio

video_reader = iio.get_reader(video_path)
frame_number_total = iio.get_reader(video_path).count_frames()
print(f"Total number of frames: {frame_number_total}")

for frame_number in range(10):
    print(f"# {frame_number}")

    # Read the current frame
    image = video_reader.get_data(frame_number)
    print(f"image shape: {image.shape}")
    print(f"image type: {type(image)}")

    # Run inference on the current frame
    predictions = pose_runner.inference_single_image(image=image)
    print(f"Predictions: {predictions}")

# Close the video reader after usage
video_reader.close()
