import os
import contextlib
import json
import math
from enum import Enum
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import satlas.model.evaluate
import satlas.model.model
from satlas.model.dataset import tasks

# Weights for the Swin-v2-Base Sentinel-2, single-image, RGB model
MODEL_WEIGHTS_URL = 'https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/sentinel2/si_sb_base.pth'


class Satlas:
    """
    Satlas interface for using the Swin-v2-Base Sentinel-2, single-image, RGB model.
    """

    img_formats = ['png', 'jpg', 'jpeg', 'jp2', 'tif', 'tiff']

    class Task(Enum):
        """
        Enum for the different tasks that the model can perform.
        """
        POLYGON = 0
        POINT = 1
        LAND_COVER = 2
        POLYLINE_BIN_SEGMENT = 3
        DEM = 4
        CROP_TYPE = 5
        TREE_COVER = 6
        PARK_SPORT = 7
        PARK_TYPE = 8
        POWER_PLANT_TYPE = 9
        QUARRY_RESOURCE = 10
        TRACK_SPORT = 11
        ROAD_TYPE = 12
        WILDFIRE = 13
        VESSEL = 14
        SMOKE = 15
        SNOW = 16
        FLOOD = 17
        CLOUD = 18

    def __init__(self):
        """
        Initialize the Satlas interface.
        Download the model if it does not exist.
        Load the model.
        """

        self.weights_path = './data/weights/si_sb_base.pth'
        self.config_path = './configs/sentinel2/si_sb_base.txt'
        self.input_image = None

        # Create directories if they do not exist
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)

        # Download the model if it doesn't exist
        if not os.path.exists(self.weights_path):
            print('Model not found. Downloading...')
            os.system(f'wget {MODEL_WEIGHTS_URL} --show-progress -O {self.weights_path}')

            # Load configuration
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            print("Configuration file not found.")
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for spec in self.config['Tasks']:
            if 'Task' not in spec:
                spec['Task'] = satlas.model.dataset.tasks[spec['Name']]

        with open(os.path.devnull, "w") as f, contextlib.redirect_stdout(f):  # Suppress stdout
            self.model = satlas.model.model.Model({
                'config': self.config['Model'],
                'channels': self.config['Channels'],
                'tasks': self.config['Tasks'],
            })

        state_dict = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _add_legend(self, image, labels):
        """
        Add a legend to the image.
        """

        scaling_factor = image.width / 1024, image.height / 1024  # Scale legend based on image size
        text_width, text_height = int(70 * scaling_factor[0]), int(30 * scaling_factor[1])
        spacing = int(20 * scaling_factor[0])
        item_width = spacing * 2 + text_width * 2 + text_height
        max_labels_per_line = image.width // item_width
        legend_height_per_line = int(50 * scaling_factor[1])
        num_lines = (len(labels) + max_labels_per_line - 1) // max_labels_per_line
        legend_height = legend_height_per_line * num_lines

        # Create a temporary image to measure text size
        font = ImageFont.load_default(size=text_height)  # Default font

        # Create a legend image
        legend = Image.new('RGB', (image.width, legend_height), (255, 255, 255))
        draw = ImageDraw.Draw(legend)

        # Variables to control label placement
        x = (image.width - item_width * min(max_labels_per_line, len(labels))) // 2  # Centered
        y = text_height // 2
        label_count = 0

        for label, color in labels.items():
            if label_count >= max_labels_per_line:
                y += legend_height_per_line
                x = (image.width - item_width * min(max_labels_per_line, len(labels))) // 2  # Centered
                label_count = 0

            draw.rectangle([x, y, x + text_height, y + text_height], fill=color, outline=(0, 0, 0))
            x += text_height + spacing  # Shift for label text
            draw.text((x, y - text_height / 5), label, fill=(0, 0, 0), font=font)
            x += item_width
            label_count += 1

        # Combine image with legend
        combined = Image.new('RGB', (image.width, image.height + legend.height))
        combined.paste(image, (0, 0))
        combined.paste(legend, (0, image.height))

        return combined

    def _image_crop_generator(self, crop_size=1024):
        """
        Generator function to yield crops of the image. It pads the image to the nearest multiple of crop_size.
        """
        image = self.input_image

        # Calculate padding to make image dimensions multiples of crop_size
        padded_height = ((image.shape[0] - 1) // crop_size + 1) * crop_size
        padded_width = ((image.shape[1] - 1) // crop_size + 1) * crop_size

        # Pad the image
        padded_image = np.zeros((padded_height, padded_width, image.shape[2]), dtype=image.dtype)
        padded_image[:image.shape[0], :image.shape[1], :] = image

        # Yield crops and their original sizes
        for row in range(0, padded_height, crop_size):
            for col in range(0, padded_width, crop_size):
                # Calculate the dimensions of the crop
                crop_end_row = min(row + crop_size, padded_height)
                crop_end_col = min(col + crop_size, padded_width)
                crop = padded_image[row:crop_end_row, col:crop_end_col, :]

                # Determine the original size of the crop (before padding)
                # If the crop includes padded area, adjust the original height and width accordingly
                original_height = min(crop_size, image.shape[0] - row)
                original_width = min(crop_size, image.shape[1] - col)

                yield crop, row, col, original_height, original_width

    def _handle_classifications_output(self, classifications, task_name):
        likelihood = classifications.tolist()
        # Print each category and its likelihood in % (rounded to 2 decimal places)
        for i, category in enumerate(tasks[task_name].get('categories', [])):
            print(f'{category}: {round(likelihood[i] * 100, 2)}%')

    def _handle_image_output(self, image, task_name, used_labels, add_legend=False):
        # Get the ids of all unique labels used
        if not used_labels:
            used_labels = list(range(len(tasks[task_name].get('categories', []))))
        else:
            used_labels = list(set(used_labels))  # Remove duplicates
        image = Image.fromarray(image)

        if add_legend:
            labels = tasks[task_name]

            labels_colors = {}
            for used in used_labels:
                labels_colors[labels["categories"][used]] = tuple(labels["colors"][used])
            image = self._add_legend(image, labels_colors)
        image.show()

    def load_image(self, image_path):
        """
        Load an image from a path.
        Supported formats: .tif, .png, .jpg, .jpeg, jp2
        Channels: RGB
        """

        # Check if file exists and format is supported
        if not os.path.exists(image_path):
            print('File not found.')
            return
        elif not image_path.lower().endswith(tuple(self.img_formats)):
            print('Unsupported image format.')
            return

        image = Image.open(image_path).convert('RGB')

        # Convert the image to a NumPy array
        self.input_image = np.asarray(image)

        return self.input_image

    def evaluate(self, task: Task = Task.POLYGON, add_legend=False):
        """
        Evaluate the model on the image.
        """

        # Check if image is loaded
        if self.input_image is None:
            print('No image loaded.')
            return

        task_index = task.value
        task_name = task.name.lower()
        task_type = tasks[task_name]['type']

        crop_size = 1024
        num_crops = (
                    math.ceil(self.input_image.shape[0] / crop_size) * math.ceil(self.input_image.shape[1] / crop_size))
        vis_output = np.zeros((self.input_image.shape[0], self.input_image.shape[1], 3), dtype=np.uint8)
        used_labels = []  # Keep track of used labels (int) to add to legend
        classification_results = torch.empty(len(tasks[task_name].get('categories', [])))

        with torch.no_grad():
            for crop, row, col, original_h, original_w in tqdm(self._image_crop_generator(), total=num_crops):
                vis_crop = crop.transpose(2, 0, 1)
                gpu_crop = torch.as_tensor(vis_crop.copy()).to(self.device).float() / 255
                outputs, _ = self.model([gpu_crop])

                result_crop, _, _, _ = satlas.model.evaluate.visualize_outputs(
                    task=self.config['Tasks'][task_index]['Task'],
                    image=crop,
                    outputs=outputs[task_index][0],
                    return_vis=True,
                )

                if task_type == 'classification':
                    new_results = outputs[task_index][0]
                    # When splitting an image, take the max from each image
                    classification_results = torch.max(new_results, classification_results)
                else:
                    if len(result_crop.shape) == 2:
                        vis_output[row:row + crop_size, col:col + crop_size, 0] = result_crop[:original_h, :original_w]
                        vis_output[row:row + crop_size, col:col + crop_size, 1] = result_crop[:original_h, :original_w]
                        vis_output[row:row + crop_size, col:col + crop_size, 2] = result_crop[:original_h, :original_w]
                    else:
                        vis_output[row:row + crop_size, col:col + crop_size, :] = result_crop[:original_h, :original_w, :]
                    if type(outputs[task_index][0]) == dict:  # Check if the model has labels in the output
                        used_labels += outputs[task_index][0]['labels'].tolist()

        if task_type == 'classification':
            self._handle_classifications_output(classification_results, task_name)

        else:
            self._handle_image_output(vis_output, task_name, used_labels, add_legend)
