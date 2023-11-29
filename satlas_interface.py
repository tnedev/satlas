import os
import contextlib
import json
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

    img_formats = ['png', 'jpg', 'jpeg', 'jp2', 'tif']

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

        # Create a temporary image to measure text size
        temp_img = Image.new('RGB', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        font = ImageFont.load_default()  # Default font

        # Calculate the required height for the legend
        max_labels_per_line = image.width // 110
        legend_height_per_line = 50
        num_lines = (len(labels) + max_labels_per_line - 1) // max_labels_per_line
        legend_height = legend_height_per_line * num_lines

        # Create a legend image
        legend = Image.new('RGB', (image.width, legend_height), (255, 255, 255))
        draw = ImageDraw.Draw(legend)

        # Variables to control label placement
        spacing = 10
        item_width = 25 + 70 + spacing
        x = (image.width - item_width * min(max_labels_per_line, len(labels))) // 2  # Centered
        y = 20
        label_count = 0

        for label, color in labels.items():
            if label_count >= max_labels_per_line:
                y += legend_height_per_line
                x = 10
                label_count = 0

            draw.rectangle([x, y - 10, x + 20, y + 10], fill=color, outline=(0, 0, 0))
            x += 25  # Shift for label text
            text_width, text_height = 70, 30
            draw.text((x, y - 10), label, fill=(0, 0, 0), font=font)
            x += text_width + spacing
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

    def evaluate(self, add_legend=False):
        """
        Evaluate the model on the image.
        """

        # Check if image is loaded
        if self.input_image is None:
            print('No image loaded.')
            return

        head_idx = 0
        crop_size = 1024
        vis_output = np.zeros((self.input_image.shape[0], self.input_image.shape[1], 3), dtype=np.uint8)

        with torch.no_grad():

            used_labels = []
            for crop, row, col, original_h, original_w in tqdm(self._image_crop_generator()):
                vis_crop = crop.transpose(2, 0, 1)
                gpu_crop = torch.as_tensor(vis_crop.copy()).to(self.device).float() / 255
                outputs, _ = self.model([gpu_crop])

                result_crop, _, _, _ = satlas.model.evaluate.visualize_outputs(
                                task=self.config['Tasks'][head_idx]['Task'],
                                image=crop,
                                outputs=outputs[head_idx][0],
                                return_vis=True,
                            )

                vis_output[row:row+crop_size, col:col+crop_size, :] = result_crop[:original_h, :original_w, :]
                used_labels += list(set(outputs[0][0]['labels'].tolist()))
            
        image = Image.fromarray(vis_output)

        if add_legend:
            labels = tasks['polygon']

            labels_colors = {}
            for used in used_labels:
                labels_colors[labels["categories"][used]] = tuple(labels["colors"][used])
            image = self._add_legend(image, labels_colors)
        image.show()


sat = Satlas()
sat.load_image('./images/test_image.png')
sat.evaluate(add_legend=True)
