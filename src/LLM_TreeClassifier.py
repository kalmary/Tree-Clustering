import numpy as np
import torch
import ollama
from PIL import Image
import tempfile
import os

class LLM_Classifier:
    def __init__(self,
                 resolution: int,
                 species: dict):
        """
        :param resolution: Pixel resolution for the depth maps
        :param species: Dictionary mapping int labels to species names {0: "Oak", 1: "Pine"}
        """
        self.resolution = resolution
        self.species = species
        # Construct the species list for the prompt
        self.prompt = (
            f"You are an expert forest ecologist. "
            f"You are given 5 grayscale depth-map images of a single tree, "
            f"each showing a different side view. "
            f"Based on the tree's overall shape, crown form, and branching structure "
            f"visible across all views, identify the most likely tree species.\n\n"
            f"Reply with ONLY the integer class number -- nothing else.\n\n"
            f"Classes:\n{self.species}\n\n"
            f"Reply with the single integer and NOTHING more."
        )
        self.model_name = 'gemma4:31b'

    def predict(self, cloud: np.ndarray) -> int:
        # 1. Generate the 5 depth-map views
        views_tensor = self._cloud2sideViews(cloud)
        
        # 2. Save views to temporary files so Ollama can read them
        image_paths = []
        temp_dir = tempfile.gettempdir()
        
        for i, view in enumerate(views_tensor):
            # Convert tensor to 0-255 uint8 grayscale image
            img_np = (view.numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            path = os.path.join(temp_dir, f"view_{i}.png")
            img.save(path)
            image_paths.append(path)

        # 4. Call Ollama
        response = ollama.chat(
            model=self.model_name,
            messages=[{
                'role': 'user',
                'content': self.prompt,
                'images': image_paths
            }],
            options={'temperature': 0} # Set to 0 for deterministic classification
        )

        # 5. Cleanup temp files
        for p in image_paths:
            os.remove(p)

        # 6. Parse and return the integer
        try:
            return int(response.message.content.strip())
        except ValueError:
            print(f"Warning: Model returned non-integer response: {response.message.content}")
            return -1

    def _cloud2sideViews(self, points: np.ndarray, margin_ratio: float = 0.05) -> torch.Tensor:
        # (Your existing logic here - corrected the reference to resolution_xy)
        resolution_xy = self.resolution
        points = torch.from_numpy(points).type(dtype=torch.float64)

        min_xyz = points.min(dim=0).values
        max_xyz = points.max(dim=0).values

        center = (min_xyz + max_xyz) / 2
        max_range = (max_xyz - min_xyz).max()
        cube_half = max_range / 2 * (1 + 2 * margin_ratio)

        cube_min = center - cube_half
        cube_max = center + cube_half

        def to_grid(val, min_val, max_val):
            return torch.clamp(
                ((val - min_val) / (max_val - min_val + 1e-8) * (resolution_xy - 1)).long(),
                0, resolution_xy - 1
            )

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        gx, gy, gz = to_grid(x, cube_min[0], cube_max[0]), to_grid(y, cube_min[1], cube_max[1]), to_grid(z, cube_min[2], cube_max[2])

        views = []

        def build_depth_map(indices_2d, distances, flip_y=False, flip_x=False):
            y_idx, x_idx = indices_2d
            if flip_y: y_idx = resolution_xy - 1 - y_idx
            if flip_x: x_idx = resolution_xy - 1 - x_idx

            flat_indices = y_idx * resolution_xy + x_idx
            depth_map = torch.full((resolution_xy * resolution_xy,), float('inf'), dtype=torch.float64)

            depth_map = torch.scatter_reduce(depth_map, 0, flat_indices, distances, reduce='amin', include_self=True)
            img = depth_map.view(resolution_xy, resolution_xy)
            img[img == float('inf')] = 0 

            nonzero_mask = img > 0
            if torch.any(nonzero_mask):
                values = img[nonzero_mask]
                img[nonzero_mask] = (values - values.min()) / (values.max() - values.min() + 1e-8)

            return img.type(torch.float32)

        # Projections
        views.append(build_depth_map((gy, gx), cube_max[2] - z)) # Top
        views.append(build_depth_map((gz, gx), cube_max[1] - y, flip_y=True)) # Front
        views.append(build_depth_map((gz, gx), y - cube_min[1], flip_y=True, flip_x=True)) # Back
        views.append(build_depth_map((gz, gy), cube_max[0] - x, flip_y=True)) # Left
        views.append(build_depth_map((gz, gy), x - cube_min[0], flip_y=True, flip_x=True)) # Right

        return torch.stack(views, dim=0)