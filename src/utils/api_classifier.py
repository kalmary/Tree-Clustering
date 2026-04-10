import os
from dotenv import load_dotenv
import torch
from torchvision import transforms
import base64
from io import BytesIO
from openai import OpenAi
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

species = {
        0:  ["Betula_pendula",         "Brzoza brodawkowata"],
        1:  ["Fagus_sylvatica",        "Buk zwyczajny"],
        2:  ["Quercus_petraea",        "Dąb bezszypułkowy"],
        3:  ["Quercus_rubra",          "Dąb czerwony"],
        4:  ["Quercus_robur",          "Dąb szypułkowy"],
        5:  ["Carpinus_betulus",       "Grab pospolity"],
        6:  ["Fraxinus_excelsior",     "Jesion wyniosły"],
        7:  ["Acer_pseudoplatanus",    "Klon jawor"],
        8:  ["Acer_campestre",         "Klon polny"],
        9:  ["Tilia_cordata",          "Lipa drobnolistna"],
        10: ["Ulmus_laevis",           "Wiąz szypułkowy"],
        11: ["Crataegus_monogyna",     "Głóg jednoszyjkowy"],
        12: ["Corylus_avellana",       "Leszczyna pospolita"],
        13: ["Pseudotsuga_menziesii",  "Daglezja zielona"],
        14: ["Abies_alba",             "Jodła pospolita"],
        15: ["Larix_decidua",          "Modrzew europejski"],
        16: ["Pinus_sylvestris",       "Sosna zwyczajna"],
        17: ["Picea_abies",            "Świerk pospolity"],
        18: ["Other",                  "Inne"],
        19: ["Incorrect segmentation", "Błędna segmentacja"],
    }

class LLM_Classifier:
    def __init__(self, resolution: int, species: dict, API_KEY:str, model: str ):
        self.resolution = resolution
        self.species = species
        self.client = OpenAi(api_key=API_KEY)
 

    def _claude2images(self, points: torch.Tensor, resolution_xy: int, margin_ratio: float = 0.05) -> torch.Tensor:
        """
        Converts a 3D point cloud into a set of 2D depth map images from multiple orthographic viewpoints.

        The point cloud is first normalized into a cubic bounding volume with an optional margin.
        Five depth maps are then rendered — top, front, back, left, and right — by projecting
        points onto the corresponding axis-aligned planes and recording the distance from each
        point to its respective camera wall. Each depth map is normalized to [0, 1] and returned
        as a float32 tensor.

        Args:
            points (torch.Tensor): A (N, 3) tensor of 3D points (x, y, z).
            resolution_xy (int): The pixel resolution of each output depth map (resolution_xy x resolution_xy).
            margin_ratio (float): Fractional padding added around the bounding cube on each side. Default is 0.05.

        Returns:
            torch.Tensor: A (5, resolution_xy, resolution_xy) float32 tensor containing the five depth maps
                        in the order: top, front, back, left, right.
        """
        points = points.type(torch.float64)

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

        gx = to_grid(x, cube_min[0], cube_max[0])
        gy = to_grid(y, cube_min[1], cube_max[1])
        gz = to_grid(z, cube_min[2], cube_max[2])

        views = []

        def build_depth_map(indices_2d, distances, flip_y=False, flip_x=False):
            y_idx, x_idx = indices_2d
            if flip_y:
                y_idx = resolution_xy - 1 - y_idx
            if flip_x:
                x_idx = resolution_xy - 1 - x_idx

            flat_indices = y_idx * resolution_xy + x_idx
            depth_map = torch.full((resolution_xy * resolution_xy,), float('inf'), dtype=torch.float64,
                                device=distances.device)

            # Use scatter_reduce to keep the minimum distance per pixel
            depth_map = torch.scatter_reduce(depth_map, 0, flat_indices, distances, reduce='amin', include_self=True)

            img = depth_map.view(resolution_xy, resolution_xy)
            img[img == float('inf')] = 0  # Replace untouched pixels

            # Normalize non-zero pixels to [0, 1]
            nonzero_mask = img > 0
            if torch.any(nonzero_mask):
                values = img[nonzero_mask]
                min_val = values.min()
                max_val = values.max()
                img[nonzero_mask] = (values - min_val) / (max_val - min_val + 1e-8)

            return img.type(torch.float32)

        # Compute distance from each wall
        dist_top = cube_max[2] - z
        views.append(build_depth_map((gy, gx), dist_top))

        dist_front = cube_max[1] - y
        views.append(build_depth_map((gz, gx), dist_front, flip_y=True))

        dist_back = y - cube_min[1]
        views.append(build_depth_map((gz, gx), dist_back, flip_y=True, flip_x=True))

        dist_left = cube_max[0] - x
        views.append(build_depth_map((gz, gy), dist_left, flip_y=True))

        dist_right = x - cube_min[0]
        views.append(build_depth_map((gz, gy), dist_right, flip_y=True, flip_x=True))

        return torch.stack(views, dim=0).type(torch.float32)
    
    def tensors_to_base64(self, tensor):
        
        img_base64 = []
        for i in range(tensor.size[0]): 
            # tensor to PIL image
            pil_image = transforms.ToPILImage()(tensor(i))

            #PIL Image to bytes
            buffer = BytesIO()
            pil_image.save(buffer, "PNG")

            # bytes to base64 string
            img_bytes = buffer.getvalue()
            img_base64.append(base64.b64encode(img_bytes).decode("utf-8"))

        return img_base64

    def api_call(self):
        system_prompt = ""

        response = self.client.responses.create(model = self.mod)

