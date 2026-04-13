import os
from dotenv import load_dotenv
import torch
from torchvision import transforms
import base64
from io import BytesIO
from openai import OpenAI
import numpy as np

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL = "gpt-5.4"

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
    def __init__(self, resolution: int, species: dict = species,
                 API_KEY: str | None = OPENAI_API_KEY, model: str = MODEL):
        self.resolution = resolution
        self.species = species
        self.client = OpenAI(api_key=API_KEY) if API_KEY is not None else None
        self.model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0


    def _claude2images(self, points: torch.Tensor,
                       resolution_xy: int | None = None,
                       margin_ratio: float = 0.05) -> torch.Tensor:
        """
        Converts a 3D point cloud into 5 orthographic depth maps.
        Returns: (5, resolution_xy, resolution_xy) float32 tensor
                 in order: top, front, back, left, right.
        """
        if resolution_xy is None:
            resolution_xy = self.resolution

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
            depth_map = torch.full((resolution_xy * resolution_xy,), float('inf'),
                                   dtype=torch.float64, device=distances.device)
            depth_map = torch.scatter_reduce(depth_map, 0, flat_indices, distances,
                                             reduce='amin', include_self=True)

            img = depth_map.view(resolution_xy, resolution_xy)
            img[img == float('inf')] = 0

            nonzero_mask = img > 0
            if torch.any(nonzero_mask):
                values = img[nonzero_mask]
                min_val = values.min()
                max_val = values.max()
                img[nonzero_mask] = (values - min_val) / (max_val - min_val + 1e-8)

            return img.type(torch.float32)

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

    def tensors_to_base64(self, tensor: torch.Tensor) -> list[str]:
        """Convert a (N, H, W) tensor of depth maps into a list of base64 PNG strings."""
        images_base64 = []
        to_pil = transforms.ToPILImage()

        for i in range(tensor.shape[0]):
            pil_image = to_pil(tensor[i])
            pil_image.show()

            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")

            images_bytes = buffer.getvalue()
            images_base64.append(base64.b64encode(images_bytes).decode("utf-8"))
        return images_base64

    def api_call(self, images_base64: list[str]) -> int:
        """
        Sends the 5 depth-map views to the model and asks it to pick
        a species key from self.species. Returns the integer key.
        """
        # Build a readable list of valid keys + Latin names for the prompt
        species_list = "\n".join(
            f"{k}: {v[0]} ({v[1]})" for k, v in self.species.items()
        )

        system_prompt = (
            "You are a forestry expert classifying trees from LiDAR point clouds. "
            "You will be given 5 orthographic depth-map views of a single tree "
            "(top, front, back, left, right). Based on overall shape, crown "
            "structure, branching pattern, trunk form, and silhouette, identify "
            "the single best matching species from the provided species dictionary. "
            "Use all 5 views together and rely only on visible structural evidence in "
            "the depth maps. Be consistent and conservative: if uncertain, choose the "
            "closest match from the provided list based on overall 3D form rather than "
            "guessing from minor artifacts. You MUST respond with ONLY a single integer "
            "— the key from the provided species dictionary. No explanation, no "
            "punctuation, no extra text. Just the integer."
        )

        user_text = (
            "Classify this tree into exactly one of the following species. "
            "Consider all 5 views before deciding, and respond with ONLY the integer key.\n\n"
            f"Valid keys and species:\n{species_list}"
        )

        content = [{"type": "input_text", "text": user_text}]

        for b64 in images_base64:
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{b64}",
            })

        response = self.client.responses.create(
            model=self.model,
            temperature=0,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )

        if hasattr(response, "usage") and response.usage is not None:
            self.prompt_tokens += response.usage.input_tokens or 0
            self.completion_tokens += response.usage.output_tokens or 0
        raw = response.output_text.strip()

        # Robust parse: grab first integer in the reply
        import re
        match = re.search(r"-?\d+", raw)
        if match is None:
            raise ValueError(f"Could not parse species key from model reply: {raw!r}")

        key = int(match.group(0))

        if key not in self.species:
            raise ValueError(
                f"Model returned key {key} not in species dict. Raw: {raw!r}"
            )

        return key

    def classify(self, path: str) -> dict:
        """
        Full pipeline: point cloud -> 5 depth maps -> base64 -> LLM -> species key.

        Returns:
            dict with keys: 'key', 'latin', 'polish'
        """
        points = np.load(path)
        points = torch.from_numpy(points)
        depth_maps = self._claude2images(points)
        images_b64 = self.tensors_to_base64(depth_maps)
        key = self.api_call(images_b64)
        latin, polish = self.species[key]
        return {"key": key, "latin": latin, "polish": polish, 'path': path}

PATH = 'trees/0_3.npy'
clf = LLM_Classifier(512)
print(clf.classify(path=PATH))
print(clf.prompt_tokens)
print(clf.completion_tokens)