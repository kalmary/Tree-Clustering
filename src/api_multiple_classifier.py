import torch
from torchvision import transforms
import base64
from io import BytesIO
from openai import OpenAI


class LLM_Classifier:
    def __init__(self, resolution: int, species: dict,
                 API_KEY: str | None, model: str):
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
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            images_bytes = buffer.getvalue()
            images_base64.append(base64.b64encode(images_bytes).decode("utf-8"))
        return images_base64

    # Fixed view order produced by _claude2images
    VIEW_NAMES = ["TOP", "FRONT", "BACK", "LEFT", "RIGHT"]

    def api_call(self, images_base64: list[str]) -> int:
        """
        Sends the 5 depth-map views to the model and asks it to pick
        a species key from self.species. Returns the integer key.
        """
        assert len(images_base64) == len(self.VIEW_NAMES), (
            f"Expected {len(self.VIEW_NAMES)} views, got {len(images_base64)}"
        )

        # Render the species list once, deterministically ordered
        species_lines = "\n".join(
            f"  <species key=\"{k}\">{v[1]}</species>"
            for k, v in sorted(self.species.items(), key=lambda kv: kv[0])
        )
        valid_keys = ", ".join(str(k) for k in sorted(self.species.keys()))

        system_prompt = (
            "<role>\n"
            "You are a dendrologist specialised in tree species identification "
            "from airborne and terrestrial LiDAR point clouds. You reason about "
            "3D tree structure from rasterised orthographic depth maps.\n"
            "</role>\n\n"
            "<task>\n"
            "Identify the single best-matching species of one tree, given five "
            "orthographic depth-map views of its point cloud.\n"
            "</task>\n\n"
            "<input_format>\n"
            "You receive exactly five grayscale depth maps in this fixed order:\n"
            "  1. TOP    — looking straight down (-Z), shows crown footprint and branch spread.\n"
            "  2. FRONT  — looking along -Y, shows full silhouette and trunk.\n"
            "  3. BACK   — looking along +Y, mirror of FRONT.\n"
            "  4. LEFT   — looking along -X.\n"
            "  5. RIGHT  — looking along +X, mirror of LEFT.\n"
            "Brighter pixels are closer to the camera; black is empty space. "
            "Each view is independently min-max normalised, so absolute brightness "
            "is not comparable across views — only shape is.\n"
            "</input_format>\n\n"
            "<reasoning_guidelines>\n"
            "Use ALL five views jointly. Base your decision on structural evidence:\n"
            "  - Overall silhouette and height-to-width ratio (FRONT/BACK/LEFT/RIGHT).\n"
            "  - Crown shape: conical, columnar, rounded, spreading, weeping, irregular.\n"
            "  - Crown footprint and symmetry (TOP).\n"
            "  - Branching pattern: opposite vs alternate, dense vs sparse, ascending vs horizontal vs drooping.\n"
            "  - Trunk form: straight single leader vs forked vs leaning; relative trunk thickness.\n"
            "  - Foliage density and texture as visible in the depth map.\n"
            "Ignore scanning artefacts, isolated stray points, and ground/understorey returns. "
            "Do NOT infer colour, bark texture, leaf shape, or seasonality — they are not visible. "
            "If multiple species are plausible, pick the one whose typical 3D form best matches "
            "the combined evidence across all five views. Never invent a species not in the list.\n"
            "</reasoning_guidelines>\n\n"
            "<output_contract>\n"
            "Respond with EXACTLY ONE integer — the key from the species list — "
            "and nothing else. No words, no punctuation, no whitespace beyond the digits, "
            "no markdown, no code fences, no explanation. Any deviation is an error.\n"
            "</output_contract>"
        )

        user_text = (
            "<species_list>\n"
            f"{species_lines}\n"
            "</species_list>\n\n"
            f"<valid_keys>{valid_keys}</valid_keys>\n\n"
            "<instructions>\n"
            "The five attached images are, in order: TOP, FRONT, BACK, LEFT, RIGHT "
            "(each labelled in the message). Examine all of them, reason silently about "
            "the tree's 3D structure, and output the integer key of the best matching species.\n"
            "</instructions>"
        )

        content = []
        for name, b64 in zip(self.VIEW_NAMES, images_base64):
            content.append({"type": "input_text", "text": f"View: {name}"})
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{b64}"
            })
        content.append({"type": "input_text", "text": user_text})

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

    def predict(self, points) -> int:
        """
        Full pipeline: point cloud -> 5 depth maps -> base64 -> LLM -> species key.
        """
        points = torch.from_numpy(points)
        depth_maps = self._claude2images(points)
        images_b64 = self.tensors_to_base64(depth_maps)
        key = self.api_call(images_b64)
        return key