import re
import torch
from torchvision import transforms
import base64
from io import BytesIO
from openai import OpenAI
import numpy as np

class DummyClassifier:
    def predict(self, cloud: np.ndarray) -> int:
        return -1

class LLM_Classifier:
    
    VIEW_NAMES = ["TOP", "FRONT", "BACK", "LEFT", "RIGHT"]

    # Bump the version suffix whenever the static prefix (system prompt + species list) changes.

    PROMPT_CACHE_KEY = "tree-classifier-v6"

    def __init__(self, resolution: int, species: dict,
                 API_KEY: str | None, model: str,
                 prompt_cache_retention: str | None = "24h"):
        self.resolution = resolution
        self.species = species
        self.client = OpenAI(api_key=API_KEY) if API_KEY is not None else None
        self.model = model
        self.prompt_cache_retention = prompt_cache_retention
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0

        self._system_prompt = self._build_system_prompt()
        self._static_user_prefix = self._build_static_user_prefix()

    def _cloud2images(self, points: torch.Tensor,
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
            valid_mask = torch.isfinite(img)

            if torch.any(valid_mask):
                values = img[valid_mask]
                min_val = values.min()
                max_val = values.max()
                normalised = (max_val - values) / (max_val - min_val + 1e-8)
                normalised = normalised * (1.0 - 1.0 / 255.0) + (1.0 / 255.0)
                img = img.clone()
                img[valid_mask] = normalised
                img[~valid_mask] = 0.0
            else:
                img = torch.zeros_like(img)

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

    # ------------------------------------------------------------------
    # Static prefix construction (cached)
    # ------------------------------------------------------------------
    def _build_system_prompt(self) -> str:
        """
        Large, fully static system prompt. Goes at the very beginning of the
        request so it becomes the cacheable prefix. Do not interpolate any
        per-call data here.
        """
        return (
                "<role>\n"
                "You are an expert dendrologist and LiDAR-interpretation "
                "specialist. Your task is to identify a single tree species "
                "from exactly five orthographic depth maps rendered from a 3D "
                "LiDAR point cloud of one tree. You will be shown the depth "
                "maps in a fixed order: TOP, FRONT, BACK, LEFT, RIGHT. Each "
                "image is preceded by a 'View: <n>' text marker.\n"
                "</role>\n"
                "\n"
                "<input_format>\n"
                "Each image is an 8-bit grayscale PNG produced by orthographic "
                "projection of the LiDAR point cloud onto one plane, with "
                "per-view min-max depth normalisation. Encoding:\n"
                "  pure black (0) : empty space (no LiDAR return)\n"
                "  dark gray      : farthest returning points\n"
                "  mid gray       : mid-depth points\n"
                "  near-white (1) : closest points to the camera\n"
                "In short: closer -> brighter; empty -> black. Brightness "
                "encodes depth, not material, so it varies strongly across a "
                "single tree (near side light, far side dark, thick interior "
                "possibly hidden). Because each view is normalised "
                "independently, absolute brightness is NOT comparable between "
                "views - only spatial distribution and within-view gradients "
                "carry information.\n"
                "\n"
                "Critical properties of the rendering:\n"
                "- These are NOT filled silhouettes. Projection happens at "
                "  pixel resolution, so interior black speckle is usually "
                "  sampling gaps, NOT real structural voids.\n"
                "- Point density varies: crowns look like dense speckled "
                "  regions, trunks like thin vertical streaks, and occluded "
                "  regions (upper trunk under a dense crown) may be almost "
                "  absent.\n"
                "- The silhouette must be INFERRED from the outer envelope of "
                "  lit pixels, not read as a clean outline.\n"
                "- Individual lit pixels are surface hits, not leaves or "
                "  branches - do not count them.\n"
                "\n"
                "What each view is best for:\n"
                "- TOP: crown footprint. Judge horizontal extent, symmetry, "
                "  and whether the crown is one compact region or fragmented. "
                "  The trunk often appears as a small dense spot near centre.\n"
                "- FRONT and BACK: opposing side profiles (rotated 180 deg). "
                "  Best for overall silhouette - total height, "
                "  height-to-width ratio, crown shape (columnar / conical / "
                "  ovoid / rounded / spreading / weeping / umbrella), trunk "
                "  visibility, and lowest-branch height.\n"
                "- LEFT and RIGHT: the perpendicular pair of side profiles.\n"
                "</input_format>\n"
                "\n"
                "<reasoning_guidelines>\n"
                "Base identification on geometric evidence only. Describe what "
                "you see in shape terms, then match to whichever species is "
                "most consistent.\n"
                "\n"
                "Useful geometric features, in rough order of diagnostic "
                "power:\n"
                "\n"
                "1. Overall crown shape from the four side views. Categorise "
                "   into a shape family before naming a species: conical with "
                "   clear apex, ovoid, broadly rounded, columnar/narrow, "
                "   spreading-with-short-trunk, weeping/pendulous, "
                "   umbrella/flat-topped, or irregular/asymmetric.\n"
                "\n"
                "2. Height-to-width aspect ratio from the side views. "
                "   Tall-and-narrow, roughly balanced, and wider-than-tall are "
                "   three distinct regimes that strongly constrain species.\n"
                "\n"
                "3. Trunk visibility and branching onset. In the lower half of "
                "   side views: long clean bare trunk before the crown, "
                "   foliage beginning close to the ground, or trunk splitting "
                "   into multiple stems? These distinguish forest-grown, "
                "   open-grown, and multi-stemmed forms.\n"
                "\n"
                "4. Crown density and internal structure. Uniformly dense "
                "   speckled interior suggests a well-foliated crown; sparse, "
                "   skeletal crown with lots of internal empty space suggests "
                "   an open or leafless architecture.\n"
                "\n"
                "5. TOP-view footprint shape. Round, elongated, fragmented, or "
                "   off-centre relative to the trunk position.\n"
                "\n"
                "What to IGNORE:\n"
                "- Stray lit pixels detached from the main structure"
                "- Thin horizontal bands of lit pixels at the very bottom of "
                "  side views - these are ground returns.\n"
                "- Isolated speckle in otherwise black regions.\n"
                "</reasoning_guidelines>\n"
                "\n"
                "<validation_rules>\n"
                "Return 16 (segmentation error / unclassifiable) when ANY of "
                "the following is true.\n"
                "\n"
                "A. Multiple-tree signal. Two or more clearly separated tree "
                "   structures: TOP view shows two or more distinct dense "
                "   regions with clear empty space between them (not just one "
                "   irregular region with sampling gaps), or side views show "
                "   two separate silhouettes with markedly different heights "
                "   or shapes that cannot be explained as one tree viewed "
                "   obliquely. A single tree with a lopsided, irregular, or "
                "   gap-filled crown is NOT a multi-tree case.\n"
                "\n"
                "B. Insufficient structure. No coherent tree envelope:\n"
                "   - No discernible trunk in any side view (no vertical "
                "     streak of denser points, no clear main axis).\n"
                "   - Lit pixels form a flat mat, scattered blob, or vaguely "
                "     horizontal vegetation with no clear vertical extent "
                "     relative to horizontal extent.\n"
                "   - Side views look like low shrubbery, understory "
                "     fragments, or ground debris.\n"
                "   - Height-to-width ratio is roughly 1 or less AND the "
                "     structure is not clearly a broad spreading crown on a "
                "     visible trunk.\n"
                "\n"
                "C. Shape inconsistency across views. The four side views "
                "   disagree so strongly about crown shape, height, or outline "
                "   that no single shape family fits. Genuine trees look "
                "   similar FRONT vs BACK (mirror) and broadly similar LEFT vs "
                "   RIGHT. If they do not, segmentation is probably flawed.\n"
                "\n"
                "</validation_rules>\n"
                "\n"
                "<output_contract>\n"
                "Output exactly one integer and nothing else: either the "
                "species key from the list, or 16 if any validation rule "
                "triggered. No explanation, no punctuation, no surrounding "
                "text.\n"
                "</output_contract>"
        )

    def _build_static_user_prefix(self) -> str:
        """
        Static portion of the user message: the species list. Must come
        BEFORE any per-call variable content (the images) so it stays in
        the cacheable prefix.
        """
        species_lines = "\n".join(
            f"  <species key=\"{k}\">{v[0]}</species>"
            for k, v in sorted(self.species.items(), key=lambda kv: kv[0])
        )
        return (
            "<species_list>\n"
            "Candidate species. Return the integer key of the "
            "identified species, or 16 for segmentation error / "
            "multi-tree / unclassifiable data.\n"
            f"{species_lines}\n"
            "</species_list>\n"
            "\n"
            "Five depth maps follow (TOP, FRONT, BACK, LEFT, RIGHT). "
            "Apply the validation rules."
        )


    def api_call(self, images_base64: list[str]) -> int:
        assert len(images_base64) == len(self.VIEW_NAMES), (
            f"Expected {len(self.VIEW_NAMES)} views, got {len(images_base64)}"
        )

        content = [
            {"type": "input_text", "text": self._static_user_prefix},
        ]
        for name, b64 in zip(self.VIEW_NAMES, images_base64):
            content.append({"type": "input_text", "text": f"View: {name}"})
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{b64}",
                "detail": "low",  # must stay identical across calls
            })

        request_kwargs = dict(
            model=self.model,
            temperature=0,
            input=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": content},
            ],
            prompt_cache_key=self.PROMPT_CACHE_KEY,
        )
        if self.prompt_cache_retention is not None:
            request_kwargs["prompt_cache_retention"] = self.prompt_cache_retention

        response = self.client.responses.create(**request_kwargs)

        if hasattr(response, "usage") and response.usage is not None:
            self.prompt_tokens += response.usage.input_tokens or 0
            self.completion_tokens += response.usage.output_tokens or 0
            details = getattr(response.usage, "input_tokens_details", None)
            self.cached_tokens += getattr(details, "cached_tokens", 0) or 0

        raw = response.output_text.strip()

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
        depth_maps = self._cloud2images(points)
        images_b64 = self.tensors_to_base64(depth_maps)
        try:
            # raise ValueError("Test exception")
            key = self.api_call(images_b64)
        except Exception as e:
            print(f"API call failed: {e}")
            key = 16
        return key