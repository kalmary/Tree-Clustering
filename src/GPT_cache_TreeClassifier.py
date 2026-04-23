import re
import torch
from torchvision import transforms
import base64
from io import BytesIO
from openai import OpenAI


class LLM_Classifier:
    # Fixed view order produced by _claude2images
    VIEW_NAMES = ["TOP", "FRONT", "BACK", "LEFT", "RIGHT"]

    # Stable key so OpenAI routes requests with the same prefix to the same
    # machine, maximising cache-hit rate. Bump the version suffix whenever
    # the static prefix (system prompt + species list) changes.
    PROMPT_CACHE_KEY = "tree-classifier-v5"

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

        # Build the static prefix once. It never changes between calls, so it
        # can live entirely in the cacheable prefix.
        self._system_prompt = self._build_system_prompt()
        self._static_user_prefix = self._build_static_user_prefix()

    # ------------------------------------------------------------------
    # Point cloud -> depth maps
    # ------------------------------------------------------------------
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

            # Normalise ONLY the valid (non-empty) cells. Using the `inf`
            # sentinel as the validity mask keeps empty cells distinct from
            # the farthest real depth until the very end, when we flush
            # empty cells to black. Brightness is inverted: closest point
            # to the camera -> near-white (1), farthest valid point ->
            # near-black (but strictly > 0), truly empty cell -> black (0).
            valid_mask = torch.isfinite(img)
            if torch.any(valid_mask):
                values = img[valid_mask]
                min_val = values.min()  # closest point
                max_val = values.max()  # farthest valid point
                normalised = (max_val - values) / (max_val - min_val + 1e-8)
                # Keep the darkest valid pixel slightly above 0 so it stays
                # visually distinguishable from truly empty background.
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
            "Each image is an 8-bit GRAYSCALE PNG produced by orthographic "
            "projection of the LiDAR point cloud onto one plane and a "
            "per-view min-max normalisation of depth. No colour, no axes, "
            "no labels, no statistics - just a grayscale image on a black "
            "background. The grayscale encoding is:\n"
            "  pure black (0)            : empty space (no LiDAR return)\n"
            "  very dark gray            : farthest returning points in\n"
            "                              that view (kept just above 0\n"
            "                              so they remain distinguishable\n"
            "                              from true empty background)\n"
            "  mid gray                  : mid-depth points\n"
            "  near-white (1)            : closest points to the camera\n"
            "\n"
            "In short: closer to the camera -> brighter; empty space is "
            "solid black.\n"
            "\n"
            "Because each view is normalised independently, the absolute "
            "brightness of a pixel is NOT comparable between views - only "
            "the spatial distribution of lit pixels and the relative "
            "brightness gradient within a single view carry information.\n"
            "\n"
            "Critical properties of the rendering that you MUST keep in "
            "mind:\n"
            "- These are NOT filled silhouettes. Projection happens at "
            "  pixel resolution so many interior cells stay black simply "
            "  because no LiDAR point landed there. Interior black "
            "  speckle inside the overall shape is usually just sampling "
            "  gaps, NOT real structural voids.\n"
            "- Brightness varies strongly across a single tree because "
            "  it encodes depth, not material: the side of the crown "
            "  facing the camera is light, the far side is dark, and the "
            "  interior of a thick crown may be completely hidden.\n"
            "- Point density varies too: crowns tend to look like dense "
            "  speckled regions, trunks like thin vertical streaks, and "
            "  occluded regions (upper trunk under a dense crown) may be "
            "  almost absent.\n"
            "- The tree's silhouette must be INFERRED from the outer "
            "  envelope of the lit pixels, not read off as a clean outline.\n"
            "- Individual lit pixels do not correspond to leaves or "
            "  branches - they are just surface hits. Do not try to count "
            "  branches or leaves.\n"
            "- You will not see any text, titles, stats, or colour in the "
            "  images. Do not invent any.\n"
            "\n"
            "What each view is best for:\n"
            "- TOP: crown footprint on the ground plane. Judge horizontal "
            "  extent, crown symmetry, and whether the crown is a single "
            "  compact region or fragmented into separate clumps. The "
            "  trunk often appears as a small dense spot near the centre.\n"
            "- FRONT and BACK: opposing side profiles (rotated 180 degrees). "
            "  Best for overall silhouette - total height, height-to-width "
            "  ratio, crown shape (columnar / conical / ovoid / rounded / "
            "  spreading / weeping / umbrella), trunk visibility, and the "
            "  height at which the lowest major branches emerge.\n"
            "- LEFT and RIGHT: the other pair of opposing side profiles, "
            "  perpendicular to FRONT/BACK. Cross-check against FRONT/BACK "
            "  to confirm features are genuinely structural rather than "
            "  artefacts of a single viewing angle.\n"
            "</input_format>\n"
            "\n"
            "<reasoning_guidelines>\n"
            "Base identification on geometric evidence only. You know the "
            "morphology of the candidate species; the prompt will NOT "
            "suggest any particular species. Describe what you see in "
            "shape terms, then match to whichever species in the list is "
            "most consistent with that shape.\n"
            "\n"
            "Useful geometric features, in rough order of diagnostic "
            "power:\n"
            "\n"
            "1. Overall crown shape, read from the four side views. "
            "   Categorise into shape families before naming a species: "
            "   conical with a clear apex, ovoid, broadly rounded, "
            "   columnar/narrow, spreading-with-short-trunk, "
            "   weeping/pendulous, umbrella/flat-topped, or "
            "   irregular/asymmetric. Cross-check the shape family "
            "   across FRONT, BACK, LEFT, RIGHT - if two perpendicular "
            "   pairs of views give clearly different shape families, "
            "   the shape is not reliable and you should lean toward 19 "
            "   (see validation rules).\n"
            "\n"
            "2. Height-to-width aspect ratio estimated from the side "
            "   views. Very tall-and-narrow, roughly balanced, or "
            "   wider-than-tall are three distinct regimes that "
            "   strongly constrain species identity.\n"
            "\n"
            "3. Trunk visibility and branching onset. Look at the lower "
            "   half of the side views: is there a long clean bare "
            "   trunk before the crown starts, does foliage begin "
            "   close to the ground, or does the trunk split into "
            "   multiple stems? These three regimes distinguish "
            "   forest-grown mature trees from open-grown ones and "
            "   from multi-stemmed forms.\n"
            "\n"
            "4. Crown density and internal structure. Compare how "
            "   densely packed the lit pixels are inside the crown "
            "   envelope. A uniformly dense speckled interior suggests "
            "   a well-foliated crown. A sparse, skeletal crown with "
            "   lots of internal empty space suggests an open or "
            "   leafless architecture.\n"
            "\n"
            "5. TOP-view footprint shape. Round, elongated, fragmented, "
            "   or off-centre relative to the trunk position.\n"
            "\n"
            "What to IGNORE:\n"
            "- Individual stray lit pixels clearly detached from the "
            "  main structure (noise, nearby vegetation, birds).\n"
            "- Thin horizontal bands of lit pixels at the very bottom "
            "  of side views - these are ground returns, not part of "
            "  the tree.\n"
            "- Isolated speckle in otherwise black regions.\n"
            "- Apparent bark pattern, leaf shape, or species-specific "
            "  colour - you cannot see any of these from a grayscale "
            "  depth projection. Pixel brightness encodes depth only.\n"
            "\n"
            "When views disagree on shape family, that is itself a "
            "signal: either the data is too poor to read reliably "
            "(lean toward 19) or multiple trees are present (also 19).\n"
            "</reasoning_guidelines>\n"
            "\n"
            "<validation_rules>\n"
            "Return 19 (segmentation error / unclassifiable) when ANY "
            "of the following is true. Err on the side of 19 when in "
            "doubt - a wrong species label is worse than a 19.\n"
            "\n"
            "A. Multiple-tree signal. The images show two or more "
            "   clearly separated tree structures. Concrete signs:\n"
            "   - TOP view shows two or more distinct dense regions "
            "     with clear empty space between them (not just one "
            "     irregular region with internal sampling gaps).\n"
            "   - Side views show two separate silhouettes side by "
            "     side with markedly different heights or crown "
            "     shapes that cannot be explained as one tree viewed "
            "     obliquely.\n"
            "   - A tall narrow form and a small low shrub-like clump "
            "     appear together but are disjoint in all side views.\n"
            "   A SINGLE tree with a lopsided, irregular, or gap-filled "
            "   crown is NOT a multi-tree case by itself.\n"
            "\n"
            "B. Insufficient structure. The point cloud does not show "
            "   a coherent tree envelope. Concrete signs:\n"
            "   - No discernible trunk in any side view (no vertical "
            "     streak of denser points, no clear main axis).\n"
            "   - The lit pixels form a flat mat, a scattered blob, or "
            "     vaguely horizontal vegetation with no clear vertical "
            "     extent relative to horizontal extent.\n"
            "   - The side views look like low shrubbery, understory "
            "     fragments, or ground debris rather than a tree.\n"
            "   - The height-to-width ratio is roughly 1 or less AND "
            "     the structure is not clearly a broad spreading "
            "     crown on a visible trunk.\n"
            "\n"
            "C. Shape inconsistency across views. The four side views "
            "   disagree so strongly about crown shape, height, or "
            "   outline that no single shape family fits. Genuine "
            "   trees look similar from FRONT vs BACK (mirror) and "
            "   broadly similar from LEFT vs RIGHT. If they do not, "
            "   the segmentation is probably flawed.\n"
            "\n"
            "D. Uncertainty rule. If you have identified a coherent "
            "   single tree but cannot confidently place it into one "
            "   specific shape family - i.e. you would be guessing "
            "   between three or more species with roughly equal "
            "   plausibility - return 19. Do NOT return 19 when you "
            "   can narrow it to one or two plausible species and "
            "   pick the better fit; only when you genuinely cannot "
            "   tell.\n"
            "</validation_rules>\n"
            "\n"
            "<output_contract>\n"
            "Output exactly one integer and nothing else: either the "
            "species key from the list, or 19 if any validation rule "
            "(A, B, C, or D) triggered. No explanation, no punctuation, "
            "no surrounding text, no code fences, no reasoning. Just "
            "the integer.\n"
            "</output_contract>"
        )

    def _build_static_user_prefix(self) -> str:
        """
        Static portion of the user message: the species list. Must come
        BEFORE any per-call variable content (the images) so it stays in
        the cacheable prefix.
        """
        species_lines = "\n".join(
            f"  <species key=\"{k}\">{v[1]}</species>"
            for k, v in sorted(self.species.items(), key=lambda kv: kv[0])
        )
        return (
            "<species_list>\n"
            "The following are the candidate species. Each line gives the "
            "integer key you must return if you identify that species. "
            "Key 19 means segmentation error, multi-tree, or "
            "unclassifiable data - use it whenever you cannot "
            "confidently identify a single tree of one specific species.\n"
            f"{species_lines}\n"
            "</species_list>\n"
            "\n"
            "Five depth maps of a single tree follow below, in the fixed "
            "order TOP, FRONT, BACK, LEFT, RIGHT. Apply the validation "
            "rules from the system prompt. When in doubt, return 19.\n"
            "Output a single integer only."
        )

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------
    def api_call(self, images_base64: list[str]) -> int:
        assert len(images_base64) == len(self.VIEW_NAMES), (
            f"Expected {len(self.VIEW_NAMES)} views, got {len(images_base64)}"
        )

        # IMPORTANT for caching: the user message begins with the fully
        # static species-list block, and ONLY AFTER that do we append the
        # per-call variable images. That way the prefix
        #   [system prompt] + [static user prefix]
        # is identical across every request and is cacheable.
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
            # Combined with the prefix hash to steer same-prefix traffic to
            # the same cache bucket. Keeping this stable across calls is
            # the single most effective thing you can do for hit rate.
            prompt_cache_key=self.PROMPT_CACHE_KEY,
        )
        if self.prompt_cache_retention is not None:
            # Accepted values: "in_memory" (default, ~5-10 min) or "24h"
            # (keeps cached prefixes alive up to 24 hours). Same per-token
            # price. If your SDK version rejects the kwarg outright,
            # construct the classifier with prompt_cache_retention=None.
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

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def cache_hit_rate(self) -> float:
        """Fraction of input tokens served from cache across all calls so far."""
        if self.prompt_tokens == 0:
            return 0.0
        return self.cached_tokens / self.prompt_tokens

    def predict(self, points) -> int:
        """
        Full pipeline: point cloud -> 5 depth maps -> base64 -> LLM -> species key.
        """
        points = torch.from_numpy(points)
        depth_maps = self._cloud2images(points)
        images_b64 = self.tensors_to_base64(depth_maps)
        key = self.api_call(images_b64)
        return key