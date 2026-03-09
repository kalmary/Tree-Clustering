import base64
import io
import re
import os
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Resolve project root (3 levels up from src/utils/this_file.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _view_to_png_bytes(channel: np.ndarray) -> bytes:
    ch = channel.astype(np.float32)
    lo, hi = ch.min(), ch.max()
    ch_norm = ((ch - lo) / (hi - lo + 1e-8) * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(ch_norm, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _parse_class_num(text: str) -> int:
    m = re.search(r"\b(\d+)\b", text)
    return int(m.group(1)) if m else 0


class TreeSpeciesClassifier:
    """
    Classify a 5-channel tree depth-map image via a vision LLM.

    Parameters
    ----------
    tree_species : dict[int, list[str]]
        Mapping of class index -> [latin_name, polish_name],
        e.g. {0: ["Betula_pendula", "Brzoza brodawkowata"], ...}
    api : "anthropic" | "google"
    view_names : list[str], optional
        Labels for each of the 5 channels. Defaults to Front/Back/Left/Right/Top.
    anthropic_api_key : str, optional  -- falls back to ANTHROPIC_API_KEY in .env / env
    google_api_key : str, optional     -- falls back to GOOGLE_API_KEY in .env / env
    """

    _PROMPT_TEMPLATE = (
        "You are an expert forest ecologist. "
        "You are given {n_views} grayscale depth-map images of a single tree, "
        "each showing a different side view: {view_names}. "
        "Based on the tree's overall shape, crown form, and branching structure "
        "visible across all views, identify the most likely tree species.\n\n"
        "Reply with ONLY the integer class number -- nothing else.\n\n"
        "Classes:\n{species_list}\n\n"
        "Reply with the single integer."
    )

    def __init__(
        self,
        tree_species: dict[int, list[str]],
        api: Literal["anthropic", "google"] = "anthropic",
        view_names: list[str] | None = None,
        anthropic_api_key: str | None = None,
        google_api_key: str | None = None,
    ):
        self.tree_species = tree_species
        self.api = api
        self.view_names = view_names or ["Front", "Back", "Left", "Right", "Top"]
        self._anthropic_key = anthropic_api_key
        self._google_key = google_api_key

    def classify(self, img: np.ndarray) -> tuple[int, str, str]:
        """
        Parameters
        ----------
        img : np.ndarray  shape (H, W, 5)
            Each channel is a depth-map side view of the tree.

        Returns
        -------
        (class_num, latin_name, polish_name)  e.g. (16, "Pinus_sylvestris", "Sosna zwyczajna")
        """
        if img.ndim != 3 or img.shape[2] != len(self.view_names):
            raise ValueError(
                f"Expected (H, W, {len(self.view_names)}) array, got {img.shape}"
            )

        views = [(name, _view_to_png_bytes(img[:, :, i]))
                 for i, name in enumerate(self.view_names)]

        species_list = "\n".join(
            f"  {k}: {v[0]} ({v[1]})" for k, v in self.tree_species.items()
        )
        prompt = self._PROMPT_TEMPLATE.format(
            n_views=len(views),
            view_names=", ".join(self.view_names),
            species_list=species_list,
        )

        if self.api == "anthropic":
            raw = self._call_anthropic(views, prompt)
        elif self.api == "google":
            raw = self._call_google(views, prompt)
        else:
            raise ValueError(f"Unknown api={self.api!r}. Use 'anthropic' or 'google'.")

        class_num = _parse_class_num(raw)
        names = self.tree_species.get(class_num, ["Unknown", "Nieznany"])
        return class_num, names[0], names[1]

    def _call_anthropic(self, views: list[tuple[str, bytes]], prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic(
            api_key=self._anthropic_key or os.environ["ANTHROPIC_API_KEY"]
        )
        content = []
        for name, png_bytes in views:
            content.append({"type": "text", "text": f"View: {name}"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(png_bytes).decode(),
                },
            })
        content.append({"type": "text", "text": prompt})

        msg = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=16,
            messages=[{"role": "user", "content": content}],
        )
        return msg.content[0].text.strip()

    def _call_google(self, views: list[tuple[str, bytes]], prompt: str) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._google_key or os.environ["GOOGLE_API_KEY"])

        content = []
        for name, png_bytes in views:
            content.append(f"View: {name}")
            content.append(types.Part.from_bytes(data=png_bytes, mime_type="image/png"))
        content.append(prompt)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=content,
        )
        return response.text.strip()


if __name__ == "__main__":
    print(f"[debug] project root : {_PROJECT_ROOT}")
    print(f"[debug] .env path    : {_PROJECT_ROOT / '.env'}")
    print(f"[debug] API key      : {repr(os.environ.get('ANTHROPIC_API_KEY'))}")

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

    clf = TreeSpeciesClassifier(tree_species=species, api="google")

    rng = np.random.default_rng(42)
    fake_img = rng.uniform(0, 1, (64, 64, 5)).astype(np.float32)
    num, latin, polish = clf.classify(fake_img)
    print(f"Predicted class: {num} → {latin} / {polish}")