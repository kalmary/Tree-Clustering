import laspy
import numpy as np
import pathlib as pth
import torch
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

from utils.visualize_trees import save_tree_projections_pdf
from array_processing_RE import TreeSegmRay

from TreePCDClass.src.TreeClassifier import TreeClassifier

from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Label translation  (old model  →  new SPECIES dict)
# ──────────────────────────────────────────────────────────────────────────────

# Old model classes ordered alphabetically as LabelEncoder produces them,
# with "others" appended last (index = n_species).  The old model has no
# "Incorrect segmentation" concept — that label (key 19 in SPECIES) is
# reserved exclusively for manual annotation by a specialist.
#
# Update this list if you retrain the old model with different species.
OLD_MODEL_CLASSES: list[str] = [
    "others",               # 0  ← explicitly assigned 0 during training
    "Betula_pendula",       # 1
    "Fagus_sylvatica",      # 2
    "Quercus_petraea",      # 3
    "Quercus_rubra",        # 4
    "Quercus_robur",        # 5
    "Carpinus_betulus",     # 6
    "Fraxinus_excelsior",   # 7
    "Acer_pseudoplatanus",  # 8
    "Acer_campestre",       # 9
    "Tilia_cordata",        # 10
    "Ulmus_laevis",         # 11
    "Crataegus_monogyna",   # 12
    "Corylus_avellana",     # 13
    "Pseudotsuga_menziesii",# 14
    "Abies_alba",           # 15
    "Larix_decidua",        # 16
    "Pinus_sylvestris",     # 17
    "Picea_abies",          # 18
]


def _build_translator(species_dict: dict) -> dict:
    """Build a lookup from any raw old-model label to a SPECIES latin name.

    The raw label coming out of TreeClassifier.predict() may be an int (the
    LabelEncoder numeric index) or a string (the decoded class name).  Both
    are handled: the returned dict has both int and str keys for every entry.

    Key 19 ("Incorrect segmentation") is never a target — only a specialist
    can assign it.
    """
    latin_to_new: dict[str, int] = {
        v[0].lower(): k
        for k, v in species_dict.items()
        if k != 19
    }

    translator: dict = {}
    n = len(OLD_MODEL_CLASSES)

    for idx, old_name in enumerate(OLD_MODEL_CLASSES):
        if old_name.lower() == "others":
            new_latin = species_dict[18][0]  # → "Other"
        else:
            normalised = old_name.lower().replace(" ", "_")
            new_key = latin_to_new.get(normalised)
            if new_key is None:
                # substring fallback
                new_key = next(
                    (k for lat, k in latin_to_new.items()
                     if normalised in lat or lat in normalised),
                    18,  # unknown → Other
                )
            new_latin = species_dict[new_key][0]

        # register both int index and string name as keys
        translator[idx]      = new_latin
        translator[old_name] = new_latin

    return translator


def translate_predictions(
    raw_predictions: dict[int, object],
    translator: dict,
) -> dict[int, str]:
    """Map raw old-model labels (int or str) to new SPECIES latin names.

    Falls back to str(raw) for any label not found in the translator —
    this covers unexpected model outputs without crashing.
    """
    return {
        tree_id: translator.get(raw, str(raw))
        for tree_id, raw in raw_predictions.items()
    }


# ──────────────────────────────────────────────────────────────────────────────
# Excel helpers
# ──────────────────────────────────────────────────────────────────────────────

def _header_cell(cell, text: str):
    cell.value     = text
    cell.font      = Font(name="Arial", bold=True, color="FFFFFF")
    cell.fill      = PatternFill("solid", start_color="2E7D32")
    cell.alignment = Alignment(horizontal="center")


def create_workbook(species_dict: dict) -> openpyxl.Workbook:
    wb = openpyxl.Workbook()

    # ── Sheet 1: Tree Labels ──────────────────────────────────────────────────
    ws_labels = wb.active
    ws_labels.title = "Tree Labels"

    _header_cell(ws_labels.cell(row=1, column=1), "File Name")
    _header_cell(ws_labels.cell(row=1, column=2), "Tree Label (auto)")
    _header_cell(ws_labels.cell(row=1, column=3), "Tree Label (verified)")

    ws_labels.freeze_panes = "A2"
    ws_labels.column_dimensions["A"].width = 30
    ws_labels.column_dimensions["B"].width = 26
    ws_labels.column_dimensions["C"].width = 26

    # ── Sheet 2: Species lookup ───────────────────────────────────────────────
    ws_species = wb.create_sheet("Available labels lookup")

    for col, h in enumerate(["Species Code", "Species (Latin)", "Species (Polish)"], 1):
        _header_cell(ws_species.cell(row=1, column=col), h)

    for row_idx, (code, names) in enumerate(species_dict.items(), 2):
        ws_species.cell(row=row_idx, column=1, value=code).font    = Font(name="Arial")
        ws_species.cell(row=row_idx, column=2, value=names[0]).font = Font(name="Arial")
        ws_species.cell(row=row_idx, column=3, value=names[1]).font = Font(name="Arial")

    ws_species.freeze_panes = "A2"
    ws_species.column_dimensions["A"].width = 14
    ws_species.column_dimensions["B"].width = 28
    ws_species.column_dimensions["C"].width = 28

    return wb


def _load_or_create_workbook(
    xlsx_path: pth.Path,
    species_dict: dict,
) -> tuple[openpyxl.Workbook, int]:
    """Return (workbook, next_data_row) for the Tree Labels sheet.

    Loads the existing file when it already exists so that rows can be
    appended without destroying previously written data.
    """
    if xlsx_path.exists():
        wb = openpyxl.load_workbook(xlsx_path)
        ws = wb["Tree Labels"]
        # Find first truly empty row after the header
        next_row = ws.max_row + 1
        # Guard against trailing blank rows reported by openpyxl
        while next_row > 2 and ws.cell(row=next_row - 1, column=1).value is None:
            next_row -= 1
        return wb, next_row
    else:
        wb = create_workbook(species_dict)
        return wb, 2  # row 1 is the header


def append_tree_rows(
    ws,
    las_idx: int,
    tree_labels: np.ndarray,
    predictions: dict,          # {tree_id: predicted_species_str}  – may be empty
    data_row_start: int,
) -> int:
    """Write one row per tree to *ws* starting at *data_row_start*.

    Column A  – file name  (black)
    Column B  – auto-predicted label  (red = not yet verified by specialist)
    Column C  – verified label  (intentionally left blank for manual entry)

    Returns the next available row index.
    """
    red_font   = Font(name="Arial", color="FF0000")
    black_font = Font(name="Arial")

    for tree_id in sorted(set(tree_labels[tree_labels != -1].tolist())):
        file_name = f"{las_idx}_{tree_id}.npy"
        predicted = predictions.get(int(tree_id), "")

        ws.cell(row=data_row_start, column=1, value=file_name).font = black_font
        cell_pred = ws.cell(row=data_row_start, column=2, value=predicted)
        cell_pred.font = red_font
        # Column C left blank – specialist fills it in
        data_row_start += 1

    return data_row_start


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main(species_dict: dict = None):
    semantic_labelled_dir = pth.Path("/mnt/DATA_SSD/BRIK/GRAJEWO_STARE")
    laz_paths = list(semantic_labelled_dir.glob("*.laz"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg = TreeSegmRay(height_min=0.6, max_diameter=0.6, distance_limit=0.25, gravity_factor=0.85, use_rays=False, ground_label=1, tree_label=7, verbose = True)

    config_dir = pth.Path(__file__).parent.joinpath("TreePCDClass/src/final_files")
    tree_class_model = TreeClassifier(
        model_name="ResNetTreeV0_61",
        config_dir=config_dir,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

    # Build label translator: old-model raw labels (int or str) → new SPECIES names.
    # OLD_MODEL_CLASSES is hardcoded above — update it if the old model is retrained.
    label_translator = _build_translator(species_dict) if species_dict is not None else {}
    # if label_translator:
    #     print(f"[LABELS] Translator ready ({len(OLD_MODEL_CLASSES)} old-model classes, "
    #           f"'others' at index 0 → 'Other'.")

    seg.start_container()

    pbar = tqdm(enumerate(laz_paths), desc = "Processing files", total = len(laz_paths))

    for las_idx, path in pbar:

        dataset_dir = path.parent / path.stem
        dataset_dir.mkdir(exist_ok=True, parents=True)

        xlsx_path = dataset_dir / f"{path.stem}.xlsx"
        pdf_path  = dataset_dir / f"{path.stem}.pdf"
        pdf_name  = f"{path.stem}.pdf"

        # Remove any partial outputs from a previous crashed run so we never
        # append onto an inconsistent file.
        for stale in (xlsx_path, pdf_path):
            if stale.exists():
                stale.unlink()
                # print(f"[CLEANUP] Removed stale {stale.name}")

        try:
            las = laspy.read(path)
            pts = np.vstack([las.x, las.y, las.z]).T
            cls = np.asarray(las.classification, dtype=np.int32)

            tree_labels = seg.segment(pts, cls)
            tree_xyz    = pts[cls == seg.tree_label]

            valid_mask = tree_labels != -1
            valid_pts  = tree_xyz[valid_mask]
            valid_lbls = tree_labels[valid_mask]

            # Returns {tree_id: raw_prediction_str}; appends pages if PDF exists.
            raw_predictions = save_tree_projections_pdf(
                points=valid_pts,
                labels=valid_lbls,
                las_idx=las_idx,
                source_name=path.stem,
                output_name=pdf_name,
                output_dir=dataset_dir,
                resolution=350,
                device=device,
                tree_classifier=tree_class_model if species_dict is not None else None,
            )

            # Remap old-model class labels to new SPECIES dict names.
            # "Incorrect segmentation" (key 19) is never emitted by the old model
            # and is left blank — the specialist decides which trees get it.
            predictions = (
                translate_predictions(raw_predictions, label_translator)
                if label_translator else raw_predictions
            )

            if species_dict is not None:
                wb, next_row = _load_or_create_workbook(xlsx_path, species_dict)
                append_tree_rows(
                    wb["Tree Labels"],
                    las_idx=las_idx,
                    tree_labels=valid_lbls,
                    predictions=predictions,
                    data_row_start=next_row,
                )
                wb.save(xlsx_path)

        finally:
            seg.rm_container()


if __name__ == "__main__":
    SPECIES = {
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

    main(species_dict=SPECIES)