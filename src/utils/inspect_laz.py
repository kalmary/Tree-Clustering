# pyright: basic

"""
Goal: Inspect a LAZ point cloud file to determine if it contains the necessary fields
to compute or extract LiDAR ray vectors (the vector from the sensor origin to the point).

What we look for:
1. Pre-computed ray or sensor fields (e.g., 'SensorX', 'RayDirX'). If these exist, 
   rays can be extracted directly.
2. Standard fields for trajectory estimation:
   - 'gps_time': To group points chronologically.
   - 'point_source_id': To separate different flight lines.
   - 'scan_angle_rank' or 'scan_angle': To estimate sensor altitude and cross-track distance.

Having these standard fields allows us to approximate the scanner's trajectory
and simulate proper rays without needing an external trajectory file.
"""

import logging
from pathlib import Path

import laspy

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)

def inspect_laz_for_rays(file_path: Path):
    logger.info(f"Starting LAZ inspection for file: {file_path}")
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return

    try:
        logger.info("Opening LAZ file and reading header...")
        with laspy.open(file_path) as f:
            header = f.header
            point_format = header.point_format
            dim_names = [dim.name for dim in point_format.dimensions]
            
            logger.info(f"File successfully opened. Total points: {header.point_count:,}")
            logger.info(f"Available dimensions: {', '.join(dim_names)}")
            
            logger.info("Analyzing dimensions for ray vector requirements...")
            
            # 1. Pre-computed rays/sensor positions
            sensor_dims = [d for d in dim_names if "sensor" in d.lower() or "ray" in d.lower() or "origin" in d.lower()]
            if sensor_dims:
                logger.info(f"Found pre-computed sensor/ray dimensions: {sensor_dims}")
                logger.info("Verdict: Rays can be directly extracted without simulation.")
            else:
                logger.info("No pre-computed sensor/ray dimensions found (e.g., SensorX).")
                
            # 2. Reconstructing from trajectory
            has_gps_time = "gps_time" in dim_names
            has_pt_src_id = "point_source_id" in dim_names
            has_scan_angle = "scan_angle_rank" in dim_names or "scan_angle" in dim_names
            
            logger.info("Checking standard fields for ray estimation:")
            logger.info(f" - gps_time: {'Present' if has_gps_time else 'Missing'}")
            logger.info(f" - point_source_id: {'Present' if has_pt_src_id else 'Missing'}")
            logger.info(f" - scan_angle_rank: {'Present' if has_scan_angle else 'Missing'}")
            
            if has_gps_time and has_pt_src_id:
                logger.info("Verdict: The file has 'gps_time' and 'point_source_id'.")
                if has_scan_angle:
                    logger.info("-> You can approximate the flight trajectory and compute proper rays.")
                else:
                    logger.info("-> You can compute precise rays if you provide an external trajectory (SBET) file.")
            elif not has_gps_time:
                logger.warning("Verdict: Missing 'gps_time'.")
                logger.warning("-> It is extremely difficult to compute proper rays without gps_time.")
                
            logger.info("Inspection finished.")
                
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    # Hardcoded path to the fixture file
    LAZ_FILE_PATH = Path("/Users/michalsiniarski/Documents/PROGRAMMING/Tree-Clustering/fixtures/Grajewo_2026_6_1_mod.laz")
    inspect_laz_for_rays(LAZ_FILE_PATH)
