#!/usr/bin/env python3
"""Audit Lance fingertip endpoints and object meshes against state54 constants."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import lance
import numpy as np

from scripts.eval import mano_action_support
from scripts.mano_state54_contract import (
    FINGERTIP_JOINT_INDICES,
    FINGERTIP_LOCAL_OFFSETS,
    ManoFingertipFK,
    OBJECT_COLLISION_BOXES,
)


def mesh_path(urdf: Path) -> tuple[Path, np.ndarray]:
    collision = ET.parse(urdf).getroot().find("link/collision")
    mesh = collision.find("geometry/mesh")
    filename = mesh.attrib["filename"]
    scale = np.asarray([float(x) for x in mesh.attrib.get("scale", "1 1 1").split()])
    return (urdf.parent / filename).resolve(), scale


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    ds=lance.dataset(str(args.dataset));fk=ManoFingertipFK()
    rows=np.linspace(507,2503,24,dtype=int);errors=[]
    for row_index in rows:
        hand=ds.take([int(row_index)],columns=["hands"]).to_pylist()[0]["hands"][0]
        q=np.asarray(hand["urdf_dof"]);j=np.asarray(hand["mano_joint_pos"])
        for frame in np.linspace(0,len(q)-1,11,dtype=int):
            errors.append(np.linalg.norm(fk(q[frame])-j[frame,FINGERTIP_JOINT_INDICES],axis=1))
    errors=np.asarray(errors)
    objects={}
    for name,contract in OBJECT_COLLISION_BOXES.items():
        path,scale=mesh_path(mano_action_support.DEFAULT_OBJECTS_URDF_DIR/f"{name}.urdf")
        vertices = np.asarray(
            [
                [float(value) for value in line.split()[1:4]]
                for line in path.read_text(encoding="utf-8", errors="strict").splitlines()
                if line.startswith("v ")
            ],
            dtype=np.float64,
        ) * scale
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
            raise ValueError(f"OBJ has no parseable vertices: {path}")
        lo,hi=vertices.min(axis=0),vertices.max(axis=0)
        half=(hi-lo)/2;center=(hi+lo)/2
        objects[name]={
            "mesh":str(path),"derived_half_extents":half.tolist(),"contract_half_extents":contract.half_extents.tolist(),
            "derived_local_center":center.tolist(),"contract_local_center":contract.local_center.tolist(),
            "max_abs_half_extent_error_m":float(np.max(np.abs(half-contract.half_extents))),
            "max_abs_center_error_m":float(np.max(np.abs(center-contract.local_center))),
        }
        if not np.allclose(half,contract.half_extents,rtol=0,atol=2e-9) or not np.allclose(center,contract.local_center,rtol=0,atol=2e-9):
            raise ValueError(f"object mesh contract drift: {name}")
    result={
        "sampled_rows":rows.tolist(),"sampled_poses":int(len(errors)),"finger_order":["index","thumb","ring","middle","pinky"],
        "fingertip_joint_indices":FINGERTIP_JOINT_INDICES.tolist(),"fingertip_local_offsets":FINGERTIP_LOCAL_OFFSETS.tolist(),
        "per_finger_max_error_mm":(errors.max(axis=0)*1000).tolist(),"overall_max_error_mm":float(errors.max()*1000),
        "per_finger_mean_error_mm":(errors.mean(axis=0)*1000).tolist(),"object_mesh_aabb":objects,
    }
    if errors.max()>=3e-5: raise ValueError(f"FK endpoint error {errors.max()}")
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
