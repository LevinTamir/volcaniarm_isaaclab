"""Apply PBR materials to the converted volcaniarm USD.

Pipeline order:
    scripts/convert_urdf.py     # urdf -> volcaniarm.usd (flat URDF colors)
    scripts/apply_materials.py  # this script: replace flat colors with PBR
    scripts/close_loop.py       # writes volcaniarm_closed.usd (sublayers base)

The closed USD sublayers `volcaniarm.usd`, so PBR materials authored here
flow through automatically. Re-running this script is idempotent: it
clobbers any material with the same name and re-binds.

Run with:
    conda activate isaaclab_env
    ~/isaac/IsaacLab/isaaclab.sh -p scripts/apply_materials.py
"""

from isaaclab.app import AppLauncher

_app = AppLauncher(headless=True).app

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

PROJECT = Path(__file__).resolve().parent.parent
BASE_USD = PROJECT / "assets/usd/volcaniarm.usd"


# Tweakable material recipes. Re-run the script after edits.
MATERIAL_GROUPS = {
    "pbr_galvanized": {
        # Galvanized steel profiles — slightly cool grey, fully metallic,
        # mid-rough so reflections diffuse rather than mirror.
        "base_color": (0.72, 0.74, 0.76),
        "metallic": 1.0,
        "roughness": 0.45,
        "links": [
            "volcaniarm_base_link",
            "volcaniarm_left_elbow_link",
            "volcaniarm_left_arm_link",
            "left_ee_link",
            "volcaniarm_right_elbow_link",
            "volcaniarm_right_arm_link",
            "right_ee_link",
            "camera_mount_linear_link",
        ],
    },
    "pbr_red_metallic": {
        # Powder-coated steel table — saturated red, glossier than the
        # arm because the coat is smoother than bare steel.
        "base_color": (0.62, 0.07, 0.09),
        "metallic": 1.0,
        "roughness": 0.35,
        "links": [
            "base_link",
            "fl_table_leg_link",
            "fr_table_leg_link",
            "rl_table_leg_link",
            "rr_table_leg_link",
        ],
    },
    "pbr_plastic_black": {
        # 3D-printed PLA mount — matte plastic, high roughness, no metal.
        "base_color": (0.04, 0.04, 0.04),
        "metallic": 0.0,
        "roughness": 0.85,
        "links": [
            "camera_mount_rev_link",
        ],
    },
}


def define_pbr_material(stage, looks_path, name, base_color, metallic, roughness):
    """Define (or replace) a UsdPreviewSurface material under `looks_path`."""
    mat_path = looks_path.AppendChild(name)
    if stage.GetPrimAtPath(mat_path).IsValid():
        stage.RemovePrim(mat_path)

    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path.AppendChild("PBRShader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*base_color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def find_link_prim(stage, link_name):
    """Find the first prim whose path-leaf-name matches `link_name`."""
    for prim in stage.Traverse():
        if prim.GetName() == link_name:
            return prim
    return None


def bind_material_recursive(link_prim, material):
    """Bind `material` to a link, overriding any descendant bindings.

    The actual mesh prims live in referenced configuration USDs, so a
    plain link-level Bind() gets shadowed by explicit per-mesh bindings
    authored upstream. Using `strongerThanDescendants` strength forces
    our binding to win without having to author into the referenced
    layers. We also bind on the link's `visuals` Xform child (the
    reference target) when present — that's where the geometry hangs.
    """
    binding_strength = UsdShade.Tokens.strongerThanDescendants

    UsdShade.MaterialBindingAPI.Apply(link_prim.GetPrim())
    binding_api = UsdShade.MaterialBindingAPI(link_prim)
    binding_api.Bind(material, binding_strength)

    bound_at = ["link"]
    for child in link_prim.GetChildren():
        if child.GetName() in {"visuals", "geometry"}:
            UsdShade.MaterialBindingAPI.Apply(child.GetPrim())
            UsdShade.MaterialBindingAPI(child).Bind(material, binding_strength)
            bound_at.append(child.GetName())
    return bound_at


def main():
    if not BASE_USD.exists():
        raise FileNotFoundError(f"Run convert_urdf.py first — missing {BASE_USD}")

    stage = Usd.Stage.Open(str(BASE_USD))
    root = stage.GetDefaultPrim()
    if not root:
        raise RuntimeError("Base USD has no default prim.")

    looks_path = root.GetPath().AppendChild("Looks")
    if not stage.GetPrimAtPath(looks_path).IsValid():
        UsdGeom.Scope.Define(stage, looks_path)

    print(f"Writing materials under: {looks_path}")
    materials = {}
    for name, spec in MATERIAL_GROUPS.items():
        materials[name] = define_pbr_material(
            stage, looks_path, name,
            spec["base_color"], spec["metallic"], spec["roughness"],
        )

    print()
    print("=== Binding ===")
    missing = []
    for name, spec in MATERIAL_GROUPS.items():
        for link_name in spec["links"]:
            link_prim = find_link_prim(stage, link_name)
            if link_prim is None:
                missing.append(link_name)
                print(f"  MISS  {link_name} (no such prim)")
                continue
            bound_at = bind_material_recursive(link_prim, materials[name])
            print(f"  BIND  {link_name:30s}  ->  {name}  (at: {','.join(bound_at)})")

    stage.GetRootLayer().Save()
    print()
    if missing:
        print(f"WARN — links not found: {missing}")
    print(f"Wrote: {BASE_USD}")


if __name__ == "__main__":
    import os

    main()
    # `_app.close()` hangs intermittently on Isaac Sim 5.1 shutdown
    # (background threads ignore SIGINT). Force-exit instead — the
    # stage was already saved by `main()` so nothing is at risk.
    os._exit(0)
