"""Robust STEP Boolean operations for complex lofted B-spline solids.

Problem: OCCT's standard BRepAlgoAPI_Fuse/Cut fail silently on complex
ThruSections lofted B-spline surfaces (topology corruption, volume inversion,
non-manifold edges).

Solution: A multi-stage pipeline that pre-conditions geometry before booleans:

  1. ShapeFix   — heal tolerances, degenerated edges, small faces
  2. Segmentation — split complex B-spline surfaces into simpler C1 patches
     via ShapeUpgrade_ShapeDivide (reduces knot multiplicity issues)
  3. UnifySameDomain — merge co-planar faces to simplify topology
  4. BOPAlgo_CellsBuilder — the most robust OCCT boolean engine:
     performs a General Fuse, splits all shapes into non-overlapping cells,
     then lets us select which cells to keep (union/difference/intersection)
  5. Post-healing — fix tolerances on the result
  6. Cascade fallback — if CellsBuilder fails, try BRepAlgoAPI with
     increasing fuzzy tolerance, then Splitter-based approach

This preserves full NURBS quality (no tessellation to mesh).
"""

from __future__ import annotations
import warnings
from typing import Optional

from OCP.TopoDS import TopoDS_Shape, TopoDS_Solid
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.BRepCheck import BRepCheck_Analyzer


# ─── Geometry healing ─────────────────────────────────────────────────────

def heal_shape(shape: TopoDS_Shape, tolerance: float = 0.01) -> TopoDS_Shape:
    """Apply ShapeFix healing pipeline to a shape.

    Fixes:
    - Edge/vertex tolerances
    - Degenerated edges
    - Small/missing faces
    - Wire orientation
    - Shell closure
    """
    from OCP.ShapeFix import ShapeFix_Shape, ShapeFix_Solid

    fixer = ShapeFix_Shape(shape)
    fixer.SetPrecision(tolerance)
    fixer.SetMaxTolerance(tolerance * 10)
    fixer.SetMinTolerance(tolerance / 10)

    # Enable all fix modes
    fixer.FixWireTool().FixRemovePCurveMode = 1
    fixer.FixFaceTool().FixOrientationMode = 1

    fixer.Perform()
    result = fixer.Shape()

    # Additional solid-specific healing
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    exp = TopExp_Explorer(result, TopAbs_SOLID)
    if exp.More():
        from OCP.TopoDS import TopoDS
        solid = TopoDS.Solid_s(exp.Current())
        sf = ShapeFix_Solid(solid)
        sf.SetPrecision(tolerance)
        sf.Perform()
        return sf.Solid()

    return result


def segment_bspline_surfaces(shape: TopoDS_Shape,
                              max_degree: int = 6,
                              max_segments: int = 12) -> TopoDS_Shape:
    """Split complex B-spline surfaces into simpler patches.

    High-degree B-spline surfaces with many knots cause boolean failures.
    ShapeUpgrade_ShapeDivide splits them into lower-degree segments
    with cleaner knot vectors, making boolean intersections tractable.
    """
    from OCP.ShapeUpgrade import ShapeUpgrade_ShapeDivide

    divider = ShapeUpgrade_ShapeDivide(shape)
    divider.SetPrecision(0.001)  # 1 micron
    divider.SetMaxTolerance(0.01)

    # Configure surface splitting
    split_tool = divider.GetSplitSurfaceTool()
    if split_tool is not None:
        split_tool.SetUSplitValues(None)  # auto
        split_tool.SetVSplitValues(None)

    divider.Perform()

    if divider.Status(0):  # ShapeExtend_OK
        return divider.Result()

    # If ShapeDivide can't help, try ShapeUpgrade_UnifySameDomain
    return _unify_domain(shape)


def _unify_domain(shape: TopoDS_Shape) -> TopoDS_Shape:
    """Merge co-planar / co-cylindrical faces sharing the same geometry.

    Reduces face count and simplifies topology before booleans.
    """
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

    unifier = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
    unifier.SetAngularTolerance(0.01)  # ~0.5 degrees
    unifier.SetLinearTolerance(0.001)  # 1 micron
    unifier.Build()
    return unifier.Shape()


def preprocess(shape: TopoDS_Shape, tolerance: float = 0.01) -> TopoDS_Shape:
    """Full preprocessing pipeline: heal → unify faces.

    Note: we skip segment_bspline_surfaces by default because
    ShapeUpgrade_ShapeDivide can sometimes introduce new issues.
    UnifySameDomain is always safe and beneficial.
    """
    healed = heal_shape(shape, tolerance)
    unified = _unify_domain(healed)
    return unified


# ─── Volume validation ────────────────────────────────────────────────────

def compute_volume(shape: TopoDS_Shape) -> float:
    """Return volume in mm³ (absolute value)."""
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return abs(props.Mass())


def is_valid_solid(shape: TopoDS_Shape) -> bool:
    """Check topology validity."""
    return BRepCheck_Analyzer(shape).IsValid()


def validate_boolean_result(
    result: TopoDS_Shape,
    operand_volumes: list[float],
    operation: str,
    volume_tolerance: float = 0.10,
) -> tuple[bool, str]:
    """Validate a boolean result by checking topology and volume sanity.

    Args:
        result: The boolean output shape
        operand_volumes: Volumes of input operands [vol_a, vol_b]
        operation: "fuse" or "cut"
        volume_tolerance: Max acceptable relative deviation

    Returns:
        (is_ok, message)
    """
    if not is_valid_solid(result):
        return False, "Result has invalid topology"

    vol_result = compute_volume(result)
    vol_a, vol_b = operand_volumes

    if vol_result < 1.0:
        return False, "Result volume is near-zero (%.2f mm³)" % vol_result

    if operation == "fuse":
        # Fused volume should be ≤ sum (overlap reduces it)
        # but definitely > max(vol_a, vol_b)
        vol_max = max(vol_a, vol_b)
        vol_sum = vol_a + vol_b
        if vol_result < vol_max * (1 - volume_tolerance):
            return False, (
                "Fuse volume %.1f < max operand %.1f (volume loss)"
                % (vol_result, vol_max)
            )
        if vol_result > vol_sum * (1 + volume_tolerance):
            return False, (
                "Fuse volume %.1f > sum %.1f (volume inflation)"
                % (vol_result, vol_sum)
            )

    elif operation == "cut":
        # Cut result should be ≤ vol_a (we remove material)
        if vol_result > vol_a * (1 + volume_tolerance):
            return False, (
                "Cut volume %.1f > base %.1f (volume inflation)"
                % (vol_result, vol_a)
            )
        if vol_result < vol_a * 0.01:
            return False, (
                "Cut volume %.1f is < 1%% of base %.1f (near-total removal)"
                % (vol_result, vol_a)
            )

    return True, "OK (vol=%.1f mm³)" % vol_result


# ─── Boolean engines ──────────────────────────────────────────────────────

def _cells_builder_fuse(
    shape_a: TopoDS_Shape,
    shape_b: TopoDS_Shape,
    fuzzy: float = 0.01,
) -> Optional[TopoDS_Shape]:
    """Fuse via BOPAlgo_CellsBuilder (most robust OCCT boolean).

    CellsBuilder performs a General Fuse that decomposes all input shapes
    into non-overlapping cells. We then select ALL cells to get the union.
    This handles cases where BRepAlgoAPI_Fuse fails.
    """
    from OCP.BOPAlgo import BOPAlgo_CellsBuilder
    from OCP.TopTools import TopTools_ListOfShape

    cb = BOPAlgo_CellsBuilder()
    cb.AddArgument(shape_a)
    cb.AddArgument(shape_b)
    cb.SetFuzzyValue(fuzzy)
    cb.SetRunParallel(True)
    cb.SetNonDestructive(True)

    # Perform the General Fuse (split into cells)
    cb.Perform()
    if cb.HasErrors():
        return None

    # Select ALL cells → union
    # AddAllToResult() adds every cell, achieving a fuse
    cb.AddAllToResult()
    cb.MakeContainers()

    result = cb.Shape()
    return result if not result.IsNull() else None


def _cells_builder_cut(
    shape_base: TopoDS_Shape,
    shape_tool: TopoDS_Shape,
    fuzzy: float = 0.01,
) -> Optional[TopoDS_Shape]:
    """Cut (difference) via BOPAlgo_CellsBuilder.

    General Fuse → select only cells that belong to shape_base
    but NOT to shape_tool.
    """
    from OCP.BOPAlgo import BOPAlgo_CellsBuilder
    from OCP.TopTools import TopTools_ListOfShape

    cb = BOPAlgo_CellsBuilder()
    cb.AddArgument(shape_base)
    cb.AddArgument(shape_tool)
    cb.SetFuzzyValue(fuzzy)
    cb.SetRunParallel(True)
    cb.SetNonDestructive(True)

    cb.Perform()
    if cb.HasErrors():
        return None

    # Build material lists:
    # To select cells IN base but NOT IN tool:
    #   take_material = [shape_base]
    #   avoid_material = [shape_tool]
    take = TopTools_ListOfShape()
    take.Append(shape_base)

    avoid = TopTools_ListOfShape()
    avoid.Append(shape_tool)

    cb.AddToResult(take, avoid)
    cb.MakeContainers()

    result = cb.Shape()
    return result if not result.IsNull() else None


def _standard_fuse(
    shape_a: TopoDS_Shape,
    shape_b: TopoDS_Shape,
    fuzzy: float = 0.01,
) -> Optional[TopoDS_Shape]:
    """Fuse via standard BRepAlgoAPI_Fuse with fuzzy tolerance."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse

    op = BRepAlgoAPI_Fuse(shape_a, shape_b)
    op.SetFuzzyValue(fuzzy)
    op.Build()
    if op.IsDone():
        return op.Shape()
    return None


def _standard_cut(
    shape_base: TopoDS_Shape,
    shape_tool: TopoDS_Shape,
    fuzzy: float = 0.01,
) -> Optional[TopoDS_Shape]:
    """Cut via standard BRepAlgoAPI_Cut with fuzzy tolerance."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut

    op = BRepAlgoAPI_Cut(shape_base, shape_tool)
    op.SetFuzzyValue(fuzzy)
    op.Build()
    if op.IsDone():
        return op.Shape()
    return None


def _splitter_cut(
    shape_base: TopoDS_Shape,
    shape_tool: TopoDS_Shape,
    fuzzy: float = 0.01,
) -> Optional[TopoDS_Shape]:
    """Cut via BRepAlgoAPI_Splitter + cell selection.

    The Splitter splits shapes without performing boolean logic.
    We then iterate the result and keep only fragments that belong
    to the base shape but not the tool.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter
    from OCP.TopTools import TopTools_ListOfShape
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopoDS import TopoDS, TopoDS_Compound
    from OCP.BRep import BRep_Builder
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.gp import gp_Pnt

    args = TopTools_ListOfShape()
    args.Append(shape_base)

    tools = TopTools_ListOfShape()
    tools.Append(shape_tool)

    splitter = BRepAlgoAPI_Splitter()
    splitter.SetArguments(args)
    splitter.SetTools(tools)
    splitter.SetFuzzyValue(fuzzy)
    splitter.Build()

    if not splitter.IsDone():
        return None

    # Collect solid fragments that are NOT inside the tool
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    n_kept = 0

    exp = TopExp_Explorer(splitter.Shape(), TopAbs_SOLID)
    while exp.More():
        solid = TopoDS.Solid_s(exp.Current())

        # Test if solid's center-of-mass is inside the tool
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(solid, props)
        center = props.CentreOfMass()

        classifier = BRepClass3d_SolidClassifier(shape_tool, center, 0.001)
        from OCP.TopAbs import TopAbs_IN
        if classifier.State() != TopAbs_IN:
            builder.Add(compound, solid)
            n_kept += 1

        exp.Next()

    if n_kept == 0:
        return None

    return compound


# ─── Main API: cascade boolean with fallbacks ─────────────────────────────

FUZZY_VALUES = [0.01, 0.05, 0.1, 0.5]  # increasing tolerance cascade


def robust_fuse(
    shape_a: TopoDS_Shape,
    shape_b: TopoDS_Shape,
    preprocess_shapes: bool = True,
    label: str = "fuse",
) -> tuple[Optional[TopoDS_Shape], str]:
    """Robust fuse with preprocessing, validation, and cascading fallbacks.

    Strategy:
      1. Preprocess both shapes (heal + unify)
      2. Try BOPAlgo_CellsBuilder at increasing fuzzy values
      3. Fall back to BRepAlgoAPI_Fuse at increasing fuzzy values
      4. Validate each result (topology + volume sanity)

    Returns:
        (result_shape, status_message)
        result_shape is None if all strategies failed.
    """
    if preprocess_shapes:
        print("[%s] Preprocessing operands..." % label)
        shape_a = preprocess(shape_a)
        shape_b = preprocess(shape_b)

    vol_a = compute_volume(shape_a)
    vol_b = compute_volume(shape_b)
    print("[%s] Operand volumes: %.1f, %.1f mm³" % (label, vol_a, vol_b))

    # Strategy 1: CellsBuilder (most robust)
    for fuzzy in FUZZY_VALUES:
        print("[%s] Trying CellsBuilder (fuzzy=%.3f)..." % (label, fuzzy))
        try:
            result = _cells_builder_fuse(shape_a, shape_b, fuzzy)
            if result is not None:
                ok, msg = validate_boolean_result(
                    result, [vol_a, vol_b], "fuse"
                )
                if ok:
                    result = heal_shape(result)
                    print("[%s] CellsBuilder OK: %s" % (label, msg))
                    return result, "CellsBuilder fuzzy=%.3f" % fuzzy
                else:
                    print("[%s] CellsBuilder result invalid: %s" % (label, msg))
        except Exception as e:
            print("[%s] CellsBuilder error: %s" % (label, e))

    # Strategy 2: Standard BRepAlgoAPI_Fuse
    for fuzzy in FUZZY_VALUES:
        print("[%s] Trying BRepAlgoAPI_Fuse (fuzzy=%.3f)..." % (label, fuzzy))
        try:
            result = _standard_fuse(shape_a, shape_b, fuzzy)
            if result is not None:
                ok, msg = validate_boolean_result(
                    result, [vol_a, vol_b], "fuse"
                )
                if ok:
                    result = heal_shape(result)
                    print("[%s] Standard Fuse OK: %s" % (label, msg))
                    return result, "BRepAlgoAPI_Fuse fuzzy=%.3f" % fuzzy
                else:
                    print("[%s] Standard Fuse result invalid: %s" % (label, msg))
        except Exception as e:
            print("[%s] Standard Fuse error: %s" % (label, e))

    return None, "all fuse strategies failed"


def robust_cut(
    shape_base: TopoDS_Shape,
    shape_tool: TopoDS_Shape,
    preprocess_shapes: bool = True,
    label: str = "cut",
) -> tuple[Optional[TopoDS_Shape], str]:
    """Robust cut (difference) with preprocessing and cascading fallbacks.

    Strategy:
      1. Preprocess both shapes
      2. Try BOPAlgo_CellsBuilder cut at increasing fuzzy values
      3. Fall back to BRepAlgoAPI_Cut
      4. Fall back to Splitter-based cut (split + classify fragments)
      5. Validate each result

    Returns:
        (result_shape, status_message)
        result_shape is None if all strategies failed.
    """
    if preprocess_shapes:
        print("[%s] Preprocessing operands..." % label)
        shape_base = preprocess(shape_base)
        shape_tool = preprocess(shape_tool)

    vol_base = compute_volume(shape_base)
    vol_tool = compute_volume(shape_tool)
    print("[%s] Base vol=%.1f, Tool vol=%.1f mm³" % (label, vol_base, vol_tool))

    # Strategy 1: CellsBuilder cut
    for fuzzy in FUZZY_VALUES:
        print("[%s] Trying CellsBuilder cut (fuzzy=%.3f)..." % (label, fuzzy))
        try:
            result = _cells_builder_cut(shape_base, shape_tool, fuzzy)
            if result is not None:
                ok, msg = validate_boolean_result(
                    result, [vol_base, vol_tool], "cut"
                )
                if ok:
                    result = heal_shape(result)
                    print("[%s] CellsBuilder cut OK: %s" % (label, msg))
                    return result, "CellsBuilder cut fuzzy=%.3f" % fuzzy
                else:
                    print("[%s] CellsBuilder cut invalid: %s" % (label, msg))
        except Exception as e:
            print("[%s] CellsBuilder cut error: %s" % (label, e))

    # Strategy 2: Standard BRepAlgoAPI_Cut
    for fuzzy in FUZZY_VALUES:
        print("[%s] Trying BRepAlgoAPI_Cut (fuzzy=%.3f)..." % (label, fuzzy))
        try:
            result = _standard_cut(shape_base, shape_tool, fuzzy)
            if result is not None:
                ok, msg = validate_boolean_result(
                    result, [vol_base, vol_tool], "cut"
                )
                if ok:
                    result = heal_shape(result)
                    print("[%s] Standard Cut OK: %s" % (label, msg))
                    return result, "BRepAlgoAPI_Cut fuzzy=%.3f" % fuzzy
                else:
                    print("[%s] Standard Cut result invalid: %s" % (label, msg))
        except Exception as e:
            print("[%s] Standard Cut error: %s" % (label, e))

    # Strategy 3: Splitter-based cut (most tolerant)
    for fuzzy in FUZZY_VALUES:
        print("[%s] Trying Splitter cut (fuzzy=%.3f)..." % (label, fuzzy))
        try:
            result = _splitter_cut(shape_base, shape_tool, fuzzy)
            if result is not None:
                ok, msg = validate_boolean_result(
                    result, [vol_base, vol_tool], "cut"
                )
                if ok:
                    result = heal_shape(result)
                    print("[%s] Splitter cut OK: %s" % (label, msg))
                    return result, "Splitter cut fuzzy=%.3f" % fuzzy
                else:
                    print("[%s] Splitter cut invalid: %s" % (label, msg))
        except Exception as e:
            print("[%s] Splitter cut error: %s" % (label, e))

    return None, "all cut strategies failed"


def robust_fuse_and_cut(
    oml_solid: TopoDS_Shape,
    shell_solid: TopoDS_Shape,
    duct_solid: TopoDS_Shape,
    label: str = "propulsion",
) -> tuple[Optional[TopoDS_Shape], str]:
    """Complete propulsion integration: Fuse(OML, Shell) then Cut(Duct).

    This is the main entry point for the v3 pipeline. Preprocessing is
    applied once at the start (not repeated per attempt).

    Returns:
        (result_shape, status_message)
    """
    print("=" * 60)
    print("[%s] Robust STEP boolean pipeline" % label)
    print("=" * 60)

    # Preprocess all three shapes once
    print("[%s] Step 1/3: Preprocessing all shapes..." % label)
    oml_pp = preprocess(oml_solid)
    shell_pp = preprocess(shell_solid)
    duct_pp = preprocess(duct_solid)

    # Step 2: Fuse OML + Shell
    print("[%s] Step 2/3: Fuse(OML, Shell)..." % label)
    fused, fuse_msg = robust_fuse(
        oml_pp, shell_pp, preprocess_shapes=False, label="fuse"
    )
    if fused is None:
        return None, "Fuse failed: %s" % fuse_msg

    # Step 3: Cut duct from fused
    print("[%s] Step 3/3: Cut(Fused, Duct)..." % label)
    result, cut_msg = robust_cut(
        fused, duct_pp, preprocess_shapes=False, label="cut"
    )
    if result is None:
        return None, "Cut failed after successful fuse: %s" % cut_msg

    vol = compute_volume(result)
    valid = is_valid_solid(result)
    summary = "OK — Fuse: %s, Cut: %s, vol=%.1f cm³, valid=%s" % (
        fuse_msg, cut_msg, vol / 1e6, valid,
    )
    print("[%s] %s" % (label, summary))
    print("=" * 60)

    return result, summary
