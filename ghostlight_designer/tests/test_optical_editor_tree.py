from __future__ import annotations

import pytest
from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtGui import QIcon

import ghostlight

from ghostlight_designer.optical_editor import OpticalEditorBody
from ghostlight_designer.optical_editor import surface_actions
from ghostlight_designer.optical_editor.columns import Column
from ghostlight_designer.optical_editor.delegates import NodeKindRole
from ghostlight_designer.optical_editor.model import OpticalTreeModel
from ghostlight_designer.optical_editor.nodes import NodeKind
from ghostlight_designer.optical_editor.toolbar import ADD_SINGLET
from ghostlight_designer.project import Project


def _collect(signal):
    received: list = []

    def slot(*args):
        received.append(args if len(args) != 1 else args[0])

    signal.connect(slot)
    return received


def _first_glass_element_index(project: Project) -> int:
    for i, el in enumerate(project.system.elements):
        if el.kind == ghostlight.ElementKind.GLASS:
            return i
    raise AssertionError("sample lens has no GLASS element")


def test_empty_project_yields_empty_root(qapp):
    project = Project()
    model = OpticalTreeModel(project)
    assert model.columnCount() == len(Column)
    assert model.rowCount(QModelIndex()) == 0


def test_tree_shape_for_sample_doublet(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    root_count = model.rowCount(QModelIndex())
    assert root_count == len(project.system.elements)
    assert root_count > 0

    for ei in range(root_count):
        el_idx = model.index(ei, 0, QModelIndex())
        element = project.system.elements[ei]
        expected_children = len(element.material_glasses) + len(element.surface_ids)
        assert model.rowCount(el_idx) == expected_children


def test_column_data_for_first_surface(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei = _first_glass_element_index(project)
    el = project.system.elements[ei]
    el_idx = model.index(ei, 0, QModelIndex())

    # Material rows come before surface rows.
    surface_row = len(el.material_glasses)
    radius_idx = model.index(surface_row, int(Column.RADIUS), el_idx)

    surf_index = el.resolve_surfaces(project.system)[0]
    surf = project.system.surfaces[surf_index]
    if int(surf.form) == int(ghostlight.SurfaceForm.SPHERE):
        assert model.data(radius_idx, Qt.DisplayRole) == f"{surf.radius:.4f}"
    else:
        assert model.data(radius_idx, Qt.DisplayRole) == ""


def test_setData_on_radius_writes_through_and_marks_dirty(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)
    dirty = _collect(project.dirtyChanged)

    ei = _first_glass_element_index(project)
    el = project.system.elements[ei]
    el_idx = model.index(ei, 0, QModelIndex())
    surface_row = len(el.material_glasses)
    surf_index = el.resolve_surfaces(project.system)[0]
    if int(project.system.surfaces[surf_index].form) != int(ghostlight.SurfaceForm.SPHERE):
        pytest.skip("first surface of first GLASS element is not SPHERE; radius edit deferred")

    radius_idx = model.index(surface_row, int(Column.RADIUS), el_idx)
    original = project.system.surfaces[surf_index].radius
    new_val = original + 5.0

    ok = model.setData(radius_idx, new_val, Qt.EditRole)
    assert ok
    assert project.system.surfaces[surf_index].radius == pytest.approx(new_val)
    assert project.is_dirty is True
    assert dirty == [True]


def test_setData_on_surface_pos_z_writes_through(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei = _first_glass_element_index(project)
    el = project.system.elements[ei]
    el_idx = model.index(ei, 0, QModelIndex())
    surface_row = len(el.material_glasses)
    surf_uuid = el.surface_ids[0]
    surf_index = el.resolve_surfaces(project.system)[0]

    # Force absolute mode to assert the historical surf.z write path.  (The
    # default is now relative — covered by its own test below.)
    model.set_pos_z_mode(surf_uuid, "absolute")

    pos_z_idx = model.index(surface_row, int(Column.POS_Z), el_idx)
    original = project.system.surfaces[surf_index].z
    new_val = original + 2.25

    ok = model.setData(pos_z_idx, new_val, Qt.EditRole)
    assert ok
    assert project.system.surfaces[surf_index].z == pytest.approx(new_val)
    assert project.is_dirty is True


def test_surface_pos_z_defaults_to_relative_thickness(qapp, sample_lens_path):
    """A freshly loaded document shows every surface's Pos Z as thickness
    (the on-disk value); flipping to absolute exposes the computed z."""
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei = _first_glass_element_index(project)
    el = project.system.elements[ei]
    el_idx = model.index(ei, 0, QModelIndex())
    surface_row = len(el.material_glasses)
    surf_uuid = el.surface_ids[0]
    surf_index = el.resolve_surfaces(project.system)[0]
    surf = project.system.surfaces[surf_index]

    pos_z_idx = model.index(surface_row, int(Column.POS_Z), el_idx)

    # Default: relative (mirrors surf.thickness, with (rel) suffix in display).
    assert model.pos_z_mode(surf_uuid) == "relative"
    assert model.data(pos_z_idx, Qt.DisplayRole) == f"{surf.thickness:.4f} (rel)"
    assert model.data(pos_z_idx, Qt.EditRole) == pytest.approx(surf.thickness)

    # Edits in the default mode write to thickness AND rederive the whole z
    # chain via OpticalSystem.finalize() so the viewport / raytracer see the
    # change (they read z, not thickness).
    delta = 1.5
    original_z_chain = [s.z for s in project.system.surfaces]
    new_thickness = surf.thickness + delta
    ok = model.setData(pos_z_idx, new_thickness, Qt.EditRole)
    assert ok
    assert project.system.surfaces[surf_index].thickness == pytest.approx(new_thickness)
    # finalize() walks backward from sensor=0: surfaces at and before the
    # edited one shift by -delta, surfaces after it stay anchored.
    for i, s in enumerate(project.system.surfaces):
        if i <= surf_index:
            assert s.z == pytest.approx(original_z_chain[i] - delta), f"surface {i} should shift"
        else:
            assert s.z == pytest.approx(original_z_chain[i]), f"surface {i} should stay"

    # Flip to absolute: display shows z, EditRole returns it, dataChanged fires.
    new_z = project.system.surfaces[surf_index].z
    changes = _collect(model.dataChanged)
    model.set_pos_z_mode(surf_uuid, "absolute")
    assert model.pos_z_mode(surf_uuid) == "absolute"
    assert model.data(pos_z_idx, Qt.DisplayRole) == f"{new_z:.4f}"
    assert model.data(pos_z_idx, Qt.EditRole) == pytest.approx(new_z)
    assert changes, "set_pos_z_mode should emit dataChanged for the affected row"


def test_pos_z_mode_cleared_on_system_replaced(qapp, sample_lens_path):
    """Loading a new document drops per-surface display overrides — they
    belong to the old document's UUIDs — and the new document falls back
    to the relative default."""
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei = _first_glass_element_index(project)
    el = project.system.elements[ei]
    surf_uuid = el.surface_ids[0]
    model.set_pos_z_mode(surf_uuid, "absolute")
    assert model.pos_z_mode(surf_uuid) == "absolute"

    project.load(str(sample_lens_path))
    assert model.pos_z_mode(surf_uuid) == "relative"


def test_cylindrical_axis_editable_on_identifier(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei = _first_glass_element_index(project)
    el = project.system.elements[ei]
    el_idx = model.index(ei, 0, QModelIndex())
    surface_row = len(el.material_glasses)
    surf_index = el.resolve_surfaces(project.system)[0]

    # Switch the surface to CYLINDRICAL so the form-modifier child appears.
    # Form changes go through the action helper (right-click "Swap Form"),
    # not through an inline editor on the Name column.
    assert surface_actions.set_surface_form(
        project, surf_index, int(ghostlight.SurfaceForm.CYLINDRICAL)
    )

    # Re-fetch indexes after the model reset.
    el_idx = model.index(ei, 0, QModelIndex())
    surf_idx = model.index(surface_row, int(Column.NAME), el_idx)
    cyl_id_idx = model.index(0, int(Column.IDENTIFIER), surf_idx)

    # Identifier cell shows the axis name and exposes the int via EditRole.
    expected_name = ghostlight.CylinderAxis(int(project.system.surfaces[surf_index].cyl_axis)).name
    assert model.data(cyl_id_idx, Qt.DisplayRole) == expected_name
    assert model.data(cyl_id_idx, Qt.EditRole) == int(project.system.surfaces[surf_index].cyl_axis)

    # Toggle to the other axis and confirm it persists.
    current = int(project.system.surfaces[surf_index].cyl_axis)
    other = int(ghostlight.CylinderAxis.AXIS_Y) if current == int(ghostlight.CylinderAxis.AXIS_X) else int(ghostlight.CylinderAxis.AXIS_X)
    assert model.setData(cyl_id_idx, other, Qt.EditRole)
    assert int(project.system.surfaces[surf_index].cyl_axis) == other


def test_element_pos_z_is_hidden(qapp, sample_lens_path):
    """Element rows don't render Pos Z — position is adjusted per-surface.
    Display/edit roles both return empty so the cell looks blank and the
    cell isn't editable from the element row."""
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei = 0
    pos_z_idx = model.index(ei, int(Column.POS_Z), QModelIndex())

    assert model.data(pos_z_idx, Qt.DisplayRole) == ""
    assert model.data(pos_z_idx, Qt.EditRole) is None
    assert not (model.flags(pos_z_idx) & Qt.ItemIsEditable)

    # Trying to setData on the element row's Pos Z is a no-op and does not
    # mutate any surface's z.
    rearmost_si = max(
        project.system.elements[ei].resolve_surfaces(project.system),
        key=lambda si: project.system.surfaces[si].z,
    )
    original_rearmost_z = project.system.surfaces[rearmost_si].z
    ok = model.setData(pos_z_idx, original_rearmost_z + 3.5, Qt.EditRole)
    assert not ok
    assert project.system.surfaces[rearmost_si].z == pytest.approx(
        original_rearmost_z
    )


def test_setData_on_element_identifier_updates_dataclass(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    changes = _collect(model.dataChanged)
    identifier_idx = model.index(0, int(Column.IDENTIFIER), QModelIndex())
    ok = model.setData(identifier_idx, "Renamed", Qt.EditRole)
    assert ok
    assert project.system.elements[0].name == "Renamed"
    assert len(changes) >= 1


def test_name_column_shows_type_labels(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei = _first_glass_element_index(project)
    el_idx = model.index(ei, 0, QModelIndex())
    assert model.data(el_idx, Qt.DisplayRole) == "Lens"

    el = project.system.elements[ei]
    if el.material_glasses:
        mat_idx = model.index(0, int(Column.NAME), el_idx)
        assert model.data(mat_idx, Qt.DisplayRole) == "Material"

    surface_row = len(el.material_glasses)
    surf_idx = model.index(surface_row, int(Column.NAME), el_idx)
    assert model.data(surf_idx, Qt.DisplayRole) == "Surface"


def test_identifier_column_shows_user_given_name(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei = _first_glass_element_index(project)
    el = project.system.elements[ei]
    el_id_idx = model.index(ei, int(Column.IDENTIFIER), QModelIndex())
    assert model.data(el_id_idx, Qt.DisplayRole) == el.name

    if el.material_glasses:
        # Material row's catalogue name lives in the Radius-strip column
        # under the new 4-slot schema (Designer / Name / nd / Vd).
        # ``material_glasses[i]`` stores the catalogue key (bare-name
        # legacy values like "N-BK7" resolve to their display name via
        # the catalogue fallback in row_schemas).
        el_idx = model.index(ei, 0, QModelIndex())
        mat_name_idx = model.index(0, int(Column.RADIUS), el_idx)
        assert model.data(mat_name_idx, Qt.DisplayRole) == el.material_glasses[0]


def test_decoration_role_returns_qicon(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    name_idx = model.index(0, int(Column.NAME), QModelIndex())
    decoration = model.data(name_idx, Qt.DecorationRole)
    assert isinstance(decoration, QIcon)


def test_node_kind_role_distinguishes_node_kinds(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei = _first_glass_element_index(project)
    el_idx = model.index(ei, 0, QModelIndex())
    assert model.data(el_idx, NodeKindRole) == int(NodeKind.ELEMENT)

    surface_row = len(project.system.elements[ei].material_glasses)
    surf_idx = model.index(surface_row, 0, el_idx)
    assert model.data(surf_idx, NodeKindRole) == int(NodeKind.SURFACE)


@pytest.mark.parametrize("form_name,expected_child_name", [
    ("ASPHERE", "Asphere Form"),
    ("CYLINDRICAL", "Cylindrical Form"),
])
def test_non_spherical_form_moves_radius_to_child_row(
    qapp, sample_lens_path, form_name, expected_child_name
):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei = _first_glass_element_index(project)
    el = project.system.elements[ei]
    el_idx = model.index(ei, 0, QModelIndex())
    surface_row = len(el.material_glasses)
    surf_index = el.resolve_surfaces(project.system)[0]
    original_radius = project.system.surfaces[surf_index].radius

    # Switch the surface form via the action helper (driven by the
    # right-click "Swap Form" submenu — the inline Name-column combo is
    # gone).
    target_form = int(getattr(ghostlight.SurfaceForm, form_name))
    assert surface_actions.set_surface_form(project, surf_index, target_form)

    # Form change triggers a model reset, so re-fetch every index.
    el_idx = model.index(ei, 0, QModelIndex())
    surf_idx = model.index(surface_row, int(Column.NAME), el_idx)
    surf_radius_idx = model.index(surface_row, int(Column.RADIUS), el_idx)

    # Surface row's Radius cell must now be blank.
    assert model.data(surf_radius_idx, Qt.DisplayRole) == ""

    # Two child rows now: the form-modifier row (at index 0, holding the
    # radius) plus the always-present coating row (appended last).
    assert model.rowCount(surf_idx) == 2
    form_row_name_idx = model.index(0, int(Column.NAME), surf_idx)
    assert model.data(form_row_name_idx, Qt.DisplayRole) == expected_child_name
    form_row_radius_idx = model.index(0, int(Column.RADIUS), surf_idx)
    assert model.data(form_row_radius_idx, Qt.DisplayRole) == f"{original_radius:.4f}"

    # Editing radius on the child row writes through to the underlying surface.
    assert model.setData(form_row_radius_idx, original_radius + 7.5, Qt.EditRole)
    assert project.system.surfaces[surf_index].radius == pytest.approx(original_radius + 7.5)


def test_parent_resolution_for_deep_indices(qapp, sample_lens_path):
    """Regression: parent() called on child indices must not blow up.

    The auto-generated dataclass __eq__ used to recurse forever via
    parent.children.index(self), because tree nodes reference each other
    cyclically. Identity-based row() / eq=False on the dataclasses fix it.
    """
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    for ei in range(model.rowCount(QModelIndex())):
        el_idx = model.index(ei, 0, QModelIndex())
        parent_of_el = model.parent(el_idx)
        assert not parent_of_el.isValid()  # element's parent is root

        for child_row in range(model.rowCount(el_idx)):
            child_idx = model.index(child_row, 0, el_idx)
            parent_idx = model.parent(child_idx)
            assert parent_idx.isValid()
            assert parent_idx.row() == ei
            assert parent_idx.internalPointer() is el_idx.internalPointer()


def test_systemReplaced_rebuilds_tree(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)
    assert model.rowCount(QModelIndex()) > 0

    resets = _collect(model.modelReset)
    project.new()
    assert len(resets) >= 1
    assert model.rowCount(QModelIndex()) == 0


def test_body_widget_loads_and_replaces(qapp, sample_lens_path):
    project = Project()
    body = OpticalEditorBody(project)
    try:
        assert body.model.rowCount(QModelIndex()) == 0
        project.load(str(sample_lens_path))
        assert body.model.rowCount(QModelIndex()) == len(project.system.elements)
        project.new()
        assert body.model.rowCount(QModelIndex()) == 0
    finally:
        body.deleteLater()


def test_add_element_only_expands_the_new_row(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        for ei in range(body.model.rowCount(QModelIndex())):
            body.tree.setExpanded(body.model.index(ei, 0, QModelIndex()), False)

        body._on_add_element_requested(ADD_SINGLET)

        new_idx = body.model.index(0, 0, QModelIndex())
        assert body.tree.isExpanded(new_idx)
        for ei in range(1, body.model.rowCount(QModelIndex())):
            sib_idx = body.model.index(ei, 0, QModelIndex())
            assert not body.tree.isExpanded(sib_idx)
    finally:
        body.deleteLater()


def _material_row(model, project, ei: int = None):
    """Return ``(el_idx, material_row=0, element)`` for the first glass
    element's first Material row."""
    if ei is None:
        ei = _first_glass_element_index(project)
    el = project.system.elements[ei]
    el_idx = model.index(ei, 0, QModelIndex())
    return el_idx, 0, el


def test_material_row_in_catalog_mode_shows_designer_and_name_only(
    qapp, sample_lens_path,
):
    """Vendor-catalogue materials show Designer + Name. nd/Vd are
    deliberately hidden in catalog mode — the canonical numbers live on
    the CatalogueMaterial and showing read-only copies next to the
    editable rows above invited misclicks. They appear only in Custom
    mode, where they're the user's to edit."""
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    el_idx, mat_row, el = _material_row(model, project)
    if not el.material_glasses:
        pytest.skip("first glass element has no materials")

    designer_idx = model.index(mat_row, int(Column.IDENTIFIER), el_idx)
    name_idx     = model.index(mat_row, int(Column.RADIUS), el_idx)
    nd_idx       = model.index(mat_row, int(Column.POS_Z), el_idx)
    vd_idx       = model.index(mat_row, int(Column.APERTURE_RAD), el_idx)

    assert model.data(designer_idx, Qt.DisplayRole) == "Schott"
    assert model.data(name_idx, Qt.DisplayRole) == el.material_glasses[mat_row]
    assert model.data(nd_idx, Qt.DisplayRole) == ""
    assert model.data(vd_idx, Qt.DisplayRole) == ""


def test_catalog_material_nd_vd_are_read_only(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)
    el_idx, mat_row, _el = _material_row(model, project)

    nd_idx = model.index(mat_row, int(Column.POS_Z), el_idx)
    vd_idx = model.index(mat_row, int(Column.APERTURE_RAD), el_idx)
    assert not (model.flags(nd_idx) & Qt.ItemIsEditable)
    assert not (model.flags(vd_idx) & Qt.ItemIsEditable)


def test_designer_switch_to_custom_adds_project_local_entry(
    qapp, sample_lens_path,
):
    """Picking ``Custom`` from the Designer combo generates a fresh
    ``Custom_<hash>`` key, copies the current nd/Vd into the project's
    bundled glass_catalogue under that key, and leaves Surface.ior
    unchanged (since the seed values match the previous material)."""
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)
    el_idx, mat_row, el = _material_row(model, project)

    # Snapshot the Surface.ior the affected surface starts at — switching
    # to Custom with the same nd should leave it untouched.
    surf_uuid = el.surface_ids[mat_row]
    surf_index = list(project.system.surface_ids).index(surf_uuid)
    ior_before = float(project.system.surfaces[surf_index].ior)

    designer_idx = model.index(mat_row, int(Column.IDENTIFIER), el_idx)
    assert model.setData(designer_idx, "Custom", Qt.EditRole)

    # material_glasses[i] now carries the synthesized key.
    new_key = el.material_glasses[mat_row]
    assert new_key.startswith("Custom_")
    # And the Designer cell renders "Custom" — the unified rule "no
    # MaterialCatalogue match = Custom" applies as soon as the key
    # changes.
    assert model.data(designer_idx, Qt.DisplayRole) == "Custom"
    # Project catalogue gained the entry; it's Abbe with the seed nd/Vd.
    entry = project.system._raw_glass_catalogue[new_key]
    assert entry["dispersion"]["model"] == "abbe"
    assert entry["dispersion"]["nd"] == pytest.approx(1.5168)
    assert entry["dispersion"]["Vd"] == pytest.approx(64.17)
    # Surface.ior unchanged on switch.
    assert project.system.surfaces[surf_index].ior == pytest.approx(ior_before)
    # nd/Vd are now editable on the row — AND now visible (hidden in
    # catalog mode, shown the moment the material becomes editable).
    nd_idx = model.index(mat_row, int(Column.POS_Z), el_idx)
    vd_idx = model.index(mat_row, int(Column.APERTURE_RAD), el_idx)
    assert model.flags(nd_idx) & Qt.ItemIsEditable
    assert model.flags(vd_idx) & Qt.ItemIsEditable
    assert model.data(nd_idx, Qt.DisplayRole) == "1.51680"
    assert model.data(vd_idx, Qt.DisplayRole) == "64.17"


def test_legacy_unknown_glass_key_shows_custom_designer(
    qapp, sample_lens_path,
):
    """A material whose glass key isn't in MaterialCatalogue (legacy
    project-local glass) should render Designer='Custom'. The unified
    rule treats anything outside the bundled vendor catalogues as Custom
    so the row's nd/Vd become user-editable on load, not after a manual
    Designer-combo round trip."""
    project = Project()
    project.load(str(sample_lens_path))
    el_idx_int, mat_row, el = _material_row_indices(
        OpticalTreeModel(project), project
    )
    if not el.material_glasses:
        pytest.skip("first glass element has no materials")

    # Replace material_glasses[i] with an unmatched name and ensure the
    # project carries a bundled Abbe entry the C++ loader would accept.
    el.material_glasses[mat_row] = "UnknownGlassXYZ"
    project.system._raw_glass_catalogue["UnknownGlassXYZ"] = {
        "name": "UnknownGlassXYZ",
        "dispersion": {"model": "abbe", "nd": 1.62, "Vd": 38.0},
    }

    model = OpticalTreeModel(project)
    el_idx = model.index(el_idx_int, 0, QModelIndex())
    designer_idx = model.index(mat_row, int(Column.IDENTIFIER), el_idx)
    nd_idx = model.index(mat_row, int(Column.POS_Z), el_idx)
    vd_idx = model.index(mat_row, int(Column.APERTURE_RAD), el_idx)

    assert model.data(designer_idx, Qt.DisplayRole) == "Custom"
    # nd/Vd are surfaced from the project's bundled dispersion entry,
    # and are editable since the material is project-local.
    assert model.flags(nd_idx) & Qt.ItemIsEditable
    assert model.data(nd_idx, Qt.DisplayRole) == "1.62000"
    assert model.data(vd_idx, Qt.DisplayRole) == "38.00"


def test_convert_material_to_vendor_finds_close_match(
    qapp, sample_lens_path,
):
    from ghostlight_designer.optical_editor.row_schemas import (
        convert_material_to_custom,
        convert_material_to_vendor,
    )

    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)
    el_idx_int, mat_row, el = _material_row_indices(model, project)

    # Detach from Schott_N-BK7 by going to Custom (seeds with the same
    # nd/Vd). Then convert back to Schott — close-enough match should
    # land on a Schott_* key.
    el_idx = model.index(el_idx_int, 0, QModelIndex())
    mat_node = model.index(mat_row, 0, el_idx).internalPointer()
    assert convert_material_to_custom(project, mat_node)
    assert el.material_glasses[mat_row].startswith("Custom_")

    ok, message = convert_material_to_vendor(project, mat_node, "Schott")
    assert ok, message
    new_key = el.material_glasses[mat_row]
    assert new_key.startswith("Schott_"), new_key


def test_convert_material_to_vendor_returns_message_when_no_match(
    qapp, sample_lens_path,
):
    from ghostlight_designer.optical_editor.row_schemas import (
        convert_material_to_vendor,
    )

    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)
    el_idx_int, mat_row, el = _material_row_indices(model, project)

    # Push the material to a hand-crafted nd/Vd no Schott glass should
    # be within tolerance of, then attempt the convert.
    custom_key = "Custom_unmatchable"
    el.material_glasses[mat_row] = custom_key
    project.system._raw_glass_catalogue[custom_key] = {
        "name": custom_key,
        "dispersion": {"model": "abbe", "nd": 9.999, "Vd": 999.0},
    }
    el_idx = model.index(el_idx_int, 0, QModelIndex())
    mat_node = model.index(mat_row, 0, el_idx).internalPointer()

    ok, message = convert_material_to_vendor(project, mat_node, "Schott")
    assert not ok
    assert "Schott" in message
    # Material stays as-is — caller is responsible for surfacing the
    # message to the user, not the function.
    assert el.material_glasses[mat_row] == custom_key


def _material_row_indices(model, project, ei=None):
    """Return ``(element_index, material_row=0, element)`` — utility for
    the Material-row tests below."""
    if ei is None:
        ei = _first_glass_element_index(project)
    el = project.system.elements[ei]
    return ei, 0, el


def test_custom_material_nd_edit_propagates_to_surface_ior(
    qapp, sample_lens_path,
):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)
    el_idx, mat_row, el = _material_row(model, project)

    # Switch to Custom mode first so nd is editable.
    designer_idx = model.index(mat_row, int(Column.IDENTIFIER), el_idx)
    assert model.setData(designer_idx, "Custom", Qt.EditRole)

    surf_uuid = el.surface_ids[mat_row]
    surf_index = list(project.system.surface_ids).index(surf_uuid)

    nd_idx = model.index(mat_row, int(Column.POS_Z), el_idx)
    new_nd = 1.6234
    assert model.setData(nd_idx, new_nd, Qt.EditRole)

    # Surface IOR and the bundled catalogue entry both reflect the edit.
    assert project.system.surfaces[surf_index].ior == pytest.approx(new_nd)
    new_key = el.material_glasses[mat_row]
    assert project.system._raw_glass_catalogue[new_key]["dispersion"]["nd"] \
        == pytest.approx(new_nd)


# EFL shares the Radius strip column on the Element row; the in-cell
# "EFL" label disambiguates from Surface's "Radius".
_EFL_COLUMN = int(Column.RADIUS)


def test_element_efl_cell_shows_finite_focal_length(qapp, sample_lens_path):
    """The doublet element row's EFL cell renders a finite paraxial
    focal length in millimetres. Exact value depends on the lens
    prescription, so the assertion is bounded — what matters is that the
    paraxial product produced a sane positive convex-doublet result."""
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei = _first_glass_element_index(project)
    el_idx = model.index(ei, _EFL_COLUMN, QModelIndex())
    display = model.data(el_idx, Qt.DisplayRole)
    assert isinstance(display, str)
    assert display.endswith("mm"), display
    # The value parses to a non-zero finite float — empty-cell / "0.000 mm"
    # would indicate the matrix product short-circuited unexpectedly.
    value = float(display.removesuffix("mm").strip())
    assert value != 0.0
    assert abs(value) < 1.0e6


def test_element_efl_cell_is_read_only(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei = _first_glass_element_index(project)
    efl_idx = model.index(ei, _EFL_COLUMN, QModelIndex())
    assert not (model.flags(efl_idx) & Qt.ItemIsEditable)


def test_aperture_stop_surface_radius_is_hidden_but_others_remain(qapp):
    """A stop surface row hides its Radius cell (the value is nominal —
    user shouldn't be tuning it) but keeps Pos Z and Aperture Rad
    populated and editable. The Radius column stays in the table so
    Pos Z / Aperture Rad still line up with the rest of the chain."""
    from ghostlight_designer.optical_editor import element_actions

    project = Project()
    element_actions.add_aperture_stop(project)
    model = OpticalTreeModel(project)

    el_idx = model.index(0, 0, QModelIndex())
    # Aperture-stop element has no materials — surface row is the first
    # child.
    surf_row = 0
    radius_idx = model.index(surf_row, int(Column.RADIUS), el_idx)
    pos_z_idx = model.index(surf_row, int(Column.POS_Z), el_idx)
    apert_idx = model.index(surf_row, int(Column.APERTURE_RAD), el_idx)

    # Radius hidden — get returns None, cell not editable, paint
    # suppression kicks in.
    assert model.data(radius_idx, Qt.DisplayRole) == ""
    assert model.data(radius_idx, Qt.EditRole) is None
    assert not (model.flags(radius_idx) & Qt.ItemIsEditable)

    # Pos Z + Aperture Rad untouched.
    assert model.data(pos_z_idx, Qt.DisplayRole) != ""
    assert model.flags(pos_z_idx) & Qt.ItemIsEditable
    assert model.data(apert_idx, Qt.DisplayRole) != ""
    assert model.flags(apert_idx) & Qt.ItemIsEditable


def test_aperture_stop_efl_is_blank(qapp):
    """Aperture stops have no glass and therefore no paraxial focal
    length; the cell displays empty."""
    from ghostlight_designer.optical_editor import element_actions

    project = Project()
    element_actions.add_aperture_stop(project)
    model = OpticalTreeModel(project)

    # The stop is the only element after add_aperture_stop on an empty
    # project (insertion goes to the front of the chain).
    efl_idx = model.index(0, _EFL_COLUMN, QModelIndex())
    assert model.data(efl_idx, Qt.DisplayRole) == ""


def test_load_into_empty_expands_all_then_add_only_expands_new(qapp, sample_lens_path):
    """Repro of the user-reported flow: empty project → load lens →
    add singlet. Load should expand all (it's a fresh document); the
    subsequent add should ONLY expand the new row, not re-expand the
    rest of the loaded chain."""
    project = Project()
    body = OpticalEditorBody(project)
    try:
        project.load(str(sample_lens_path))
        # Fresh load → every row expanded.
        for ei in range(body.model.rowCount(QModelIndex())):
            assert body.tree.isExpanded(body.model.index(ei, 0, QModelIndex()))

        # User collapses everything.
        for ei in range(body.model.rowCount(QModelIndex())):
            body.tree.setExpanded(body.model.index(ei, 0, QModelIndex()), False)

        body._on_add_element_requested(ADD_SINGLET)

        new_idx = body.model.index(0, 0, QModelIndex())
        assert body.tree.isExpanded(new_idx)
        for ei in range(1, body.model.rowCount(QModelIndex())):
            sib_idx = body.model.index(ei, 0, QModelIndex())
            assert not body.tree.isExpanded(sib_idx)
    finally:
        body.deleteLater()


# ---------------------------------------------------------------------------
# Asphere form row — term counter (Identifier) + dynamic coefficient columns
# ---------------------------------------------------------------------------


def _switch_first_surface_to_asphere(project):
    """Helper: flip the first glass element's front surface to ASPHERE and
    return ``(element_index, surface_row, surface_index)``."""
    ei = _first_glass_element_index(project)
    el = project.system.elements[ei]
    surface_row = len(el.material_glasses)
    surf_index = el.resolve_surfaces(project.system)[0]
    assert surface_actions.set_surface_form(
        project, surf_index, int(ghostlight.SurfaceForm.ASPHERE)
    )
    return ei, surface_row, surf_index


def test_asphere_identifier_holds_term_count(qapp, sample_lens_path):
    """The asphere child row's Identifier cell is the term counter.

    EditRole returns an ``int`` (the value scrubber dispatches off the
    type, so int matters), DisplayRole formats it, and the cell is
    editable.
    """
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei, surface_row, surf_index = _switch_first_surface_to_asphere(project)
    surf = project.system.surfaces[surf_index]
    assert int(surf.n_asphere_terms) == 0  # default after form swap

    el_idx = model.index(ei, 0, QModelIndex())
    surf_idx = model.index(surface_row, int(Column.NAME), el_idx)
    asphere_idx = model.index(0, int(Column.IDENTIFIER), surf_idx)

    assert model.data(asphere_idx, Qt.DisplayRole) == "0"
    edit_value = model.data(asphere_idx, Qt.EditRole)
    assert isinstance(edit_value, int)
    assert edit_value == 0
    assert model.flags(asphere_idx) & Qt.ItemIsEditable


def test_asphere_term_count_growth_adds_coefficient_columns(
    qapp, sample_lens_path
):
    """Raising the term count grows the tree's column count live so the
    new coefficient cells have somewhere to render."""
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    # Coating rows pack their live slots inside the canonical strip, so a
    # loaded lens reserves no trailing columns until an asphere needs them.
    base_cols = model.columnCount()
    assert base_cols == len(Column)

    ei, surface_row, surf_index = _switch_first_surface_to_asphere(project)
    # No terms yet → still no trailing columns.
    assert model.columnCount() == len(Column)

    el_idx = model.index(ei, 0, QModelIndex())
    surf_idx = model.index(surface_row, int(Column.NAME), el_idx)
    asphere_idx = model.index(0, int(Column.IDENTIFIER), surf_idx)

    assert model.setData(asphere_idx, 3, Qt.EditRole)
    assert int(project.system.surfaces[surf_index].n_asphere_terms) == 3
    # Three coefficient columns (A4, A6, A8) — the width follows the asphere
    # term count.
    assert model.columnCount() == len(Column) + 3

    # Drop back to 1 — the asphere block shrinks with it.
    el_idx = model.index(ei, 0, QModelIndex())
    surf_idx = model.index(surface_row, int(Column.NAME), el_idx)
    asphere_idx = model.index(0, int(Column.IDENTIFIER), surf_idx)
    assert model.setData(asphere_idx, 1, Qt.EditRole)
    assert model.columnCount() == len(Column) + 1


def test_asphere_inactive_coefficient_cells_are_blank(qapp, sample_lens_path):
    """Coefficient slots beyond ``n_asphere_terms`` have no value, no
    display, and aren't editable — even if columns exist globally because
    another asphere surface has more terms."""
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei, surface_row, surf_index = _switch_first_surface_to_asphere(project)
    el_idx = model.index(ei, 0, QModelIndex())
    surf_idx = model.index(surface_row, int(Column.NAME), el_idx)
    asphere_idx = model.index(0, int(Column.IDENTIFIER), surf_idx)

    # Activate two coefficients.
    assert model.setData(asphere_idx, 2, Qt.EditRole)
    el_idx = model.index(ei, 0, QModelIndex())
    surf_idx = model.index(surface_row, int(Column.NAME), el_idx)
    first_trailing = len(Column)  # column 5

    a4_idx = model.index(0, first_trailing, surf_idx)
    a6_idx = model.index(0, first_trailing + 1, surf_idx)

    assert model.flags(a4_idx) & Qt.ItemIsEditable
    assert model.flags(a6_idx) & Qt.ItemIsEditable
    assert model.data(a4_idx, Qt.EditRole) == pytest.approx(0.0)
    assert model.data(a6_idx, Qt.EditRole) == pytest.approx(0.0)

    # Activate every slot so all trailing columns exist, then drop the
    # count back so trailing slots on THIS row are inactive while the
    # columns remain.
    asphere_idx = model.index(0, int(Column.IDENTIFIER), surf_idx)
    assert model.setData(asphere_idx, 8, Qt.EditRole)
    el_idx = model.index(ei, 0, QModelIndex())
    surf_idx = model.index(surface_row, int(Column.NAME), el_idx)
    asphere_idx = model.index(0, int(Column.IDENTIFIER), surf_idx)
    assert model.setData(asphere_idx, 3, Qt.EditRole)
    # After the drop the global max is 3 — columns shrink — so the test
    # just confirms cells past index 3 don't appear within the active
    # column range. Active cells at offsets 0, 1, 2 stay editable.
    el_idx = model.index(ei, 0, QModelIndex())
    surf_idx = model.index(surface_row, int(Column.NAME), el_idx)
    for offset in range(3):
        idx = model.index(0, first_trailing + offset, surf_idx)
        assert model.flags(idx) & Qt.ItemIsEditable
    # No trailing columns beyond offset 2 (term count = 3).
    assert model.columnCount() == first_trailing + 3


def test_asphere_coefficient_write_round_trips(qapp, sample_lens_path):
    """Writing to an active coefficient cell updates the underlying
    surface's ``asphere_terms`` array."""
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei, surface_row, surf_index = _switch_first_surface_to_asphere(project)
    el_idx = model.index(ei, 0, QModelIndex())
    surf_idx = model.index(surface_row, int(Column.NAME), el_idx)
    asphere_idx = model.index(0, int(Column.IDENTIFIER), surf_idx)
    assert model.setData(asphere_idx, 2, Qt.EditRole)

    el_idx = model.index(ei, 0, QModelIndex())
    surf_idx = model.index(surface_row, int(Column.NAME), el_idx)
    first_trailing = len(Column)
    a4_idx = model.index(0, first_trailing, surf_idx)
    a6_idx = model.index(0, first_trailing + 1, surf_idx)

    assert model.setData(a4_idx, 1.25e-4, Qt.EditRole)
    assert model.setData(a6_idx, -3.75e-7, Qt.EditRole)

    surf = project.system.surfaces[surf_index]
    assert int(surf.n_asphere_terms) == 2
    assert float(surf.asphere_terms[0]) == pytest.approx(1.25e-4)
    assert float(surf.asphere_terms[1]) == pytest.approx(-3.75e-7)


def test_asphere_conic_k_lives_in_pos_z_column(qapp, sample_lens_path):
    """The asphere row reuses the Pos Z canonical column for the conic
    constant K — the in-cell ``K`` label disambiguates it from Surface
    rows' Pos Z, since slot.label differs from the strip header."""
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei, surface_row, surf_index = _switch_first_surface_to_asphere(project)
    el_idx = model.index(ei, 0, QModelIndex())
    surf_idx = model.index(surface_row, int(Column.NAME), el_idx)
    k_idx = model.index(0, int(Column.POS_Z), surf_idx)

    # Default conic_k is 0 after the form swap.
    assert model.data(k_idx, Qt.EditRole) == pytest.approx(0.0)

    assert model.setData(k_idx, -1.0, Qt.EditRole)
    assert float(project.system.surfaces[surf_index].conic_k) == \
        pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Aperture form row — shape discriminator + dynamic blade / rotation cells
# ---------------------------------------------------------------------------


def _add_aperture_stop_project(qapp):
    """Helper: fresh project with one aperture-stop element, model attached.

    Returns ``(project, model, stop_surface_index)``.
    """
    from ghostlight_designer.optical_editor import element_actions

    project = Project()
    element_actions.add_aperture_stop(project)
    model = OpticalTreeModel(project)
    el = project.system.elements[0]
    surf_index = el.resolve_surfaces(project.system)[0]
    return project, model, surf_index


def test_aperture_stop_surface_has_aperture_form_child(qapp):
    """Every aperture-stop surface gets an Aperture Form child row exposing
    aperture_shape and shape-specific scalars."""
    project, model, _surf_index = _add_aperture_stop_project(qapp)

    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    # A fresh aperture stop is uncoated, and stops can't take coatings anyway,
    # so its only child is the aperture form row.
    assert model.rowCount(surf_idx) == 1
    form_idx = model.index(0, int(Column.NAME), surf_idx)
    assert model.data(form_idx, Qt.DisplayRole) == "Aperture Form"


def test_non_stop_surface_has_no_aperture_form_child(qapp, sample_lens_path):
    """Regular optical surfaces don't render an Aperture Form row even
    though the C++ struct carries the same aperture-shape fields. We
    surface this UI only on aperture stops per the editor scope."""
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei = _first_glass_element_index(project)
    el = project.system.elements[ei]
    surface_row = len(el.material_glasses)
    el_idx = model.index(ei, 0, QModelIndex())
    surf_idx = model.index(surface_row, int(Column.NAME), el_idx)
    # A plain SPHERE non-stop surface has no aperture/form-modifier child —
    # only the always-present coating row. Its single child must be the
    # coating row, not an aperture form.
    surf_index = el.resolve_surfaces(project.system)[0]
    if int(project.system.surfaces[surf_index].form) == int(ghostlight.SurfaceForm.SPHERE):
        assert model.rowCount(surf_idx) == 1
        child_idx = model.index(0, int(Column.NAME), surf_idx)
        assert model.data(child_idx, NodeKindRole) == int(NodeKind.COATING_FORM)


def test_aperture_shape_combo_starts_at_circle(qapp):
    """Newly-added stop surfaces default to CIRCLE; the Identifier cell
    reflects that as an int (EditRole) + label (DisplayRole)."""
    project, model, surf_index = _add_aperture_stop_project(qapp)

    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    shape_idx = model.index(0, int(Column.IDENTIFIER), surf_idx)

    assert model.data(shape_idx, Qt.EditRole) == int(ghostlight.ApertureShape.CIRCLE)
    assert model.data(shape_idx, Qt.DisplayRole) == "Circle"
    assert model.flags(shape_idx) & Qt.ItemIsEditable


def test_aperture_aspect_cell_is_editable_in_all_shapes(qapp):
    """aperture_aspect drives the bounding ellipse on every shape, so the
    Radius-column slot is always editable + scrubbable."""
    project, model, surf_index = _add_aperture_stop_project(qapp)

    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    aspect_idx = model.index(0, int(Column.RADIUS), surf_idx)

    assert model.flags(aspect_idx) & Qt.ItemIsEditable
    assert model.data(aspect_idx, Qt.EditRole) == pytest.approx(1.0)

    # Edit: round-trips onto the Surface struct.
    assert model.setData(aspect_idx, 1.5, Qt.EditRole)
    assert float(project.system.surfaces[surf_index].aperture_aspect) == \
        pytest.approx(1.5)


def test_polygon_only_cells_blank_and_uneditable_in_circle_mode(qapp):
    """Blade count + rotation cells are POLYGON-only — in CIRCLE they
    render blank, get returns None, and the flags don't include
    ItemIsEditable so the value-scrubber's Ctrl+MMB gate also rejects."""
    project, model, _surf_index = _add_aperture_stop_project(qapp)

    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    blades_idx = model.index(0, int(Column.POS_Z), surf_idx)
    rotation_idx = model.index(0, int(Column.APERTURE_RAD), surf_idx)

    assert model.data(blades_idx, Qt.DisplayRole) == ""
    assert model.data(blades_idx, Qt.EditRole) is None
    assert not (model.flags(blades_idx) & Qt.ItemIsEditable)

    assert model.data(rotation_idx, Qt.DisplayRole) == ""
    assert model.data(rotation_idx, Qt.EditRole) is None
    assert not (model.flags(rotation_idx) & Qt.ItemIsEditable)


def test_switching_shape_to_polygon_exposes_blades_and_rotation(qapp):
    """Setting aperture_shape = POLYGON flips the blade + rotation cells
    to editable; the discriminator write requires a model reset so the
    column flags refresh in lockstep with the underlying surface."""
    project, model, surf_index = _add_aperture_stop_project(qapp)

    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    shape_idx = model.index(0, int(Column.IDENTIFIER), surf_idx)

    assert model.setData(shape_idx, int(ghostlight.ApertureShape.POLYGON), Qt.EditRole)
    assert int(project.system.surfaces[surf_index].aperture_shape) == \
        int(ghostlight.ApertureShape.POLYGON)

    # Re-fetch after the model reset.
    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    blades_idx = model.index(0, int(Column.POS_Z), surf_idx)
    rotation_idx = model.index(0, int(Column.APERTURE_RAD), surf_idx)

    assert model.flags(blades_idx) & Qt.ItemIsEditable
    assert model.flags(rotation_idx) & Qt.ItemIsEditable
    # Shape-write seeds aperture_blades to a sensible polygon default
    # (covered by its own test below); rotation stays at the surface's
    # current value, which is 0 for a freshly-added stop.
    assert model.data(blades_idx, Qt.EditRole) >= 3
    assert model.data(rotation_idx, Qt.EditRole) == pytest.approx(0.0)


def test_aperture_blades_write_round_trips(qapp):
    """Blade count edits land on Surface.aperture_blades as an int."""
    project, model, surf_index = _add_aperture_stop_project(qapp)

    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    shape_idx = model.index(0, int(Column.IDENTIFIER), surf_idx)
    assert model.setData(shape_idx, int(ghostlight.ApertureShape.POLYGON), Qt.EditRole)

    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    blades_idx = model.index(0, int(Column.POS_Z), surf_idx)

    # Pick a value that differs from the polygon-seeded default so the
    # write is observable (the no-op detector refuses identical writes).
    assert model.setData(blades_idx, 9, Qt.EditRole)
    assert int(project.system.surfaces[surf_index].aperture_blades) == 9


def test_aperture_rotation_writes_degrees_stores_radians(qapp):
    """The Rotation cell shows / accepts degrees; storage on
    Surface.aperture_rotation_rad is in radians, matching the writer's
    .lens-format convention."""
    import math as _math

    project, model, surf_index = _add_aperture_stop_project(qapp)

    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    shape_idx = model.index(0, int(Column.IDENTIFIER), surf_idx)
    assert model.setData(shape_idx, int(ghostlight.ApertureShape.POLYGON), Qt.EditRole)

    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    rotation_idx = model.index(0, int(Column.APERTURE_RAD), surf_idx)

    assert model.setData(rotation_idx, 45.0, Qt.EditRole)
    assert float(project.system.surfaces[surf_index].aperture_rotation_rad) == \
        pytest.approx(_math.pi / 4.0)
    # Read-back reports the same degrees the user typed.
    assert model.data(rotation_idx, Qt.EditRole) == pytest.approx(45.0)


def test_aperture_form_slots_are_scrubbable_when_active(qapp):
    """All numeric slots (aspect, blades, rotation) opt into the value
    scrubber; the body's gate combines editability + the option flag,
    so polygon-only cells become scrubbable only after the discriminator
    flips to POLYGON."""
    from ghostlight_designer.optical_editor.delegates import SlotRole

    project, model, _surf_index = _add_aperture_stop_project(qapp)

    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    aspect_idx = model.index(0, int(Column.RADIUS), surf_idx)
    blades_idx = model.index(0, int(Column.POS_Z), surf_idx)
    rotation_idx = model.index(0, int(Column.APERTURE_RAD), surf_idx)

    aspect_slot = model.data(aspect_idx, SlotRole)
    blades_slot = model.data(blades_idx, SlotRole)
    rotation_slot = model.data(rotation_idx, SlotRole)
    assert aspect_slot is not None
    assert blades_slot is not None
    assert rotation_slot is not None
    assert aspect_slot.options.get("scrubbable") is True
    assert blades_slot.options.get("scrubbable") is True
    assert rotation_slot.options.get("scrubbable") is True


def test_aperture_aspect_zero_write_is_noop(qapp):
    """Aspect 0 would collapse the bounding ellipse to a line and the
    tessellator silently falls back to 1.0 — refuse the write so the
    UI value stays in sync with what the renderer actually uses."""
    project, model, surf_index = _add_aperture_stop_project(qapp)

    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    aspect_idx = model.index(0, int(Column.RADIUS), surf_idx)

    original = float(project.system.surfaces[surf_index].aperture_aspect)
    ok = model.setData(aspect_idx, 0.0, Qt.EditRole)
    assert not ok
    assert float(project.system.surfaces[surf_index].aperture_aspect) == \
        pytest.approx(original)


def test_aperture_shape_label_in_combo_is_human_readable(qapp):
    """The discriminator combo formats values as Circle / Polygon /
    Image (title case) rather than the bound enum's SCREAMING_CASE,
    which would look out of place next to the Pos Z / Aperture Rad
    user-facing labels. IMAGE still has a label so legacy lens files
    that pre-date the UI scope render correctly even though the user
    can't pick IMAGE from the dropdown."""
    from ghostlight_designer.optical_editor.row_schemas import _aperture_shape_label

    assert _aperture_shape_label(int(ghostlight.ApertureShape.CIRCLE)) == "Circle"
    assert _aperture_shape_label(int(ghostlight.ApertureShape.POLYGON)) == "Polygon"
    assert _aperture_shape_label(int(ghostlight.ApertureShape.IMAGE)) == "Image"


def test_aperture_shape_combo_excludes_image(qapp):
    """IMAGE is hidden from the dropdown — the writer's image-aperture
    round-trip is still open work, so allowing the user to pick IMAGE
    from the UI would leave the project in a non-savable state. Only
    CIRCLE and POLYGON should be selectable.

    Builds the editor through the schema's slot options so the test
    exercises the same `exclude` plumbing the production combo uses,
    not a hand-rolled reimplementation."""
    from PySide6.QtWidgets import QWidget
    from ghostlight_designer.optical_editor.delegates import SlotRole
    from ghostlight_designer.optical_editor.delegates import _EDITOR_OPS
    from ghostlight_designer.optical_editor.row_schemas import SlotEditor

    project, model, _surf_index = _add_aperture_stop_project(qapp)
    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    shape_idx = model.index(0, int(Column.IDENTIFIER), surf_idx)
    slot = model.data(shape_idx, SlotRole)
    assert slot is not None and slot.editor == SlotEditor.ENUM_COMBO

    parent = QWidget()
    try:
        editor = _EDITOR_OPS[SlotEditor.ENUM_COMBO].create(parent, slot)
        values = [editor.itemData(i) for i in range(editor.count())]
        names = [editor.itemText(i) for i in range(editor.count())]
        assert int(ghostlight.ApertureShape.CIRCLE) in values
        assert int(ghostlight.ApertureShape.POLYGON) in values
        assert int(ghostlight.ApertureShape.IMAGE) not in values
        assert "IMAGE" not in names
    finally:
        parent.deleteLater()


def test_switching_shape_to_polygon_seeds_default_blade_count(qapp):
    """A freshly-added stop carries aperture_blades = 0 (CIRCLE default).
    Flipping to POLYGON with blades still 0 would silently render as
    an ellipse — the tessellator's `blades >= 3` gate falls through.
    The shape-write seeds a sensible default (6, common iris) the first
    time the user enters polygon mode so the cell + viewport agree."""
    project, model, surf_index = _add_aperture_stop_project(qapp)
    assert int(project.system.surfaces[surf_index].aperture_blades) == 0

    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    shape_idx = model.index(0, int(Column.IDENTIFIER), surf_idx)
    assert model.setData(shape_idx, int(ghostlight.ApertureShape.POLYGON), Qt.EditRole)

    assert int(project.system.surfaces[surf_index].aperture_blades) >= 3

    # Re-fetch and confirm the blades cell mirrors the seeded value.
    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    blades_idx = model.index(0, int(Column.POS_Z), surf_idx)
    assert model.data(blades_idx, Qt.EditRole) >= 3


def test_polygon_shape_switch_preserves_existing_blade_count(qapp):
    """If the user has already set blades to an in-range value (>= 3)
    and toggles shape away from POLYGON and back, their choice survives.
    Only the C++ default 0 (or any below-floor value) gets re-seeded."""
    project, model, surf_index = _add_aperture_stop_project(qapp)

    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    shape_idx = model.index(0, int(Column.IDENTIFIER), surf_idx)
    assert model.setData(shape_idx, int(ghostlight.ApertureShape.POLYGON), Qt.EditRole)

    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    blades_idx = model.index(0, int(Column.POS_Z), surf_idx)
    assert model.setData(blades_idx, 8, Qt.EditRole)
    assert int(project.system.surfaces[surf_index].aperture_blades) == 8

    # POLYGON → CIRCLE → POLYGON: blade count survives.
    shape_idx = model.index(0, int(Column.IDENTIFIER),
                            model.index(0, int(Column.NAME),
                                        model.index(0, 0, QModelIndex())))
    assert model.setData(shape_idx, int(ghostlight.ApertureShape.CIRCLE), Qt.EditRole)
    shape_idx = model.index(0, int(Column.IDENTIFIER),
                            model.index(0, int(Column.NAME),
                                        model.index(0, 0, QModelIndex())))
    assert model.setData(shape_idx, int(ghostlight.ApertureShape.POLYGON), Qt.EditRole)
    assert int(project.system.surfaces[surf_index].aperture_blades) == 8


def test_aperture_blade_writes_clamp_to_min_3(qapp):
    """Writing a sub-3 blade count clamps up to the minimum rather than
    silently storing a value that renders as an ellipse. Spinbox UI also
    enforces this via options.min, but the model-level clamp covers the
    setData(EditRole) path used by the value scrubber + programmatic
    writes."""
    project, model, surf_index = _add_aperture_stop_project(qapp)

    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    shape_idx = model.index(0, int(Column.IDENTIFIER), surf_idx)
    assert model.setData(shape_idx, int(ghostlight.ApertureShape.POLYGON), Qt.EditRole)

    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    blades_idx = model.index(0, int(Column.POS_Z), surf_idx)

    # The shape switch already seeded 6 above the floor; force a value
    # the spinbox would otherwise rebuff and confirm the writer clamps.
    assert model.setData(blades_idx, 1, Qt.EditRole)
    assert int(project.system.surfaces[surf_index].aperture_blades) == 3


def test_aperture_blade_writes_clamp_to_max(qapp):
    """Symmetric to the lower clamp — writes past the ceiling cap at 32
    rather than letting the C++ tracer accept a degenerate count."""
    project, model, surf_index = _add_aperture_stop_project(qapp)

    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    shape_idx = model.index(0, int(Column.IDENTIFIER), surf_idx)
    assert model.setData(shape_idx, int(ghostlight.ApertureShape.POLYGON), Qt.EditRole)

    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    blades_idx = model.index(0, int(Column.POS_Z), surf_idx)

    assert model.setData(blades_idx, 999, Qt.EditRole)
    assert int(project.system.surfaces[surf_index].aperture_blades) == 32


def test_aperture_blade_spinbox_options_advertise_clamp_range(qapp):
    """The spinbox editor pulls its min/max from slot.options — verify
    those values match the model-level clamp so the UI never offers a
    value the writer would just snap back."""
    from ghostlight_designer.optical_editor.delegates import SlotRole

    project, model, _surf_index = _add_aperture_stop_project(qapp)
    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    shape_idx = model.index(0, int(Column.IDENTIFIER), surf_idx)
    assert model.setData(shape_idx, int(ghostlight.ApertureShape.POLYGON), Qt.EditRole)

    el_idx = model.index(0, 0, QModelIndex())
    surf_idx = model.index(0, int(Column.NAME), el_idx)
    blades_idx = model.index(0, int(Column.POS_Z), surf_idx)
    slot = model.data(blades_idx, SlotRole)
    assert slot is not None
    assert slot.options["min"] == 3
    assert slot.options["max"] == 32
