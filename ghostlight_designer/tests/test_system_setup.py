from __future__ import annotations

from PySide6.QtCore import QModelIndex, Qt

from ghostlight_designer.project import Project
from ghostlight_designer.system_setup import SystemSetupBody, SYSTEM_SETUP_TYPE_ID
from ghostlight_designer.system_setup.columns import Column
from ghostlight_designer.system_setup.model import SystemSetupTreeModel
from ghostlight_designer.system_setup.nodes import (
    CategoryNode,
    DistributionFieldNode,
    DistributionProp,
    FieldFieldNode,
    FieldNode,
    FieldProp,
    SensorProp,
    SensorPropNode,
    SequenceFieldNode,
    SequenceNode,
    SequenceProp,
    SourceFieldNode,
    SourceNode,
    SourceProp,
    WavelengthNode,
    WavelengthsFieldNode,
    WavelengthsNode,
    WavelengthsProp,
)
from ghostlight_designer.system_setup_data import (
    ApertureType,
    CUSTOM_PRESET,
    DistributionType,
    FieldType,
    SourceType,
    SystemSetup,
    find_preset,
)


def _collect(signal):
    received: list = []

    def slot(*args):
        received.append(args if len(args) != 1 else args[0])

    signal.connect(slot)
    return received


# ---------------------------------------------------------------------------
# Defaults & lifecycle
# ---------------------------------------------------------------------------


def test_default_system_setup_has_one_sequence_and_defaults(qapp):
    setup = Project().system_setup
    assert isinstance(setup, SystemSetup)
    assert len(setup.sequences) == 1
    seq = setup.sequences[0]
    assert seq.name == "Auto Sequence 1"
    assert seq.aperture_type == ApertureType.FROM_STOP
    assert seq.field_type == FieldType.ANGLE
    assert seq.stop_surface is None
    assert seq.source.type == SourceType.PLANE_WF
    assert seq.source.aperture_radius == 20.0
    assert seq.source.distribution.type == DistributionType.Y_FAN
    assert seq.source.distribution.ray_count == 8
    wc = seq.source.wavelengths
    assert [round(w.value_nm, 2) for w in wc.wavelengths] == [486.13, 587.56, 656.27]
    assert wc.primary_index == 1
    assert wc.reference_index is None
    names = [f.name for f in seq.source.fields]
    assert names == ["Axial"]


def test_new_resets_setup_and_emits_signal(qapp):
    project = Project()
    project.system_setup.sequences[0].name = "Mutated"
    received = _collect(project.systemSetupChanged)
    project.new()
    assert received
    assert project.system_setup.sequences[0].name == "Auto Sequence 1"


def test_load_resets_setup(qapp, sample_lens_path):
    project = Project()
    project.system_setup.sequences[0].source.fields[0].name = "Mutated"
    received = _collect(project.systemSetupChanged)
    project.load(str(sample_lens_path))
    assert received
    assert project.system_setup.sequences[0].source.fields[0].name == "Axial"


def test_mark_system_setup_modified_does_not_dirty(qapp):
    project = Project()
    dirty_received = _collect(project.dirtyChanged)
    setup_received = _collect(project.systemSetupChanged)
    project.system_setup.sequences[0].source.aperture_radius = 25.0
    project.mark_system_setup_modified()
    assert setup_received
    assert not dirty_received
    assert project.is_dirty is False


# ---------------------------------------------------------------------------
# Tree shape
# ---------------------------------------------------------------------------


def test_tree_top_level_is_sequences_and_sensor(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    assert model.columnCount() == len(Column)
    assert model.rowCount(QModelIndex()) == 2
    sensor_cat_idx = model.index(0, 0, QModelIndex())
    seq_cat_idx = model.index(1, 0, QModelIndex())
    assert model.data(sensor_cat_idx, Qt.DisplayRole) == "Image Sensor"
    assert model.data(seq_cat_idx, Qt.DisplayRole) == "Sequences"


def test_tree_under_sequence_has_aperture_field_stop_source(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    seq_cat_idx = model.index(1, 0, QModelIndex())
    seq_idx = model.index(0, 0, seq_cat_idx)
    assert model.data(seq_idx, Qt.DisplayRole) == "Auto Sequence 1"
    labels = [
        model.data(model.index(r, 0, seq_idx), Qt.DisplayRole)
        for r in range(model.rowCount(seq_idx))
    ]
    assert labels == ["Aperture Type", "Field Type", "Stop Surface", "Source"]


def test_tree_under_source_has_type_radius_dist_waves_fields(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    seq_cat_idx = model.index(1, 0, QModelIndex())
    seq_idx = model.index(0, 0, seq_cat_idx)
    source_idx = model.index(3, 0, seq_idx)
    assert model.data(source_idx, Qt.DisplayRole) == "Source"
    labels = [
        model.data(model.index(r, 0, source_idx), Qt.DisplayRole)
        for r in range(model.rowCount(source_idx))
    ]
    assert labels == [
        "Source Type",
        "Aperture Radius",
        "Distribution",
        "Wavelengths",
        "Fields",
    ]


def test_wavelengths_have_primary_reference_and_three_entries(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    waves_idx = _find_index(model, ["Sequences", "Auto Sequence 1", "Source", "Wavelengths"])
    labels = [
        model.data(model.index(r, 0, waves_idx), Qt.DisplayRole)
        for r in range(model.rowCount(waves_idx))
    ]
    assert labels[:2] == ["Primary", "Reference"]
    assert labels[2:] == ["Wavelength 1", "Wavelength 2", "Wavelength 3"]


def test_fields_have_default_axial_with_tilt_props(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    fields_idx = _find_index(model, ["Sequences", "Auto Sequence 1", "Source", "Fields"])
    assert model.rowCount(fields_idx) == 1
    f_idx = model.index(0, 0, fields_idx)
    labels = [
        model.data(model.index(rr, 0, f_idx), Qt.DisplayRole)
        for rr in range(model.rowCount(f_idx))
    ]
    assert labels == ["Tilt X (°)", "Tilt Y (°)"]


def _find_index(model, path):
    """Walk the tree by name-column labels."""
    cur = QModelIndex()
    for label in path:
        n = model.rowCount(cur)
        found = None
        for r in range(n):
            child = model.index(r, 0, cur)
            if model.data(child, Qt.DisplayRole) == label:
                found = child
                break
        assert found is not None, f"missing tree node: {label} under {path}"
        cur = found
    return cur


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------


def _val_index(model, path):
    name_idx = _find_index(model, path)
    return model.index(name_idx.row(), int(Column.VALUE), name_idx.parent())


def test_sequence_name_edit(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    idx = _val_index(model, ["Sequences", "Auto Sequence 1"])
    assert model.setData(idx, "Reflex 1", Qt.EditRole)
    assert project.system_setup.sequences[0].name == "Reflex 1"


def test_aperture_type_enum_edit(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    idx = _val_index(model, ["Sequences", "Auto Sequence 1", "Aperture Type"])
    assert model.setData(idx, "None", Qt.EditRole)
    assert project.system_setup.sequences[0].aperture_type == ApertureType.NONE


def test_field_type_enum_edit(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    idx = _val_index(model, ["Sequences", "Auto Sequence 1", "Field Type"])
    assert model.setData(idx, "Free", Qt.EditRole)
    assert project.system_setup.sequences[0].field_type == FieldType.FREE


def test_stop_surface_set_and_auto(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    idx = _val_index(model, ["Sequences", "Auto Sequence 1", "Stop Surface"])
    assert model.setData(idx, 3, Qt.EditRole)
    assert project.system_setup.sequences[0].stop_surface == 3
    assert model.setData(idx, "Auto", Qt.EditRole)
    assert project.system_setup.sequences[0].stop_surface is None


def test_source_type_and_aperture_radius_edit(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    st = _val_index(model, ["Sequences", "Auto Sequence 1", "Source", "Source Type"])
    ar = _val_index(model, ["Sequences", "Auto Sequence 1", "Source", "Aperture Radius"])
    assert model.setData(st, "Point Source", Qt.EditRole)
    assert project.system_setup.sequences[0].source.type == SourceType.POINT_SOURCE
    assert model.setData(ar, 12.5, Qt.EditRole)
    assert project.system_setup.sequences[0].source.aperture_radius == 12.5


def test_distribution_edits(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    type_idx = _val_index(
        model, ["Sequences", "Auto Sequence 1", "Source", "Distribution", "Type"]
    )
    rc_idx = _val_index(
        model, ["Sequences", "Auto Sequence 1", "Source", "Distribution", "Ray Count"]
    )
    assert model.setData(type_idx, "Ring", Qt.EditRole)
    assert (
        project.system_setup.sequences[0].source.distribution.type
        == DistributionType.RING
    )
    assert model.setData(rc_idx, 32, Qt.EditRole)
    assert project.system_setup.sequences[0].source.distribution.ray_count == 32


def test_distribution_ray_count_rejects_zero_and_negative(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    rc_idx = _val_index(
        model, ["Sequences", "Auto Sequence 1", "Source", "Distribution", "Ray Count"]
    )
    assert not model.setData(rc_idx, 0, Qt.EditRole)
    assert not model.setData(rc_idx, -5, Qt.EditRole)
    assert project.system_setup.sequences[0].source.distribution.ray_count == 8


def test_distribution_row_has_only_type_and_ray_count(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    dist_idx = _find_index(
        model, ["Sequences", "Auto Sequence 1", "Source", "Distribution"]
    )
    labels = [
        model.data(model.index(r, 0, dist_idx), Qt.DisplayRole)
        for r in range(model.rowCount(dist_idx))
    ]
    assert labels == ["Type", "Ray Count"]


def test_wavelength_primary_pick(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    primary_idx = _val_index(
        model, ["Sequences", "Auto Sequence 1", "Source", "Wavelengths", "Primary"]
    )
    assert model.setData(primary_idx, "Wavelength 3", Qt.EditRole)
    assert project.system_setup.sequences[0].source.wavelengths.primary_index == 2


def test_wavelength_reference_primary_and_specific(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    ref_idx = _val_index(
        model, ["Sequences", "Auto Sequence 1", "Source", "Wavelengths", "Reference"]
    )
    assert model.setData(ref_idx, "Wavelength 1", Qt.EditRole)
    assert project.system_setup.sequences[0].source.wavelengths.reference_index == 0
    assert model.setData(ref_idx, "Primary", Qt.EditRole)
    assert project.system_setup.sequences[0].source.wavelengths.reference_index is None


def test_wavelength_value_edit(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    w1_idx = _val_index(
        model,
        ["Sequences", "Auto Sequence 1", "Source", "Wavelengths", "Wavelength 1"],
    )
    assert model.setData(w1_idx, 450.0, Qt.EditRole)
    assert (
        project.system_setup.sequences[0]
        .source.wavelengths.wavelengths[0]
        .value_nm
        == 450.0
    )


def test_field_name_and_tilt_edits(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    axial_idx = _val_index(
        model, ["Sequences", "Auto Sequence 1", "Source", "Fields", "Axial"]
    )
    assert model.setData(axial_idx, "On-Axis", Qt.EditRole)
    assert project.system_setup.sequences[0].source.fields[0].name == "On-Axis"
    ty_idx = _val_index(
        model,
        [
            "Sequences",
            "Auto Sequence 1",
            "Source",
            "Fields",
            "On-Axis",
            "Tilt Y (°)",
        ],
    )
    assert model.setData(ty_idx, 5.0, Qt.EditRole)
    assert project.system_setup.sequences[0].source.fields[0].tilt_y_deg == 5.0


# ---------------------------------------------------------------------------
# Sensor (unchanged behavior carried forward)
# ---------------------------------------------------------------------------


def test_sensor_preset_pick_rewrites_mm(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    preset_idx = _val_index(model, ["Image Sensor", "Preset"])
    width_idx = _val_index(model, ["Image Sensor", "Width (mm)"])
    height_idx = _val_index(model, ["Image Sensor", "Height (mm)"])
    assert model.setData(preset_idx, "Full Frame", Qt.EditRole)
    ff = find_preset("Full Frame")
    assert project.system_setup.sensor.preset_name == "Full Frame"
    assert project.system_setup.sensor.width_mm == ff.width_mm
    assert float(model.data(width_idx, Qt.EditRole)) == ff.width_mm
    assert float(model.data(height_idx, Qt.EditRole)) == ff.height_mm


def test_sensor_mm_edit_flips_preset_to_custom(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    height_idx = _val_index(model, ["Image Sensor", "Height (mm)"])
    preset_idx = _val_index(model, ["Image Sensor", "Preset"])
    assert model.setData(height_idx, 25.0, Qt.EditRole)
    assert project.system_setup.sensor.preset_name == CUSTOM_PRESET
    assert model.data(preset_idx, Qt.DisplayRole) == CUSTOM_PRESET


# ---------------------------------------------------------------------------
# Body + panel type
# ---------------------------------------------------------------------------


def test_body_constructs_and_expands(qapp):
    project = Project()
    body = SystemSetupBody(project)
    sensor_cat_idx = body.model.index(0, 0, QModelIndex())
    seq_cat_idx = body.model.index(1, 0, QModelIndex())
    assert body.tree.isExpanded(sensor_cat_idx)
    assert body.tree.isExpanded(seq_cat_idx)
    body.deleteLater()


def test_panel_type_registers(qapp):
    from ghostlight_designer.panel_system import registry
    from ghostlight_designer.system_setup import register_system_setup_panel_type

    register_system_setup_panel_type()
    t = registry.get(SYSTEM_SETUP_TYPE_ID)
    assert t is not None
    assert t.display_name == "System Setup"
