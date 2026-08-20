"""Tree-node wrappers for the optical-design editor.

Wrappers carry indices + UUIDs, never live C++ ``Surface`` refs, so a model
rebuild after ``Project.systemReplaced`` can't leave us with dangling
pointers. Live ``Surface`` instances are always resolved on read via
``project.system.surfaces[node.surface_index]``.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import ghostlight


class NodeKind(enum.IntEnum):
    ROOT = 0
    ELEMENT = 1
    MATERIAL = 2
    SURFACE = 3
    ASPHERE_FORM = 4
    CYLINDRICAL_FORM = 5
    APERTURE_FORM = 6
    COATING_FORM = 7
    BLADE_SHAPE_FORM = 8


@dataclass(eq=False)
class TreeNode:
    kind: NodeKind
    parent: Optional["TreeNode"] = None
    children: list["TreeNode"] = field(default_factory=list)
    icon_name: str = ""

    def row(self) -> int:
        if self.parent is None:
            return 0
        for i, child in enumerate(self.parent.children):
            if child is self:
                return i
        return 0


@dataclass(eq=False)
class RootNode(TreeNode):
    def __init__(self) -> None:
        super().__init__(kind=NodeKind.ROOT, parent=None, children=[], icon_name="")


@dataclass(eq=False)
class ElementNode(TreeNode):
    element: "ghostlight.Element" = None  # type: ignore[assignment]
    element_index: int = -1
    # Set by build_tree from Element.is_muted(system). Drives icon
    # dispatch and the italic FontRole hint.
    muted: bool = False

    def __init__(
        self,
        element: "ghostlight.Element",
        element_index: int,
        parent: TreeNode,
        *,
        muted: bool = False,
    ) -> None:
        import ghostlight

        if element.kind == ghostlight.ElementKind.STOP:
            icon_name = "node-stop"
        elif muted:
            icon_name = "node-element-muted"
        else:
            icon_name = "node-element"
        super().__init__(
            kind=NodeKind.ELEMENT,
            parent=parent,
            children=[],
            icon_name=icon_name,
        )
        self.element = element
        self.element_index = element_index
        self.muted = bool(muted)


@dataclass(eq=False)
class MaterialNode(TreeNode):
    element: "ghostlight.Element" = None  # type: ignore[assignment]
    material_index: int = -1

    def __init__(self, element: "ghostlight.Element", material_index: int, parent: TreeNode) -> None:
        super().__init__(
            kind=NodeKind.MATERIAL,
            parent=parent,
            children=[],
            icon_name="node-material",
        )
        self.element = element
        self.material_index = material_index


@dataclass(eq=False)
class SurfaceNode(TreeNode):
    surface_uuid: str = ""
    surface_index: int = -1
    element: "ghostlight.Element" = None  # type: ignore[assignment]
    # Set by build_tree from Project.ghost_solo_surface_uuids.
    # Drives the surf-solo icon glyph.
    ghost_solo: bool = False

    def __init__(
        self,
        surface_uuid: str,
        surface_index: int,
        element: "ghostlight.Element",
        parent: TreeNode,
        *,
        ghost_solo: bool = False,
    ) -> None:
        icon = "node-surface-solo" if ghost_solo else "node-surface"
        super().__init__(
            kind=NodeKind.SURFACE,
            parent=parent,
            children=[],
            icon_name=icon,
        )
        self.surface_uuid = surface_uuid
        self.surface_index = surface_index
        self.element = element
        self.ghost_solo = bool(ghost_solo)


@dataclass(eq=False)
class AsphereFormNode(TreeNode):
    surface_index: int = -1

    def __init__(self, surface_index: int, parent: TreeNode) -> None:
        super().__init__(
            kind=NodeKind.ASPHERE_FORM,
            parent=parent,
            children=[],
            icon_name="node-asphere",
        )
        self.surface_index = surface_index


@dataclass(eq=False)
class CylindricalFormNode(TreeNode):
    surface_index: int = -1

    def __init__(self, surface_index: int, parent: TreeNode) -> None:
        super().__init__(
            kind=NodeKind.CYLINDRICAL_FORM,
            parent=parent,
            children=[],
            icon_name="node-cylindrical",
        )
        self.surface_index = surface_index


@dataclass(eq=False)
class ApertureFormNode(TreeNode):
    surface_index: int = -1

    def __init__(self, surface_index: int, parent: TreeNode) -> None:
        super().__init__(
            kind=NodeKind.APERTURE_FORM,
            parent=parent,
            children=[],
            icon_name="node-aperture",
        )
        self.surface_index = surface_index


@dataclass(eq=False)
class BladeShapeFormNode(TreeNode):
    """Blade-shape controls for a bladed stop.

    A sibling of ApertureFormNode rather than more columns on it: the aperture
    row's four cells are already spoken for by shape / aspect / blades /
    rotation, and these four only exist while the shape is POLYGON.
    """

    surface_index: int = -1

    def __init__(self, surface_index: int, parent: TreeNode) -> None:
        super().__init__(
            kind=NodeKind.BLADE_SHAPE_FORM,
            parent=parent,
            children=[],
            icon_name="node-aperture",
        )
        self.surface_index = surface_index


@dataclass(eq=False)
class CoatingFormNode(TreeNode):
    surface_index: int = -1
    # Set by build_tree from row_schemas.coating_ui_model_for_surface — a
    # CoatingModelUI int. The coating row packs only the slots its model
    # actually uses into consecutive columns, and ``row_schemas.slot_at``
    # takes a node, not a system, so the discriminator is baked here.
    # Every path that can change it (Model combo, Preset picker, the surface
    # right-click apply/remove) rebuilds the tree, so it can't go stale.
    coating_ui_model: Optional[int] = None

    def __init__(
        self,
        surface_index: int,
        parent: TreeNode,
        *,
        coating_ui_model: Optional[int] = None,
    ) -> None:
        super().__init__(
            kind=NodeKind.COATING_FORM,
            parent=parent,
            children=[],
            icon_name="node-coating",
        )
        self.surface_index = surface_index
        self.coating_ui_model = coating_ui_model


def surface_uuid_for(node: TreeNode) -> str:
    """Return the surface UUID this ``node`` sits under, or ``""``.

    * ``SurfaceNode`` — its own ``surface_uuid``.
    * ``AsphereFormNode`` / ``CylindricalFormNode`` / ``ApertureFormNode`` /
      ``CoatingFormNode`` — their parent's ``surface_uuid``.
    * Anything else (element, material, root) — ``""``.

    Used by :mod:`ghostlight_designer.optical_editor.delegates` to look up
    the row's variable-flag state on the project and by
    ``element_actions`` / ``surface_actions`` handlers that need to key
    on UUID rather than surface index.
    """
    if node is None:
        return ""
    if isinstance(node, SurfaceNode):
        return node.surface_uuid or ""
    parent = getattr(node, "parent", None)
    if isinstance(parent, SurfaceNode):
        return parent.surface_uuid or ""
    return ""


def build_tree(system, *, solo_uuids: frozenset | set | None = None) -> RootNode:
    """Walk a ``ghostlight.OpticalSystem`` and produce a wrapper-node tree.

    ``solo_uuids`` is the optional set of surface UUIDs the user has
    marked ghost-solo (sourced from ``Project.ghost_solo_surface_uuids``).
    Surface rows whose UUID is in the set get the solo glyph + flag.
    None / empty produces a tree where no surface is solo'd; that
    matches the default-not-yet-loaded state of a Project.
    """
    import ghostlight

    from .coating_actions import surface_has_coating as _surface_has_coating
    # Deferred: row_schemas imports this module at module scope, so the
    # dependency only closes at call time.
    from .row_schemas import (
        coating_ui_model_for_surface as _coating_ui_model_for_surface,
    )

    root = RootNode()
    elements = list(system.elements) if system is not None else []
    if not elements:
        return root

    uuid_to_index = {uuid: i for i, uuid in enumerate(system.surface_ids)}
    surfaces = system.surfaces
    solo_set = set(solo_uuids) if solo_uuids else set()

    for ei, element in enumerate(elements):
        # AttributeError guards against Element-like test stubs.
        try:
            muted = bool(element.is_muted(system))
        except AttributeError:
            muted = False
        el_node = ElementNode(
            element=element, element_index=ei, parent=root, muted=muted,
        )
        root.children.append(el_node)

        if element.kind == ghostlight.ElementKind.GLASS:
            for mi in range(len(element.material_glasses)):
                el_node.children.append(
                    MaterialNode(element=element, material_index=mi, parent=el_node)
                )

        for uuid in element.surface_ids:
            surface_index = uuid_to_index.get(uuid, -1)
            surf_node = SurfaceNode(
                surface_uuid=uuid,
                surface_index=surface_index,
                element=element,
                parent=el_node,
                ghost_solo=uuid in solo_set,
            )
            el_node.children.append(surf_node)

            if surface_index >= 0 and 0 <= surface_index < len(surfaces):
                surface = surfaces[surface_index]
                form = int(surface.form)
                if form == int(ghostlight.SurfaceForm.ASPHERE):
                    surf_node.children.append(
                        AsphereFormNode(surface_index=surface_index, parent=surf_node)
                    )
                elif form == int(ghostlight.SurfaceForm.CYLINDRICAL):
                    surf_node.children.append(
                        CylindricalFormNode(surface_index=surface_index, parent=surf_node)
                    )
                is_stop = bool(getattr(surface, "is_stop", False))
                if is_stop:
                    surf_node.children.append(
                        ApertureFormNode(surface_index=surface_index, parent=surf_node)
                    )
                    # Blade shape exists only on a bladed stop. A shape change
                    # rebuilds the tree (the aperture row's write returns
                    # requires_reset), so this can't go stale.
                    if int(getattr(surface, "aperture_shape", 0)) == int(
                        ghostlight.ApertureShape.POLYGON
                    ):
                        surf_node.children.append(
                            BladeShapeFormNode(
                                surface_index=surface_index, parent=surf_node
                            )
                        )
                # The coating row appears only when a coating is actually
                # applied, and never on an aperture stop (a stop bounds the
                # pupil — it isn't a coatable optical interface). Appended LAST
                # so the optional form/aperture child rows keep their positions.
                # A bare surface stays a leaf until the user applies a coating
                # (via the surface's right-click "Coating" preset menu).
                if not is_stop and _surface_has_coating(surface):
                    surf_node.children.append(
                        CoatingFormNode(
                            surface_index=surface_index,
                            parent=surf_node,
                            coating_ui_model=_coating_ui_model_for_surface(
                                system, surface_index
                            ),
                        )
                    )

    return root
