"""In-memory document model wrapping ``ghostlight.OpticalSystem``.

Widgets observe this object:

* connect to ``systemReplaced`` to rebuild caches when a New / Open / Undo
  swaps or reloads the underlying ``OpticalSystem``,
* connect to ``systemModified`` to refresh incrementally when the same
  ``OpticalSystem`` mutates in place,
* wrap mutations in ``project.edit(label)`` (or ``begin_compound`` /
  ``end_compound`` for coalesced drags) so they participate in undo/redo.
  Direct callers of :meth:`mark_modified` still work but are not undoable —
  reserve that escape hatch for non-user-driven changes.
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional

from PySide6.QtCore import QObject, Signal

import ghostlight
from ghostlight.writer import build_optical_system_doc

from .system_setup_data import SystemSetup


@dataclass(frozen=True)
class VariableBounds:
    """Optional min / max clamp for a surface attribute flagged as variable.

    ``None`` on either side means "unbounded on that side" — the optimizer
    translates that into ±inf when handing off to scipy.optimize.least_squares.
    The default ``VariableBounds()`` is a fully-unbounded flag, which is the
    default when the user toggles a cell to variable without opening the
    Bounds dialog.
    """
    lo: Optional[float] = None
    hi: Optional[float] = None

    def is_unbounded(self) -> bool:
        return self.lo is None and self.hi is None


@dataclass(frozen=True)
class SubstitutionSpec:
    """Material-substitution flag for one ``(element, material_index)`` pair.

    Only the target ``vendor`` is stored; the hammer optimizer resolves
    candidate glasses at run time from
    :mod:`ghostlight_designer.material_substitution`. Optional nd/Vd bounds
    restrict the search region — ``None`` on any side means "use the
    vendor's convex hull".
    """
    vendor: str = ""
    nd_lo: Optional[float] = None
    nd_hi: Optional[float] = None
    vd_lo: Optional[float] = None
    vd_hi: Optional[float] = None

# Project doesn't need to know what a MeritFunction looks like beyond "list of
# opaque objects keyed by mf_id"; the import exists for type checkers only.
if TYPE_CHECKING:
    from .optimization_panel.data import MeritFunction


_DEFAULT_BUDGET_BYTES = 256 * 1024 * 1024


class EditHandle:
    """Returned by :meth:`Project.edit`. Call :meth:`abort` to skip the
    undo-entry push for a no-op mutation."""

    __slots__ = ("aborted",)

    def __init__(self) -> None:
        self.aborted: bool = False

    def abort(self) -> None:
        self.aborted = True


class Project(QObject):
    systemReplaced = Signal(object)
    systemModified = Signal()
    systemSetupChanged = Signal()
    dirtyChanged = Signal(bool)
    pathChanged = Signal(object)
    canUndoChanged = Signal(bool)
    canRedoChanged = Signal(bool)
    # Currently-selected ``ghostlight.Element`` (or ``None``). Panels both read and
    # write via :meth:`set_selected_element` to stay in sync.
    selectionChanged = Signal(object)
    # Currently-selected global surface index (int into ``system.surfaces``)
    # or ``None``.  Always paired with a ``selected_element`` — the surface
    # belongs to that element — but the signals are independent so listeners
    # can subscribe to whichever granularity they care about.
    surfaceSelectionChanged = Signal(object)
    # The set of surface UUIDs marked "ghost-solo" by the user changed.
    # Ghost-solo is a designer-only debugging affordance: render panels
    # build an `ghostlight.GhostFilter` from this set so only ghosts where a
    # solo'd surface participates are drawn. In-memory only (not in .lens);
    # cleared on systemReplaced.
    ghostSoloChanged = Signal()
    # Merit-function list mutated in place (goal added/removed, target/weight
    # edited, etc.) — listeners refresh their displayed values without
    # rebuilding the whole tree.
    meritFunctionsChanged = Signal()
    # The merit_functions list itself was replaced (load, undo, etc.) —
    # listeners rebuild from scratch.
    meritFunctionsReplaced = Signal()
    # A single variable-flag entry changed (toggle / bounds edit) — cell
    # delegates repaint the affected cells without a model reset. Payload:
    # (surface_uuid, attr) so listeners can be narrow if they want to be.
    variableFlagChanged = Signal(str, str)
    # The whole variable-flag map was replaced (new / load, bulk clear).
    # Listeners repaint everything from scratch.
    variableFlagsReplaced = Signal()
    # A single material-substitution flag changed. Payload:
    # (element_id, material_index) — mirror of variableFlagChanged for
    # the catalogue-hammer material-substitution flow.
    materialFlagChanged = Signal(str, int)
    # Bulk / structural change to the material-flag map (new / load,
    # clear-all, pruning after an element was removed).
    materialFlagsReplaced = Signal()
    # An element's "Off Axis" reveal was toggled in the optical editor. The
    # tree hides / shows the Pos X..Pivot Z column block in response. Purely
    # view state — the pose VALUES are lens geometry and round-trip through
    # .lens, but which columns are on screen does not.
    offAxisRevealChanged = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._system: ghostlight.OpticalSystem = ghostlight.OpticalSystem()
        self._system_setup: SystemSetup = SystemSetup()
        self._path: Optional[str] = None
        self._dirty: bool = False
        self._selected_element: Optional[ghostlight.Element] = None
        self._selected_surface_index: Optional[int] = None
        # In-memory set of surface UUIDs marked "ghost-solo". Render
        # panels read this to build their FlareConfig.ghost_filter. Not
        # persisted to .lens (it's a view filter, not lens geometry);
        # cleared on systemReplaced so a fresh document starts clean.
        self._ghost_solo_surface_uuids: set[str] = set()
        # Element IDs whose off-axis columns the user has explicitly revealed.
        # This is only the EXPLICIT half of the state: an element holding a
        # non-zero pose value counts as revealed regardless of membership here
        # (see ``optical_editor.row_schemas.element_has_off_axis_value``), so
        # decenter / tilt can never be hidden behind a collapsed column.
        # In-memory only; cleared on systemReplaced.
        self._off_axis_revealed_element_ids: set[str] = set()

        # Merit functions are project-only / in-memory (matches SystemSetup);
        # not persisted to .lens and not on the undo stack. The
        # *result* of running one is what hits undo, via project.edit(...).
        self._merit_functions: list = []

        # Variable-flag map: {surface_uuid -> {attr_name -> VariableBounds}}.
        # Only surfaces with at least one flagged attribute have an entry.
        # In-memory only (matches merit_functions and ghost_solo); cleared
        # on new / load, pruned to live UUIDs on modify. Read by the
        # optimizer via :func:`optimization_panel.variables.collect_variables`.
        self._variable_flags: dict[str, dict[str, VariableBounds]] = {}

        # Material-substitution flag map:
        # {element_id -> {material_index -> SubstitutionSpec}}.
        # Only elements with at least one flagged material have an entry.
        # Same lifecycle as _variable_flags. When any entry is present the
        # optimizer switches to the catalogue-hammer path
        # (:mod:`.optimization_panel.optimizer`).
        self._material_flags: dict[str, dict[int, SubstitutionSpec]] = {}

        self._undo: list[tuple[str, dict]] = []
        self._redo: list[tuple[str, dict]] = []
        self._compound_depth: int = 0
        self._compound_before: Optional[tuple[str, dict]] = None
        self._snapshot_budget_bytes: int = _DEFAULT_BUDGET_BYTES
        self._snapshot_sizes: list[int] = []
        self._redo_sizes: list[int] = []
        self._last_can_undo: bool = False
        self._last_can_redo: bool = False
        # Why the most recent ``systemReplaced`` fired: ``"load"`` for a
        # user-initiated New / Open (viewport may refit camera), or
        # ``"restore"`` for an internal snapshot reload via undo / redo
        # (viewport should preserve the camera so the user can compare
        # before / after states without losing their framing).
        self._last_replacement_kind: str = "load"

    @property
    def system(self) -> ghostlight.OpticalSystem:
        return self._system

    @property
    def last_replacement_kind(self) -> str:
        """``"load"`` if the most recent ``systemReplaced`` came from a user
        New / Open, ``"restore"`` if it came from undo / redo. Viewports
        check this to decide whether to refit the camera — refit on load,
        preserve framing on restore."""
        return self._last_replacement_kind

    @property
    def system_setup(self) -> SystemSetup:
        return self._system_setup

    def mark_system_setup_modified(self) -> None:
        """Widgets call this after mutating ``self.system_setup`` in place.

        SystemSetup is project-only / in-memory: it does not mark the
        project dirty and does not participate in undo/redo.
        """
        self.systemSetupChanged.emit()

    # ------------------------------------------------------------------
    # Merit functions — same in-memory pattern as system_setup
    # ------------------------------------------------------------------

    @property
    def merit_functions(self) -> list:
        """Live list of merit functions.

        Callers may mutate this in place; pair every mutation with one of
        :meth:`mark_merit_functions_modified` (value tweak, goal cached)
        or :meth:`mark_merit_functions_replaced` (add/remove/reorder of
        merit functions or goals) so observers know whether they can do a
        narrow refresh or have to rebuild.
        """
        return self._merit_functions

    def mark_merit_functions_modified(self) -> None:
        """Emit after editing a goal's target / weight / params or after
        a Run finished updating cached values. Does NOT mark the project
        dirty — merit functions are in-memory, not part of the .lens."""
        self.meritFunctionsChanged.emit()

    def mark_merit_functions_replaced(self) -> None:
        """Emit after adding/removing a merit function or goal, or after
        replacing the whole list. Listeners rebuild from scratch."""
        self.meritFunctionsReplaced.emit()

    @property
    def path(self) -> Optional[str]:
        return self._path

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def display_name(self) -> str:
        base = os.path.basename(self._path) if self._path else "Untitled"
        return f"{base}*" if self._dirty else base

    def new(self) -> None:
        self._system = ghostlight.OpticalSystem()
        self._system_setup = SystemSetup()
        self._merit_functions = []
        path_changed = self._path is not None
        self._path = None
        self._set_dirty(False)
        self._clear_history()
        self._clear_selection_if_any()
        self._drop_ghost_solo_silent()
        self._drop_off_axis_reveal_silent()
        self._drop_variable_flags_silent()
        self._drop_material_flags_silent()
        self._last_replacement_kind = "load"
        self.systemReplaced.emit(self._system)
        self.systemSetupChanged.emit()
        self.meritFunctionsReplaced.emit()
        self.variableFlagsReplaced.emit()
        self.materialFlagsReplaced.emit()
        if path_changed:
            self.pathChanged.emit(None)

    def load(self, path: str) -> None:
        system = ghostlight.OpticalSystem.load(path)
        self._system = system
        self._system_setup = SystemSetup()
        self._merit_functions = []
        self._path = os.fspath(path)
        self._set_dirty(False)
        self._drop_ghost_solo_silent()
        self._drop_off_axis_reveal_silent()
        self._drop_variable_flags_silent()
        self._drop_material_flags_silent()
        self._clear_history()
        self._clear_selection_if_any()
        self._last_replacement_kind = "load"
        self.systemReplaced.emit(self._system)
        self.systemSetupChanged.emit()
        self.meritFunctionsReplaced.emit()
        self.variableFlagsReplaced.emit()
        self.materialFlagsReplaced.emit()
        self.pathChanged.emit(self._path)

    def save(self) -> None:
        if self._path is None:
            raise ValueError("Project.save(): no path set; use save_as")
        self._system.save(self._path)
        self._set_dirty(False)

    def save_as(self, path: str) -> None:
        target = os.fspath(path)
        self._system.save(target)
        path_changed = target != self._path
        self._path = target
        self._set_dirty(False)
        if path_changed:
            self.pathChanged.emit(self._path)

    def mark_modified(self) -> None:
        """Widgets call this after mutating ``self.system`` in place.

        Mutations made through :meth:`edit` / :meth:`begin_compound` already
        invoke this internally — calling it twice is harmless (just a second
        ``systemModified`` emission)."""
        self.systemModified.emit()
        self._set_dirty(True)
        self._validate_selection_against_system()
        self._prune_dead_ghost_solo()
        self._prune_dead_variable_flags()
        self._prune_dead_material_flags()

    def _prune_dead_ghost_solo(self) -> None:
        """Drop solo UUIDs that no longer match any surface in the system.

        Called after every mutation so removing an element with a solo'd
        surface (or undoing back across such an addition) cleans up
        without leaving stale UUIDs that would silently fail to filter
        anything at render time. Emits :attr:`ghostSoloChanged` only if
        the set actually shrinks."""
        if not self._ghost_solo_surface_uuids:
            return
        live = set(self._system.surface_ids)
        survivors = self._ghost_solo_surface_uuids & live
        if survivors == self._ghost_solo_surface_uuids:
            return
        self._ghost_solo_surface_uuids = survivors
        self.ghostSoloChanged.emit()

    def _prune_dead_variable_flags(self) -> None:
        """Drop flag entries for surfaces that no longer exist.

        Symmetric with :meth:`_prune_dead_ghost_solo`. Emits
        :attr:`variableFlagsReplaced` when anything was dropped so cell
        delegates repaint the affected column."""
        if not self._variable_flags:
            return
        live = set(self._system.surface_ids)
        dead = [uuid for uuid in self._variable_flags if uuid not in live]
        if not dead:
            return
        for uuid in dead:
            self._variable_flags.pop(uuid, None)
        self.variableFlagsReplaced.emit()

    # ------------------------------------------------------------------
    # Selection — element + optional surface
    # ------------------------------------------------------------------

    @property
    def selected_element(self) -> Optional["ghostlight.Element"]:
        return self._selected_element

    @property
    def selected_surface_index(self) -> Optional[int]:
        return self._selected_surface_index

    def set_selected_element(self, element: Optional["ghostlight.Element"]) -> None:
        """Set the project-wide selected element, emitting ``selectionChanged``.

        ``element`` must be ``None`` or an instance from
        ``self.system.elements`` — passing an Element that doesn't belong to
        the current system clears the selection (panels can hand off
        whatever instance they have; the project canonicalizes).

        Surface selection is **not** auto-cleared: panels often want to keep
        a surface highlighted while the element selection naturally tracks
        it.  Call :meth:`set_selected_surface_index` (with ``None``) to drop
        surface selection explicitly, or :meth:`set_selection` to set both
        atomically.
        """
        if element is not None and element not in self._system.elements:
            element = None
        if element is self._selected_element:
            return
        self._selected_element = element
        self.selectionChanged.emit(element)
        # Re-validate the surface — it must belong to the new element to
        # remain meaningful.
        self._validate_surface_against_element()

    def set_selected_surface_index(self, surface_index: Optional[int]) -> None:
        """Set the project-wide selected surface index.

        ``surface_index`` must index ``self.system.surfaces`` and belong to
        the currently-selected element; otherwise it is coerced to ``None``.
        """
        coerced = self._coerce_surface_index(surface_index)
        if coerced == self._selected_surface_index:
            return
        self._selected_surface_index = coerced
        self.surfaceSelectionChanged.emit(coerced)

    def set_selection(
        self,
        element: Optional["ghostlight.Element"],
        surface_index: Optional[int] = None,
    ) -> None:
        """Set element + surface atomically.

        Useful when the viewport picks an element+surface pair in one click —
        emits ``selectionChanged`` first (so a tree listener can re-parent
        before surface resolution) then ``surfaceSelectionChanged``.
        """
        self.set_selected_element(element)
        self.set_selected_surface_index(surface_index)

    def _coerce_surface_index(self, surface_index: Optional[int]) -> Optional[int]:
        if surface_index is None:
            return None
        try:
            si = int(surface_index)
        except (TypeError, ValueError):
            return None
        if si < 0 or si >= len(self._system.surfaces):
            return None
        el = self._selected_element
        if el is None:
            return None
        try:
            indices = el.resolve_surfaces(self._system)
        except (AttributeError, KeyError):
            return None
        return si if si in indices else None

    # ------------------------------------------------------------------
    # Ghost-solo — per-surface "show only my ghosts" filter
    # ------------------------------------------------------------------

    @property
    def ghost_solo_surface_uuids(self) -> frozenset:
        """Read-only snapshot of which surfaces are currently solo'd.

        Returned as a frozenset so callers can iterate without worrying
        about concurrent mutation; the Project keeps a mutable set
        internally for fast membership checks.
        """
        return frozenset(self._ghost_solo_surface_uuids)

    def is_surface_ghost_solo(self, surface_uuid: str) -> bool:
        return bool(surface_uuid) and surface_uuid in self._ghost_solo_surface_uuids

    def set_surface_ghost_solo(self, surface_uuid: str, solo: bool) -> bool:
        """Toggle the ghost-solo flag for one surface (by UUID).

        Returns True when the flag actually changed; False when it was
        already in the requested state (so callers can skip refresh work).
        Emits :attr:`ghostSoloChanged` on change. Solo state is purely a
        render-time view filter — it's NOT participating in the
        :meth:`edit` / undo path because the lens itself isn't changing
        and we don't want Ctrl+Z to surprise the user by undoing a
        debugging-view toggle.
        """
        if not surface_uuid:
            return False
        desired = bool(solo)
        present = surface_uuid in self._ghost_solo_surface_uuids
        if desired == present:
            return False
        if desired:
            self._ghost_solo_surface_uuids.add(surface_uuid)
        else:
            self._ghost_solo_surface_uuids.discard(surface_uuid)
        self.ghostSoloChanged.emit()
        return True

    def set_surfaces_ghost_solo(
        self, surface_uuids, solo: bool,
    ) -> bool:
        """Batch-toggle solo across many UUIDs with a single signal emission.

        Cheaper than looping :meth:`set_surface_ghost_solo` when a caller
        (element-level toggle, viewport radial menu) needs to flip many
        surfaces at once — each individual call would fire
        :attr:`ghostSoloChanged` and thrash listeners' rebuilds.
        """
        desired = bool(solo)
        changed = False
        for uuid in surface_uuids:
            if not uuid:
                continue
            present = uuid in self._ghost_solo_surface_uuids
            if desired and not present:
                self._ghost_solo_surface_uuids.add(uuid)
                changed = True
            elif not desired and present:
                self._ghost_solo_surface_uuids.discard(uuid)
                changed = True
        if changed:
            self.ghostSoloChanged.emit()
        return changed

    def clear_ghost_solo(self) -> bool:
        """Drop every solo'd surface. Returns True when anything cleared."""
        if not self._ghost_solo_surface_uuids:
            return False
        self._ghost_solo_surface_uuids.clear()
        self.ghostSoloChanged.emit()
        return True

    def _drop_ghost_solo_silent(self) -> None:
        """Clear ghost-solo state without emitting ghostSoloChanged.

        Used during ``new`` / ``load`` where ``systemReplaced`` is about
        to fire and listeners rebuild from scratch — firing the solo
        signal too would just cause a redundant repaint pass."""
        self._ghost_solo_surface_uuids.clear()

    # ------------------------------------------------------------------
    # Off-axis reveal — per-element ">>>" column expansion
    #
    # Explicit reveals only. An element with a non-zero pose value is
    # revealed whether or not it's in this set, and can't be un-revealed;
    # the optical editor's model composes the two halves. Kept off the
    # undo path for the same reason as ghost-solo: Ctrl+Z should undo lens
    # edits, not collapse a column the user just opened.
    # ------------------------------------------------------------------

    @property
    def off_axis_revealed_element_ids(self) -> frozenset:
        """Read-only snapshot of the explicitly-revealed element IDs."""
        return frozenset(self._off_axis_revealed_element_ids)

    def is_off_axis_revealed(self, element_id: str) -> bool:
        """True iff this element was EXPLICITLY revealed.

        Does not account for non-zero pose values — callers wanting the
        effective state should use the optical editor's model, which folds
        both halves together.
        """
        return bool(element_id) and element_id in self._off_axis_revealed_element_ids

    def set_off_axis_revealed(self, element_id: str, revealed: bool) -> bool:
        """Toggle the explicit reveal for one element. True when it changed."""
        if not element_id:
            return False
        desired = bool(revealed)
        present = element_id in self._off_axis_revealed_element_ids
        if desired == present:
            return False
        if desired:
            self._off_axis_revealed_element_ids.add(element_id)
        else:
            self._off_axis_revealed_element_ids.discard(element_id)
        self.offAxisRevealChanged.emit()
        return True

    def _drop_off_axis_reveal_silent(self) -> None:
        """Clear reveal state without emitting — see ``_drop_ghost_solo_silent``."""
        self._off_axis_revealed_element_ids.clear()

    # ------------------------------------------------------------------
    # Variable flags — per-surface-attribute optimization variables
    # ------------------------------------------------------------------

    def all_variable_flags(self) -> dict:
        """Read-only snapshot of the whole flag map.

        Returns a shallow copy so callers can iterate without worrying
        about concurrent mutation via toggle / clear.
        """
        return {
            uuid: dict(attrs) for uuid, attrs in self._variable_flags.items()
        }

    def get_variable_flags(self, surface_uuid: str) -> dict:
        """Return this surface's ``{attr -> VariableBounds}`` map.

        Empty dict when the surface has no flagged attributes; callers
        can safely ``.get(attr)`` without pre-checking.
        """
        if not surface_uuid:
            return {}
        return dict(self._variable_flags.get(surface_uuid, {}))

    def is_variable_flagged(self, surface_uuid: str, attr: str) -> bool:
        if not surface_uuid or not attr:
            return False
        return attr in self._variable_flags.get(surface_uuid, {})

    def variable_bounds(
        self, surface_uuid: str, attr: str,
    ) -> Optional[VariableBounds]:
        """Bounds for one flagged attribute, or ``None`` if not flagged."""
        if not surface_uuid or not attr:
            return None
        return self._variable_flags.get(surface_uuid, {}).get(attr)

    def set_variable_flag(
        self,
        surface_uuid: str,
        attr: str,
        bounds: Optional[VariableBounds] = None,
    ) -> bool:
        """Mark ``attr`` on ``surface_uuid`` as variable with optional bounds.

        ``bounds=None`` is treated as the default unbounded flag
        (:class:`VariableBounds()`). Idempotent: re-setting the same
        bounds returns False and emits nothing.

        Returns True when the flag actually changed. Emits
        :attr:`variableFlagChanged` on change. Variable flags do NOT
        participate in undo/redo — same call as merit_functions and
        ghost_solo.
        """
        if not surface_uuid or not attr:
            return False
        b = VariableBounds() if bounds is None else bounds
        current = self._variable_flags.get(surface_uuid, {}).get(attr)
        if current == b:
            return False
        self._variable_flags.setdefault(surface_uuid, {})[attr] = b
        self.variableFlagChanged.emit(surface_uuid, attr)
        return True

    def clear_variable_flag(self, surface_uuid: str, attr: str) -> bool:
        """Unmark ``attr`` on ``surface_uuid``. Returns True on change."""
        if not surface_uuid or not attr:
            return False
        attrs = self._variable_flags.get(surface_uuid)
        if attrs is None or attr not in attrs:
            return False
        del attrs[attr]
        if not attrs:
            # Prune empty surface entries so ``all_variable_flags`` stays tidy.
            del self._variable_flags[surface_uuid]
        self.variableFlagChanged.emit(surface_uuid, attr)
        return True

    def toggle_variable_flag(
        self,
        surface_uuid: str,
        attr: str,
        *,
        default_bounds: Optional[VariableBounds] = None,
    ) -> bool:
        """Convenience: clear if flagged, else set with ``default_bounds``.

        Returns the new flagged state (True = now flagged, False = now
        unflagged).
        """
        if self.is_variable_flagged(surface_uuid, attr):
            self.clear_variable_flag(surface_uuid, attr)
            return False
        self.set_variable_flag(surface_uuid, attr, default_bounds)
        return True

    def clear_all_variable_flags(self) -> bool:
        """Drop every flag. Returns True when anything was cleared.

        Emits :attr:`variableFlagsReplaced` (bulk signal) instead of one
        :attr:`variableFlagChanged` per entry so listeners take one
        repaint pass.
        """
        if not self._variable_flags:
            return False
        self._variable_flags.clear()
        self.variableFlagsReplaced.emit()
        return True

    def bulk_set_variable_flags(
        self,
        entries,
        *,
        default_bounds: Optional[VariableBounds] = None,
    ) -> bool:
        """Flag many (surface_uuid, attr) pairs with one signal emission.

        Used by the bulk "Flag All Radii" / "Flag All Thicknesses"
        menu actions in the Optical Design Editor. Skips pairs that
        are already flagged (with any bounds) so an existing user-set
        bounds is not clobbered by the bulk default.
        """
        b = VariableBounds() if default_bounds is None else default_bounds
        changed = False
        for uuid, attr in entries:
            if not uuid or not attr:
                continue
            existing = self._variable_flags.get(uuid, {}).get(attr)
            if existing is not None:
                continue
            self._variable_flags.setdefault(uuid, {})[attr] = b
            changed = True
        if changed:
            # Bulk mutation → coarse signal so the tree repaints once.
            self.variableFlagsReplaced.emit()
        return changed

    def _drop_variable_flags_silent(self) -> None:
        """Clear the flag map without emitting variableFlagsReplaced.

        Used during ``new`` / ``load`` where the caller emits the bulk
        signal itself alongside ``systemReplaced``."""
        self._variable_flags.clear()

    # ------------------------------------------------------------------
    # Material-substitution flags — per-material catalogue-hammer picks
    # ------------------------------------------------------------------

    def all_material_flags(self) -> dict:
        """Read-only snapshot of the whole material-flag map.

        Shape: ``{element_id -> {material_index -> SubstitutionSpec}}``.
        Returns a nested shallow copy so callers can iterate without
        worrying about concurrent mutation.
        """
        return {
            eid: dict(mats) for eid, mats in self._material_flags.items()
        }

    def get_material_flags(self, element_id: str) -> dict:
        """Return this element's ``{material_index -> SubstitutionSpec}`` map."""
        if not element_id:
            return {}
        return dict(self._material_flags.get(element_id, {}))

    def is_material_flagged(self, element_id: str, material_index: int) -> bool:
        if not element_id:
            return False
        return material_index in self._material_flags.get(element_id, {})

    def material_flag_spec(
        self, element_id: str, material_index: int,
    ) -> Optional[SubstitutionSpec]:
        """Spec for one flagged material, or ``None`` if not flagged."""
        if not element_id:
            return None
        return self._material_flags.get(element_id, {}).get(material_index)

    def set_material_flag(
        self,
        element_id: str,
        material_index: int,
        spec: Optional[SubstitutionSpec] = None,
    ) -> bool:
        """Mark ``(element_id, material_index)`` as flagged for substitution.

        ``spec=None`` maps to the default ``SubstitutionSpec()`` (empty
        vendor — same semantics as "unbounded" for VariableBounds; the
        Optimizer treats this as "no vendor picked yet" and skips).
        Idempotent: re-setting the same spec returns False and emits
        nothing.

        Returns True when the flag actually changed. Emits
        :attr:`materialFlagChanged` on change.
        """
        if not element_id or material_index < 0:
            return False
        s = SubstitutionSpec() if spec is None else spec
        current = self._material_flags.get(element_id, {}).get(material_index)
        if current == s:
            return False
        self._material_flags.setdefault(element_id, {})[material_index] = s
        self.materialFlagChanged.emit(element_id, int(material_index))
        return True

    def clear_material_flag(
        self, element_id: str, material_index: int,
    ) -> bool:
        """Unmark ``(element_id, material_index)``. Returns True on change."""
        if not element_id or material_index < 0:
            return False
        mats = self._material_flags.get(element_id)
        if mats is None or material_index not in mats:
            return False
        del mats[material_index]
        if not mats:
            del self._material_flags[element_id]
        self.materialFlagChanged.emit(element_id, int(material_index))
        return True

    def toggle_material_flag(
        self,
        element_id: str,
        material_index: int,
        *,
        default_spec: Optional[SubstitutionSpec] = None,
    ) -> bool:
        """Convenience: clear if flagged, else set with ``default_spec``.

        Returns the new flagged state (True = now flagged).
        """
        if self.is_material_flagged(element_id, material_index):
            self.clear_material_flag(element_id, material_index)
            return False
        self.set_material_flag(element_id, material_index, default_spec)
        return True

    def clear_all_material_flags(self) -> bool:
        """Drop every material flag. Emits :attr:`materialFlagsReplaced`."""
        if not self._material_flags:
            return False
        self._material_flags.clear()
        self.materialFlagsReplaced.emit()
        return True

    def _prune_dead_material_flags(self) -> None:
        """Drop flag entries for elements or material indices that no
        longer exist. Called after every mutation, symmetric with
        :meth:`_prune_dead_variable_flags`.

        Element deletion / undo removes the element wrapper; a shrinking
        ``material_glasses`` list (from a Doublet → Singlet convert, say)
        also drops the trailing indices. Both cases are covered here so
        the flag map never references phantom (element_id, index) pairs.
        """
        if not self._material_flags:
            return
        live_by_id: dict[str, int] = {}
        try:
            for el in self._system.elements:
                eid = getattr(el, "element_id", None)
                if eid:
                    live_by_id[str(eid)] = len(el.material_glasses)
        except Exception:
            live_by_id = {}
        dead_elements: list[str] = []
        changed = False
        for eid, mats in list(self._material_flags.items()):
            n_live = live_by_id.get(eid, -1)
            if n_live < 0:
                dead_elements.append(eid)
                changed = True
                continue
            dead_indices = [mi for mi in mats if mi >= n_live]
            for mi in dead_indices:
                del mats[mi]
                changed = True
            if not mats:
                dead_elements.append(eid)
        for eid in dead_elements:
            self._material_flags.pop(eid, None)
        if changed:
            self.materialFlagsReplaced.emit()

    def _drop_material_flags_silent(self) -> None:
        """Clear the flag map without emitting materialFlagsReplaced.

        Used during ``new`` / ``load`` where the caller emits the bulk
        signal itself alongside ``systemReplaced``."""
        self._material_flags.clear()

    def _clear_selection_if_any(self) -> None:
        cleared_element = self._selected_element is not None
        cleared_surface = self._selected_surface_index is not None
        if not (cleared_element or cleared_surface):
            return
        self._selected_element = None
        self._selected_surface_index = None
        if cleared_element:
            self.selectionChanged.emit(None)
        if cleared_surface:
            self.surfaceSelectionChanged.emit(None)

    def _capture_selection_handles(self) -> tuple[Optional[str], Optional[str]]:
        """Snapshot the current selection by stable identifiers (element_id
        + surface UUID) so it can be re-resolved across a reload that
        creates fresh Element / Surface wrappers."""
        el = self._selected_element
        element_id = el.element_id if el is not None else None
        surface_uuid: Optional[str] = None
        si = self._selected_surface_index
        if si is not None:
            try:
                surface_uuid = str(self._system.surface_ids[si])
            except (IndexError, AttributeError):
                surface_uuid = None
        return element_id, surface_uuid

    def _resolve_selection_handles(
        self,
        element_id: Optional[str],
        surface_uuid: Optional[str],
    ) -> None:
        """Re-resolve selection from UUID handles into the current system,
        emitting ``selectionChanged`` / ``surfaceSelectionChanged`` only
        when the resolved values differ from what's already set.

        Used by ``undo`` / ``redo`` so the user's selection survives the
        snapshot reload — losing the highlighted row + viewport surface
        on every Ctrl+Z would make A/B comparing impossible."""
        new_element = None
        if element_id:
            for el in self._system.elements:
                if el.element_id == element_id:
                    new_element = el
                    break
        new_surface_index: Optional[int] = None
        if new_element is not None and surface_uuid:
            try:
                idx = list(self._system.surface_ids).index(surface_uuid)
            except ValueError:
                idx = None
            if idx is not None:
                # The picked surface must still belong to the resolved
                # element — otherwise the surface highlight is meaningless.
                try:
                    if idx in new_element.resolve_surfaces(self._system):
                        new_surface_index = idx
                except KeyError:
                    pass

        element_changed = new_element is not self._selected_element
        surface_changed = new_surface_index != self._selected_surface_index
        self._selected_element = new_element
        self._selected_surface_index = new_surface_index
        if element_changed:
            self.selectionChanged.emit(new_element)
        if surface_changed:
            self.surfaceSelectionChanged.emit(new_surface_index)

    def _validate_selection_against_system(self) -> None:
        if self._selected_element is not None and self._selected_element not in self._system.elements:
            self._clear_selection_if_any()
            return
        self._validate_surface_against_element()

    def _validate_surface_against_element(self) -> None:
        if self._selected_surface_index is None:
            return
        coerced = self._coerce_surface_index(self._selected_surface_index)
        if coerced != self._selected_surface_index:
            self._selected_surface_index = coerced
            self.surfaceSelectionChanged.emit(coerced)

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_label(self) -> Optional[str]:
        return self._undo[-1][0] if self._undo else None

    @property
    def redo_label(self) -> Optional[str]:
        return self._redo[-1][0] if self._redo else None

    @contextlib.contextmanager
    def edit(self, label: str) -> Iterator["EditHandle"]:
        """Context manager: snapshot before, run body, push undo entry on exit.

        Yields an :class:`EditHandle`; call ``handle.abort()`` to cancel the
        push (e.g. when the mutation turned out to be a no-op). The body's
        return value / exceptions are propagated normally — an exception
        also cancels the push.

        Inside an active compound, the inner ``edit`` does not push its own
        entry — its mutation contributes to the compound's single
        before/after snapshot pair — but it DOES still call
        :meth:`mark_modified` per inner edit so observers (viewport, etc.)
        repaint live during a scrub drag.
        """
        handle = EditHandle()
        if self._compound_depth > 0:
            yield handle
            if not handle.aborted:
                self.mark_modified()
            return
        before = self._snapshot()
        try:
            yield handle
        except BaseException:
            raise
        else:
            if handle.aborted:
                return
            self._push_undo(label, before)
            self.mark_modified()

    def begin_compound(self, label: str) -> None:
        """Open a compound — N inner edits collapse into one undo entry.

        Nestable: only the outermost begin/end pair takes snapshots and
        pushes. Mismatched extra ``end_compound`` calls are tolerated.
        """
        if self._compound_depth == 0:
            self._compound_before = (label, self._snapshot())
        self._compound_depth += 1

    def end_compound(self) -> None:
        if self._compound_depth == 0:
            return
        self._compound_depth -= 1
        if self._compound_depth > 0:
            return
        before = self._compound_before
        self._compound_before = None
        if before is None:
            return
        label, snapshot = before
        self._push_undo(label, snapshot)
        self.mark_modified()

    def undo(self) -> None:
        if not self._undo:
            return
        after = self._snapshot()
        label, before = self._undo.pop()
        self._snapshot_sizes.pop()
        # Snapshot selection by UUID before the reload nukes our wrappers.
        sel_element_id, sel_surface_uuid = self._capture_selection_handles()
        try:
            self._restore_snapshot(before)
        except Exception:
            # Restoration failed mid-way — the C++ loader clears the
            # surfaces / surface_ids arrays before parsing, so a rejected
            # snapshot (e.g. an empty ``optical_system`` block, which
            # ``OpticalSystem.load`` refuses) leaves the system in a
            # half-empty state where ``_elements`` still references UUIDs
            # that no longer exist. Roll forward to the AFTER snapshot
            # we just captured so the UI shows the (still-valid) pre-undo
            # state, refresh listeners, and re-raise so the caller sees
            # the underlying failure. The failing entry is left dropped
            # from the undo stack — it can never be restored, so keeping
            # it would just trap the user on a perpetually-failing entry.
            self._refresh_can_undo_redo()
            try:
                self._restore_snapshot(after)
            except Exception:
                # Even the rollback failed — nothing more we can do
                # to recover state. The original failure is still
                # the more informative error to surface.
                pass
            self._resolve_selection_handles(sel_element_id, sel_surface_uuid)
            self._last_replacement_kind = "restore"
            self.systemReplaced.emit(self._system)
            raise
        self._redo.append((label, after))
        self._redo_sizes.append(self._dict_size(after))
        self._enforce_budget()
        self._set_dirty(True)
        self._resolve_selection_handles(sel_element_id, sel_surface_uuid)
        self._last_replacement_kind = "restore"
        self.systemReplaced.emit(self._system)
        # Restored system may have shed surfaces whose UUIDs were solo'd
        # in the outgoing state — drop them so the set stays truthful.
        self._prune_dead_ghost_solo()
        self._refresh_can_undo_redo()

    def redo(self) -> None:
        if not self._redo:
            return
        before = self._snapshot()
        label, after = self._redo.pop()
        self._redo_sizes.pop()
        sel_element_id, sel_surface_uuid = self._capture_selection_handles()
        try:
            self._restore_snapshot(after)
        except Exception:
            # Mirror the undo error path — see :meth:`undo` for the
            # rationale on dropping the failing entry and rolling
            # forward to the captured pre-redo snapshot.
            self._refresh_can_undo_redo()
            try:
                self._restore_snapshot(before)
            except Exception:
                pass
            self._resolve_selection_handles(sel_element_id, sel_surface_uuid)
            self._last_replacement_kind = "restore"
            self.systemReplaced.emit(self._system)
            raise
        self._undo.append((label, before))
        self._snapshot_sizes.append(self._dict_size(before))
        self._enforce_budget()
        self._set_dirty(True)
        self._resolve_selection_handles(sel_element_id, sel_surface_uuid)
        self._last_replacement_kind = "restore"
        self.systemReplaced.emit(self._system)
        self._prune_dead_ghost_solo()
        self._refresh_can_undo_redo()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _snapshot(self) -> dict:
        return build_optical_system_doc(
            system=self._system,
            metadata=self._system._raw_metadata,
            glass_catalogue=self._system._raw_glass_catalogue,
        )

    def _restore_snapshot(self, doc: dict) -> None:
        # OpticalSystem.reload() goes through the C++ loader, which only
        # accepts a path. Round-trip via a short-lived temp file. This
        # happens at most once per Ctrl+Z / Ctrl+Y — capture stays in memory.
        fd, tmp_path = tempfile.mkstemp(suffix=".lens")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
            self._system.reload(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _push_undo(self, label: str, snapshot: dict) -> None:
        self._undo.append((label, snapshot))
        self._snapshot_sizes.append(self._dict_size(snapshot))
        self._redo.clear()
        self._redo_sizes.clear()
        self._enforce_budget()
        self._refresh_can_undo_redo()

    @staticmethod
    def _dict_size(d: dict) -> int:
        return len(json.dumps(d))

    def _enforce_budget(self) -> None:
        budget = self._snapshot_budget_bytes
        total = sum(self._snapshot_sizes) + sum(self._redo_sizes)
        while total > budget and self._undo:
            total -= self._snapshot_sizes.pop(0)
            self._undo.pop(0)
        # Once undo is empty, redo entries also become unreachable history;
        # drop them too rather than letting them block fresh snapshots.
        while total > budget and self._redo:
            total -= self._redo_sizes.pop(0)
            self._redo.pop(0)

    def _clear_history(self) -> None:
        self._undo.clear()
        self._redo.clear()
        self._snapshot_sizes.clear()
        self._redo_sizes.clear()
        self._compound_depth = 0
        self._compound_before = None
        self._refresh_can_undo_redo()

    def _refresh_can_undo_redo(self) -> None:
        cu = self.can_undo
        cr = self.can_redo
        if cu != self._last_can_undo:
            self._last_can_undo = cu
            self.canUndoChanged.emit(cu)
        if cr != self._last_can_redo:
            self._last_can_redo = cr
            self.canRedoChanged.emit(cr)

    def _set_dirty(self, value: bool) -> None:
        if value != self._dirty:
            self._dirty = value
            self.dirtyChanged.emit(value)
