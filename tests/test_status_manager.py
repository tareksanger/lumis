from enum import Enum

from lumis.core.utils.status_manager import StatusManager

import pytest


class Section(Enum):
    INTRO = 0
    BODY = 1
    END = 2


@pytest.fixture
def manager():
    return StatusManager(Section)


@pytest.mark.parametrize("name,code", [("notStarted", 0), ("pending", 1), ("error", 2), ("complete", 3)])
def test_section_updates_are_independent(manager, name, code):
    manager.update_status(Section.BODY, "complete")
    manager.update_status(Section.INTRO, name)
    assert manager.get_status(Section.INTRO) == code
    assert manager.get_section_status_display(Section.INTRO) == name
    assert manager.get_status(Section.BODY) == manager.COMPLETE
    assert manager.get_status(Section.END) == manager.NOT_STARTED


def test_aggregate_status_precedence_and_filters(manager):
    assert manager.is_not_started()
    assert manager.all_sections_have_status("notStarted")
    assert manager.get_status_display() == "notStarted"
    assert manager.with_status("error") is None
    manager.update_status(Section.INTRO, "complete")
    assert not manager.is_not_started()
    manager.update_status(Section.BODY, "pending")
    assert manager.is_pending()
    assert manager.get_status_display() == "pending"
    manager.update_status(Section.END, "error")
    assert manager.has_error()
    assert manager.get_status_display() == "error"
    assert manager.with_status("complete") == ["INTRO"]
    for section in Section:
        manager.update_status(section, "complete")
    assert manager.is_complete()
    assert manager.get_status_display() == "complete"
    assert not manager.has_error()
    assert not manager.is_pending()


@pytest.mark.parametrize("method", ["check_sections_for_status", "all_sections_have_status", "with_status"])
def test_invalid_status_rejected(manager, method):
    with pytest.raises(ValueError, match="Invalid status"):
        getattr(manager, method)("unknown")


def test_invalid_section_rejected(manager):
    class Other(Enum):
        INTRO = 0

    for method in [manager.get_status, manager.get_section_status_display]:
        with pytest.raises(ValueError, match="Invalid section"):
            method(Other.INTRO)
    with pytest.raises(ValueError, match="Invalid section"):
        manager.update_status(Other.INTRO, "pending")
    assert manager.status == 0


def test_print_status(manager, capsys):
    manager.update_status(Section.INTRO, "pending")
    manager.print_status()
    assert capsys.readouterr().out == "INTRO: Generating\nBODY: Not Started\nEND: Not Started\n"
