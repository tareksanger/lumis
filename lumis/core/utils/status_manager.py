from enum import Enum
import logging
from typing import List, Literal, Optional, Type

logger = logging.getLogger(__name__)


class StatusManager:
    """
    A class that manages the status of sections in a report.

    The design assumes a fixed number of sections (up to 16 for a 32-bit integer or 32 for a 64-bit integer if using BigIntegerField).
    If the number of report sections exceeds this limit, you would need to adjust the implementation, possibly by using multiple fields or
    a different data structure.

    Each section's status is represented by 2 bits, allowing for only four possible states (e.g., Not Started, Pending, Error, Complete).
    If you need to track more detailed states or additional information for each section, the current setup would be inadequate.

    Attributes:
        sections_enum (Type[Enum]): The enumeration class representing the sections.
        status (int): The current status of the report.

    """

    Status = Literal["notStarted", "pending", "error", "complete"]

    # Status Codes
    NOT_STARTED: int = 0b00
    PENDING: int = 0b01
    ERROR: int = 0b10
    COMPLETE: int = 0b11

    def __init__(self, sections_enum: Type[Enum]):
        """
        Initializes a ReportStatusManager instance.

        Args:
            sections_enum (Type[Enum]): The enumeration class representing the sections.

        """
        self.sections_enum: Type[Enum] = sections_enum
        self.status: int = 0

    def _get_status_value_from_name(self, status_name: Status) -> int:
        status_map = {"notStarted": self.NOT_STARTED, "pending": self.PENDING, "error": self.ERROR, "complete": self.COMPLETE}
        if status_name not in status_map:
            raise ValueError("Invalid status name")

        return status_map[status_name]

    def update_status(self, section: Enum, new_status: Status):
        """
        Updates the status of a section.

        Args:
            section (Enum): The section to update the status for.
            new_status (int): The new status code for the section.

        Raises:
            ValueError: If the section is invalid.

        """
        status_value = self._get_status_value_from_name(new_status)

        if section not in self.sections_enum:
            raise ValueError("Invalid section")

        section_index: int = section.value
        self.status &= ~(0b11 << (section_index * 2))
        self.status |= status_value << (section_index * 2)

        logger.info(f"Updated status of {section.name} to {new_status}")
        return self.status

    def get_status(self, section: Enum) -> int:
        """
        Retrieves the status of a section.

        Args:
            section (Enum): The section to retrieve the status for.

        Returns:
            int: The status code of the section.

        Raises:
            ValueError: If the section is invalid.

        """
        if section not in self.sections_enum:
            raise ValueError("Invalid section")

        section_index: int = section.value
        return (self.status >> (section_index * 2)) & 0b11

    def check_sections_for_status(self, status: Status) -> bool:
        """
        Checks if any section has the specified status.

        Args:
            status (int): The status code to check.

        Returns:
            bool: True if any section has the specified status, False otherwise.

        Raises:
            ValueError: If the status is invalid.

        """
        status_value = self._get_status_value_from_name(status)
        if status_value not in [self.NOT_STARTED, self.PENDING, self.ERROR, self.COMPLETE]:
            raise ValueError("Invalid status")
        return any(self.get_status(section) == status_value for section in self.sections_enum)

    def all_sections_have_status(self, status: Status) -> bool:
        """
        Checks if any section has the specified status.

        Args:
            status (int): The status code to check.

        Returns:
            bool: True if any section has the specified status, False otherwise.

        Raises:
            ValueError: If the status is invalid.

        """
        status_value = self._get_status_value_from_name(status)
        if status_value not in [self.NOT_STARTED, self.PENDING, self.ERROR, self.COMPLETE]:
            raise ValueError("Invalid status")
        return all(self.get_status(section) == status_value for section in self.sections_enum)

    def is_not_started(self) -> bool:
        """
        Checks if all section has started.

        Returns:
            bool: True if any section has started, False otherwise.

        """
        return not self.all_sections_have_status("notStarted")

    def has_error(self) -> bool:
        """
        Checks if any section has an error.

        Returns:
            bool: True if any section has an error, False otherwise.

        """
        return self.check_sections_for_status("error")

    def is_pending(self) -> bool:
        """
        Checks if any section is pending.

        Returns:
            bool: True if any section is pending, False otherwise.

        """
        pending = self.check_sections_for_status("pending")

        return pending

    def is_complete(self) -> bool:
        """
        Checks if all sections are complete.

        Returns:
            bool: True if all sections are complete, False otherwise.

        """
        return self.all_sections_have_status("complete")

    def with_status(self, status: Status) -> Optional[List[str]]:
        """
        Returns a list of section names that have the specified status.

        Args:
            status (Status): The status to filter sections by.

        Returns:
            Optional[List[str]]: A list of section names that have the specified status. If no sections have the specified status, returns None.
        """

        status_value = self._get_status_value_from_name(status)
        sections_in_status: List[str] = [section.name for section in self.sections_enum if self.get_status(section) == status_value]

        if sections_in_status:
            return sections_in_status
        else:
            return None

    def get_status_display(self) -> Status:
        """
        Retrieves the display representation of the status.

        Returns:
            str: The display representation of the status.

        """
        if self.has_error():
            return "error"
        elif self.is_pending():
            return "pending"
        elif self.is_complete():
            return "complete"
        else:
            return "notStarted"

    def get_section_status_display(self, section: Enum) -> Status:
        """
        Retrieves the status of a section.

        Args:
            section (Enum): The section to retrieve the status for.

        Returns:
            int: The status code of the section.

        Raises:
            ValueError: If the section is invalid.

        """
        if section not in self.sections_enum:
            raise ValueError("Invalid section")

        is_error = self.get_status(section) == self.ERROR
        is_pending = self.get_status(section) == self.PENDING
        is_complete = self.get_status(section) == self.COMPLETE

        if is_error:
            return "error"
        elif is_pending:
            return "pending"
        elif is_complete:
            return "complete"
        else:
            return "notStarted"

    def print_status(self) -> None:
        """
        Prints the status of each section.

        """
        statuses: list[str] = ["Not Started", "Generating", "Error", "Complete"]
        for section in self.sections_enum:
            status_code: int = self.get_status(section)
            print(f"{section.name}: {statuses[status_code]}")
