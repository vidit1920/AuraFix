from calendar_utils.leap import is_leap_year


def days_in_february(year: int) -> int:
    return 29 if is_leap_year(year) else 28


def days_in_year(year: int) -> int:
    return 366 if is_leap_year(year) else 365
