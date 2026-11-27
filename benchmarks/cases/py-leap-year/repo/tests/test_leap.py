from calendar_utils.leap import is_leap_year


def test_common_leap_year():
    assert is_leap_year(2024)


def test_common_year():
    assert not is_leap_year(2023)


def test_century_is_not_leap():
    assert not is_leap_year(1900)


def test_400_year_rule():
    assert is_leap_year(2000)
