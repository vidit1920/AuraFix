from calendar_utils.dates import days_in_february, days_in_year


def test_february_2000():
    assert days_in_february(2000) == 29


def test_year_2023():
    assert days_in_year(2023) == 365
