
import pytest

from predict_student import get_valid_number


def test_accepts_valid_number(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "8.5")

    assert get_valid_number("Study hours: ", 0, 24) == 8.5


def test_rejects_number_above_maximum(monkeypatch, capsys):
    answers = iter(["25", "5"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    result = get_valid_number("Study hours: ", 0, 24)

    assert result == 5
    assert "Enter a value between 0 and 24." in capsys.readouterr().out


def test_rejects_negative_number(monkeypatch, capsys):
    answers = iter(["-1", "6"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    result = get_valid_number("Study hours: ", 0, 24)

    assert result == 6
    assert "Enter a value between 0 and 24." in capsys.readouterr().out


def test_rejects_non_numeric_input(monkeypatch, capsys):
    answers = iter(["hello", "7"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    result = get_valid_number("Study hours: ", 0, 24)

    assert result == 7
    assert "Invalid input. Please enter a number." in capsys.readouterr().out
