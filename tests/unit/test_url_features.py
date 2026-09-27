"""Structural URL features are computed from the string only."""
from __future__ import annotations

from phishlens.adapters.url_model import url_features


def test_ip_host_detected():
    f = url_features("http://192.0.2.10/login")
    assert f["host_is_ip_address"] is True


def test_at_symbol_detected():
    f = url_features("http://192.0.2.10/a@b")
    assert f["has_at_symbol"] is True


def test_https_flag():
    assert url_features("https://example.com/")["uses_https"] is True
    assert url_features("http://example.com/")["uses_https"] is False


def test_dot_and_hyphen_counts():
    f = url_features("https://a-b-c.example.com/x.y.z")
    assert f["hyphen_count"] == 2
    assert f["dot_count"] >= 2


def test_punycode_host_detected():
    f = url_features("https://xn--exmple-cua.com/")
    assert f["host_is_punycode"] is True


def test_no_network_access_in_feature_extraction(monkeypatch):
    import socket

    def boom(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)
    url_features("https://example.com/a-b-c@192.0.2.5")
