"""Unit tests must not reach provider APIs, databases, or download model weights."""

import socket

import pytest


@pytest.fixture(autouse=True)
def block_external_connections(monkeypatch):
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def guard(method):
        def connect(sock, address):
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                raise AssertionError(f"Network access is disabled in unit tests: {address!r}")
            return method(sock, address)

        return connect

    monkeypatch.setattr(socket.socket, "connect", guard(original_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guard(original_connect_ex))
