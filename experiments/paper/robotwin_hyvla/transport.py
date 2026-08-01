from __future__ import annotations

import pickle
import socket
import struct
from typing import Any


HEADER = struct.Struct("!Q")


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("peer closed the socket")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> Any:
    size = HEADER.unpack(_recv_exact(sock, HEADER.size))[0]
    return pickle.loads(_recv_exact(sock, size))


def send_message(sock: socket.socket, value: Any) -> None:
    payload = pickle.dumps(value, protocol=5)
    sock.sendall(HEADER.pack(len(payload)))
    sock.sendall(payload)


def request(sock: socket.socket, value: Any) -> Any:
    send_message(sock, value)
    response = recv_message(sock)
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "Hy-VLA request failed"))
    return response
