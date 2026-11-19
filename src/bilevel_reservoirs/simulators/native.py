"""Shared access to the pinned LOSLR reference for the Ontario adapter."""

from functools import cache

from reservoir_control.systems.loslr.native import NativeReference


@cache
def native_reference():
    return NativeReference()
