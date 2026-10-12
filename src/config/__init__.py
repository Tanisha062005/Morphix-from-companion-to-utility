"""
Morphix Desktop Companion Robot - Config Module
===============================================
File: src/config/__init__.py
"""

from src.config.profile_manager import (
    DEFAULT_PROFILE_DATA,
    DEFAULT_PROFILE_PATH,
    add_household_member,
    get_default_profile_path,
    get_household_members,
    get_primary_user,
    load_user_profile,
    remove_household_member,
    save_user_profile,
    set_primary_user,
)

__all__ = [
    "DEFAULT_PROFILE_PATH",
    "DEFAULT_PROFILE_DATA",
    "get_default_profile_path",
    "load_user_profile",
    "save_user_profile",
    "get_primary_user",
    "set_primary_user",
    "get_household_members",
    "add_household_member",
    "remove_household_member",
]
