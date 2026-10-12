"""
Morphix Desktop Companion Robot - Generic User Profile Manager
==============================================================
File: src/config/profile_manager.py
Description:
    Provides dynamic and generic helper functions to load, update, and manage
    user profiles and household members without hardcoded individual identities.
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default relative or absolute path to user_profile.json
_DEFAULT_PROFILE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROFILE_PATH = os.path.join(_DEFAULT_PROFILE_DIR, "user_profile.json")

DEFAULT_PROFILE_DATA: Dict[str, Any] = {
    "primary_user": "User",
    "household_members": []
}


def get_default_profile_path() -> str:
    """
    Returns the default path to the user profile JSON file.

    :return: Absolute file path to user_profile.json.
    """
    return DEFAULT_PROFILE_PATH


def load_user_profile(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads the user profile configuration dictionary.
    Creates default template if file does not exist.

    :param config_path: Optional custom path to user_profile.json.
    :return: Dict containing profile configuration.
    """
    target_path = config_path or DEFAULT_PROFILE_PATH

    if not os.path.exists(target_path):
        logger.info(f"User profile not found at '{target_path}'. Initializing generic template.")
        save_user_profile(DEFAULT_PROFILE_DATA.copy(), config_path=target_path)
        return DEFAULT_PROFILE_DATA.copy()

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                logger.warning(f"Invalid format in '{target_path}'. Using default template.")
                return DEFAULT_PROFILE_DATA.copy()
            # Ensure standard keys exist
            if "primary_user" not in data:
                data["primary_user"] = "User"
            if "household_members" not in data or not isinstance(data["household_members"], list):
                data["household_members"] = []
            return data
    except Exception as e:
        logger.error(f"Error reading user profile from '{target_path}': {e}")
        return DEFAULT_PROFILE_DATA.copy()


def save_user_profile(profile_data: Dict[str, Any], config_path: Optional[str] = None) -> bool:
    """
    Safely saves the user profile configuration dictionary to disk atomically.

    :param profile_data: Dict containing profile data to save.
    :param config_path: Optional custom path to user_profile.json.
    :return: True if successfully saved, False otherwise.
    """
    target_path = config_path or DEFAULT_PROFILE_PATH
    target_dir = os.path.dirname(target_path)

    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    try:
        # Atomic write via temporary file
        temp_dir = target_dir if target_dir else None
        with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as tf:
            json.dump(profile_data, tf, indent=2, ensure_ascii=False)
            temp_name = tf.name

        # Replace target with temp file
        os.replace(temp_name, target_path)
        return True
    except Exception as e:
        logger.error(f"Error saving user profile to '{target_path}': {e}")
        return False


def get_primary_user(config_path: Optional[str] = None) -> str:
    """
    Dynamically retrieves the primary user's name from configuration.

    :param config_path: Optional path to config file.
    :return: Primary user's name string.
    """
    profile = load_user_profile(config_path=config_path)
    return profile.get("primary_user", "User")


def set_primary_user(name: str, config_path: Optional[str] = None) -> bool:
    """
    Dynamically sets and persists the primary user's name.

    :param name: New primary user name.
    :param config_path: Optional path to config file.
    :return: True if updated successfully, False otherwise.
    """
    if not name or not isinstance(name, str) or not name.strip():
        logger.error("Primary user name must be a non-empty string.")
        return False

    cleaned_name = name.strip()
    profile = load_user_profile(config_path=config_path)
    profile["primary_user"] = cleaned_name
    return save_user_profile(profile, config_path=config_path)


def get_household_members(config_path: Optional[str] = None) -> List[str]:
    """
    Retrieves the list of registered household members.

    :param config_path: Optional path to config file.
    :return: List of household member names.
    """
    profile = load_user_profile(config_path=config_path)
    return list(profile.get("household_members", []))


def add_household_member(name: str, config_path: Optional[str] = None) -> bool:
    """
    Adds a new household member if not already present.

    :param name: Household member name.
    :param config_path: Optional path to config file.
    :return: True if added or already present, False on failure.
    """
    if not name or not isinstance(name, str) or not name.strip():
        logger.error("Member name must be a non-empty string.")
        return False

    cleaned_name = name.strip()
    profile = load_user_profile(config_path=config_path)
    members: List[str] = profile.get("household_members", [])

    if cleaned_name not in members:
        members.append(cleaned_name)
        profile["household_members"] = members
        return save_user_profile(profile, config_path=config_path)

    return True


def remove_household_member(name: str, config_path: Optional[str] = None) -> bool:
    """
    Removes a household member from configuration.

    :param name: Household member name to remove.
    :param config_path: Optional path to config file.
    :return: True if removed, False if not found or on error.
    """
    cleaned_name = name.strip()
    profile = load_user_profile(config_path=config_path)
    members: List[str] = profile.get("household_members", [])

    if cleaned_name in members:
        members.remove(cleaned_name)
        profile["household_members"] = members
        return save_user_profile(profile, config_path=config_path)

    return False
