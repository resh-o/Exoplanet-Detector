"""
Patches astropy logger._set_defaults AttributeError on Python 3.13.

Run this once after pip install -r requirements.txt if you hit:
  AttributeError: 'Logger' object has no attribute '_set_defaults'

Root cause: logging.getLogger("astropy") can return a plain logging.Logger
instance in Python 3.13 when the logger was already registered before
AstropyLogger was set as the logger class. Plain Logger has no _set_defaults.
"""

import os
import sys


def find_astropy_logger():
    for path in sys.path:
        candidate = os.path.join(path, "astropy", "logger.py")
        if os.path.exists(candidate):
            return candidate
    return None


def patch():
    logger_path = find_astropy_logger()
    if not logger_path:
        print("ERROR: Could not find astropy logger.py")
        sys.exit(1)

    with open(logger_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "except AttributeError" in content and "log._set_defaults()" in content:
        print("Patch already applied.")
        return

    old = "        log._set_defaults()"
    new = (
        "        try:\n"
        "            log._set_defaults()\n"
        "        except AttributeError:\n"
        "            pass"
    )

    if old not in content:
        print("ERROR: Could not find patch target in logger.py")
        print("The astropy version may have changed. Check manually.")
        print(f"Logger path: {logger_path}")
        sys.exit(1)

    patched = content.replace(old, new, 1)

    with open(logger_path, "w", encoding="utf-8") as f:
        f.write(patched)

    print(f"Patch applied successfully to {logger_path}")


if __name__ == "__main__":
    patch()
