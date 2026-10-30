"""Input overlays cannot redirect writes or select an unregistered month."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pathveil.core import activate
from pathveil.core.io import open_data
from pathveil.utils import write_json


class CoreInputPaths(unittest.TestCase):
    def test_read_mapping_never_redirects_a_write_to_the_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, local, manifest = (
                root / "original.txt",
                root / "local.txt",
                root / "data.json",
            )
            original.write_text("immutable")
            write_json(manifest, {"read_path_map": {str(local): str(original)}})
            with patch.dict(os.environ, {"PATHVEIL_DATA_MANIFEST": str(manifest)}):
                with open_data(local) as stream:
                    self.assertEqual(stream.read(), "immutable")
                with open_data(local, "w") as stream:
                    stream.write("generated")
            self.assertEqual(original.read_text(), "immutable")
            self.assertEqual(local.read_text(), "generated")

    def test_core_requires_an_explicit_asrank_binding(self):
        activate()
        import environment

        instance = environment.Env.__new__(environment.Env)
        with patch.dict(os.environ):
            os.environ.pop("PATHVEIL_ASRANK_LINKS", None)
            with self.assertRaisesRegex(ValueError, "PATHVEIL_ASRANK_LINKS"):
                instance._nearest_asrank_link_file()


if __name__ == "__main__":
    unittest.main()
