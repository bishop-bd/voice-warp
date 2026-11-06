"""Shared checkpoint metadata policy; fake tensor shapes need no ML packages."""
from types import SimpleNamespace
import unittest

from voice_warp_checkpoint import normalized_voice_config, voice_metadata


class CheckpointTests(unittest.TestCase):
    def checkpoint(self, rate="48k"):
        return {"config": [0] * 15 + [999, 256, rate], "version": "v2", "f0": 1,
                "weight": {"emb_g.weight": SimpleNamespace(shape=(7, 256))}}

    def test_normalizes_rates_and_actual_speaker_count_without_mutation(self):
        for encoded, expected in (("32k", 32000), ("40k", 40000), ("48k", 48000),
                                  (32000, 32000), (40000, 40000), (48000, 48000)):
            checkpoint = self.checkpoint(encoded)
            with self.subTest(rate=encoded):
                config = normalized_voice_config(checkpoint)
                self.assertEqual(config[-3:], [7, 256, expected])
                self.assertEqual(checkpoint["config"][-3:], [999, 256, encoded])
                self.assertEqual(voice_metadata(checkpoint)["samplingRate"], expected)

    def test_metadata_keeps_v1_projection_and_v2_features_distinct(self):
        checkpoint = self.checkpoint()
        self.assertEqual(voice_metadata(checkpoint)["embChannels"], 768)
        checkpoint["version"] = None
        metadata = voice_metadata(checkpoint)
        self.assertEqual((metadata["embChannels"], metadata["embOutputLayer"], metadata["useFinalProj"]),
                         (256, 9, True))

    def test_rejects_unsupported_models_and_missing_weights(self):
        valid = self.checkpoint()
        invalid = [None, [], {**valid, "config": [0] * 19},
                   {**valid, "version": "custom"}, {**valid, "version": 0},
                   {**valid, "f0": "yes"}, {**valid, "f0": None},
                   self.checkpoint("44k"), {**valid, "weight": None},
                   {**valid, "weight": {}},
                   {**valid, "weight": {"emb_g.weight": SimpleNamespace(shape=(0, 256))}},
                   {**valid, "weight": {"emb_g.weight": SimpleNamespace(shape=(7,))}}]
        for checkpoint in invalid:
            with self.subTest(checkpoint=checkpoint), self.assertRaises(ValueError):
                normalized_voice_config(checkpoint)


if __name__ == "__main__":
    unittest.main()
