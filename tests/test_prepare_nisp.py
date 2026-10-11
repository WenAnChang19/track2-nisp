import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_nisp  # noqa: E402


HEADER = (
    "Speaker_ID Gender Mother_Tongue Height Shoulder_size Waist_size "
    "Weight Age Native_State Native_District"
)


class NispFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.metadata = root / "metadata"
        self.audio = root / "audio"
        self.metadata.mkdir(parents=True)
        self.audio.mkdir()

    def write_metadata(self, rows):
        text = HEADER + "\n" + "\n".join(
            f"{speaker_id} {sex} Hindi {height} 40 80 {weight} {age} State District"
            for speaker_id, sex, height, weight, age in rows
        ) + "\n"
        (self.metadata / "total_spkrinfo.list").write_text(text, encoding="utf-8")

    def write_splits(self, train, test):
        (self.metadata / "train_spkrID").write_text(
            "".join(f"{speaker_id}___TRN\n" for speaker_id in train), encoding="utf-8"
        )
        (self.metadata / "test_spkrID").write_text(
            "".join(f"{speaker_id}___TST\n" for speaker_id in test), encoding="utf-8"
        )

    def add_wav(self, speaker_id, utterance="1", language="Eng", folder="RECS/1"):
        directory = self.audio / folder
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{speaker_id}_{language}_m_{utterance}.wav"
        path.write_bytes(b"")
        return path


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class PrepareNispTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = NispFixture(self.root)
        self.rows = [
            ("Hin_1", "Female", "200", "100", "20"),
            ("Hin_2", "Female", "160", "64", "21"),
            ("Hin_3", "Female", "170", "60", "22"),
            ("Hin_4", "Female", "165", "55", "23"),
            ("Kan_10", "Male", "180", "81", "24"),
            ("Kan_11", "Male", "175", "70", "25"),
            ("Kan_12", "Male", "190", "90", "26"),
            ("Kan_13", "Male", "185", "85", "27"),
            ("Tam_101", "Female", "155", "50", "28"),
            ("Tel_9", "Male", "172", "72", "29"),
        ]
        self.train = [item[0] for item in self.rows[:8]]
        self.test = [item[0] for item in self.rows[8:]]
        self.fixture.write_metadata(self.rows)
        self.fixture.write_splits(self.train, self.test)
        for speaker_id in self.train + self.test:
            self.fixture.add_wav(speaker_id)

    def tearDown(self):
        self.temporary.cleanup()

    def test_builds_stratified_disjoint_manifests_and_bmi(self):
        self.fixture.add_wav("Hin_1", utterance="2")
        self.fixture.add_wav("Hin_1", language="Hin", utterance="excluded")

        audit = prepare_nisp.prepare(self.root, val_fraction=0.25, seed=42)

        assignments = audit["derived_split_assignments"]
        train_ids = set(assignments["train"])
        val_ids = set(assignments["val"])
        test_ids = set(assignments["test"])
        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(val_ids & test_ids)
        self.assertEqual(test_ids, set(self.test))
        self.assertEqual(
            sorted(self._sex_for(speaker_id) for speaker_id in val_ids),
            ["Female", "Male"],
        )

        all_rows = read_csv(self.root / "manifests/all.csv")
        self.assertEqual({row["speaker_id"] for row in all_rows if row["split"] == "test"}, set(self.test))
        hin_1 = next(row for row in all_rows if row["speaker_id"] == "Hin_1")
        self.assertEqual(hin_1["bmi"], "25.000000")
        self.assertEqual(hin_1["height_cm"], "200")
        self.assertEqual(audit["exclusions"]["non_english_wav"]["count"], 1)

    def test_rerun_is_byte_for_byte_deterministic(self):
        prepare_nisp.prepare(self.root, val_fraction=0.25, seed=7)
        output = self.root / "manifests"
        first = {path.name: path.read_bytes() for path in output.iterdir()}

        prepare_nisp.prepare(self.root, val_fraction=0.25, seed=7)
        second = {path.name: path.read_bytes() for path in output.iterdir()}

        self.assertEqual(first, second)

    def test_cli_reports_each_split(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = prepare_nisp.main(
                ["--root", str(self.root), "--val-fraction", "0.25", "--seed", "42"]
            )

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        for split in ("train", "val", "test"):
            self.assertIn(f"{split}:", output)
        self.assertIn("missing speakers", output)

    def test_csv_write_failure_does_not_leave_success_audit(self):
        stderr = io.StringIO()
        with mock.patch.object(prepare_nisp, "write_csv", side_effect=OSError("disk full")):
            with redirect_stderr(stderr):
                exit_code = prepare_nisp.main(
                    ["--root", str(self.root), "--val-fraction", "0.25", "--seed", "42"]
                )

        self.assertEqual(exit_code, 2)
        self.assertIn("disk full", stderr.getvalue())
        audit = json.loads((self.root / "manifests/audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["status"], "preparing")
        self.assertFalse(audit["manifests_written"])

    def test_unknown_audio_speaker_fails_and_is_audited(self):
        self.fixture.add_wav("Mal_999", utterance="unknown")

        with self.assertRaisesRegex(prepare_nisp.PreparationError, "unsafe audio mismatch"):
            prepare_nisp.prepare(self.root, val_fraction=0.25, seed=42)

        audit = json.loads((self.root / "manifests/audit.json").read_text(encoding="utf-8"))
        unknown = audit["anomalies"]["unknown_audio_speakers"]
        self.assertEqual(unknown["speaker_ids"], ["Mal_999"])
        self.assertEqual(audit["status"], "error")
        self.assertFalse(audit["manifests_written"])

    def test_split_with_no_recorded_speakers_fails(self):
        for speaker_id in self.test:
            for path in self.fixture.audio.rglob(f"{speaker_id}_*.wav"):
                path.unlink()

        with self.assertRaisesRegex(prepare_nisp.PreparationError, "may be stale"):
            prepare_nisp.prepare(self.root, val_fraction=0.25, seed=42)

        audit = json.loads((self.root / "manifests/audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["anomalies"]["empty_recorded_splits"]["splits"], ["test"])
        self.assertEqual(audit["counts_by_split"]["test"]["recorded_speakers"], 0)

    def test_official_train_test_overlap_fails(self):
        self.fixture.write_splits(self.train, self.test + [self.train[0]])

        with self.assertRaisesRegex(prepare_nisp.PreparationError, "train/test overlap"):
            prepare_nisp.prepare(self.root, val_fraction=0.25, seed=42)

    def test_missing_recordings_remain_in_assignments_and_audit(self):
        missing_id = self.train[0]
        for path in self.fixture.audio.rglob(f"{missing_id}_*.wav"):
            path.unlink()

        audit = prepare_nisp.prepare(self.root, val_fraction=0.25, seed=42)

        derived = audit["derived_split_assignments"]
        assigned_split = next(split_name for split_name, ids in derived.items() if missing_id in ids)
        self.assertIn(missing_id, audit["speakers_with_no_recordings"][assigned_split])
        speaker_rows = read_csv(self.root / "manifests/speakers.csv")
        self.assertNotIn(missing_id, {row["speaker_id"] for row in speaker_rows})
        self.assertIn(missing_id, audit["official_split_assignments"]["train"])

    def _sex_for(self, speaker_id):
        return next(sex for current_id, sex, *_ in self.rows if current_id == speaker_id)


if __name__ == "__main__":
    unittest.main()
