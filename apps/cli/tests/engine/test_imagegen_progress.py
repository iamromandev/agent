"""ProgressParser and format_progress: mflux's tqdm stderr into display-only progress values."""

from __future__ import annotations

from src.engine.imagegen import ImageProgress, ProgressParser, clock, format_progress

S0 = b"  0%|          | 0/9 [00:00<?, ?it/s]\r"
S3 = b" 33%|\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x8e      | 3/9 [01:14<02:26, 24.35s/it]\r"
S9 = b"100%|\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88| 9/9 [03:36<00:00, 23.75s/it]\r"


def test_start_is_loading_or_downloading() -> None:
    assert ProgressParser(9, False).start() == ImageProgress("loading")
    assert ProgressParser(9, True).start() == ImageProgress("downloading")


def test_step_zero_keeps_the_phase() -> None:
    assert ProgressParser(9, False).feed(S0) == []


def test_a_step_line_gives_step_and_left() -> None:
    assert ProgressParser(9, False).feed(S3) == [ImageProgress("steps", 3, 9, 146)]


def test_unknown_left_is_none() -> None:
    line = b" 11%|#1        | 1/9 [00:09<?, 9.00s/it]\r"
    assert ProgressParser(9, False).feed(line) == [ImageProgress("steps", 1, 9, None)]


def test_last_step_is_followed_once_by_saving_even_when_printed_twice() -> None:
    values = ProgressParser(9, False).feed(S9 + S9 + b"\n")
    assert values == [ImageProgress("steps", 9, 9, 0), ImageProgress("saving")]


def test_hour_clocks() -> None:
    line = b" 11%|#1        | 1/9 [1:00:09<1:02:03, 9.00s/it]\r"
    assert ProgressParser(9, False).feed(line)[0].left == 3723
    assert clock(59) == "0:59"
    assert clock(3600) == "1:00:00"
    assert clock(3723) == "1:02:03"


def test_split_chunks_wait_for_a_complete_bar() -> None:
    parser = ProgressParser(9, False)
    cut = S3.index(b"<") + 3  # inside the left clock
    assert parser.feed(S3[:20]) == []
    assert parser.feed(S3[20:cut]) == []
    assert parser.feed(S3[cut:]) == [ImageProgress("steps", 3, 9, 146)]


def test_an_unterminated_bar_is_reported_at_once() -> None:
    parser = ProgressParser(9, False)
    assert parser.feed(b"\r 1%|x| 1/9 [00:26<03:34, 26.78s/it]") == [ImageProgress("steps", 1, 9, 214)]
    assert parser.feed(b"") == []  # the same tail again gives nothing
    assert parser.feed(b"\r 22%|x| 2/9 [00:50<02:50, 1s/it]") == [ImageProgress("steps", 2, 9, 170)]


def test_a_bar_cut_mid_number_gives_nothing_wrong() -> None:
    for cut in (b"\r 11%|x| 1/", b"\r 11%|x| 1/9 [00:2", b"\r 11%|x| 1/9 [00:26<03:3", b"\r 11%|x| 1/9 [00:26<03:34"):
        assert ProgressParser(9, False).feed(cut) == []


def test_the_real_tqdm_byte_sequence() -> None:
    parser = ProgressParser(9, False)
    out = parser.feed(b"\r  0%|          | 0/9 [00:00<?, ?it/s]")
    out += parser.feed("\r 11%|█         | 1/9 [00:26<03:34, 26.78s/it]".encode())
    out += parser.feed("\r 22%|██▏       | 2/9 [00:53<03:07, 26.8s/it]".encode())
    assert out == [ImageProgress("steps", 1, 9, 214), ImageProgress("steps", 2, 9, 187)]


def test_huge_digit_runs_do_not_raise() -> None:
    assert ProgressParser(9, False).feed(b"\r 1%|x| " + b"9" * 5000 + b"/9 [00:26<03:34, 1s/it]") == []


def test_download_then_other_output_switches_to_loading() -> None:
    parser = ProgressParser(9, True)
    bar = b"\rmodel.safetensors: 100%|####| 2.3G/5.5G [00:30<00:41, 77MB/s]"
    assert parser.feed(bar) == [ImageProgress("downloading", percent=100)]
    assert parser.feed(b"\n") == [ImageProgress("loading")]
    assert parser.feed(b"Loading weights\n") == []


def test_other_output_after_a_download_bar_switches_to_loading() -> None:
    parser = ProgressParser(9, True)
    parser.feed(b"\rmodel.safetensors: 42%|####| 2.3G/5.5G [00:30<00:41, 77MB/s]\r")
    assert parser.feed(b"Loading weights\n") == [ImageProgress("loading")]


def test_without_download_output_the_phase_stays_downloading() -> None:
    parser = ProgressParser(9, True)
    assert parser.feed(b"Loading weights\n") == []
    assert parser.feed(b"Fetching 12 files: 5/12 weird\n") == []


def test_garbage_and_unrelated_lines_give_nothing() -> None:
    assert ProgressParser(9, False).feed(b"\xff\xfe\x00 hello\nLoading weights\n 50%|#####| 5/\n") == []


def test_wrong_total_decreasing_and_prefixed_lines_are_ignored() -> None:
    parser = ProgressParser(9, False)
    assert parser.feed(b"100%|##########| 3/3 [00:03<00:00, 1.00s/it]\r") == []
    assert parser.feed(S3) != []
    assert parser.feed(b" 11%|#1 | 1/9 [00:01<00:08, 1.00s/it]\r") == []
    assert parser.feed(b"x 66%|#1 | 6/9 [00:01<00:08, 1.00s/it]\r") == []
    assert parser.feed(b" 66%|#1 | 10/9 [00:01<00:08, 1.00s/it]\r") == []


def test_download_percent_before_the_first_step_only() -> None:
    parser = ProgressParser(9, True)
    line = b"model.safetensors:  42%|####2     | 2.3G/5.5G [00:30<00:41, 77MB/s]\r"
    assert parser.feed(line) == [ImageProgress("downloading", percent=42)]
    assert parser.feed(S3) == [ImageProgress("steps", 3, 9, 146)]
    assert parser.feed(line) == []


def test_download_lines_are_ignored_when_not_downloading() -> None:
    assert ProgressParser(9, False).feed(b"model.safetensors:  42%|####2     | 2.3G/5.5G [00:30<00:41, 77MB/s]\r") == []


def test_unparseable_download_output_gives_nothing() -> None:
    assert ProgressParser(9, True).feed(b"Fetching 12 files: 5/12 weird\n") == []


def test_the_unfinished_tail_is_capped() -> None:
    parser = ProgressParser(9, False)
    assert parser.feed(b"a" * 100_000) == []
    assert len(parser._tail) <= 4096
    assert parser.feed(b"\n" + S3) == [ImageProgress("steps", 3, 9, 146)]


def test_format_texts() -> None:
    assert format_progress(ImageProgress("starting"), 1) == "starting… 0:01"
    assert format_progress(ImageProgress("downloading"), 65) == "downloading model… 1:05"
    assert format_progress(ImageProgress("downloading", percent=42), 65) == "downloading model… 42% · 1:05"
    assert format_progress(ImageProgress("loading"), 12) == "loading model… 0:12"
    assert format_progress(ImageProgress("steps", 3, 9, 146), 74) == "step 3/9 · 1:14 elapsed · ~2:26 left"
    assert format_progress(ImageProgress("steps", 3, 9, None), 74) == "step 3/9 · 1:14 elapsed"
    assert format_progress(ImageProgress("saving"), 220) == "saving… 3:40"


def test_format_is_cut_to_80_characters() -> None:
    assert len(format_progress(ImageProgress("steps", 10**80, 10**80, 10**80), 1)) == 80
