from tribev2.demo_utils import DirectTextToEvents


def test_direct_text_events_word_schema_and_context():
    events = DirectTextToEvents(
        text="Hello, world. This is a timing test.",
        words_per_minute=180,
        max_context_words=4,
    ).get_events()

    words = events[events.type == "Word"]
    assert len(words) == 7
    assert set(words.type.unique()) == {"Word"}
    assert "Audio" not in set(events.type.unique())
    assert words.context.astype(bool).all()
    assert words.context.iloc[-1] == "is a timing test"
    assert words.iloc[-1].context.endswith(words.iloc[-1].text)
    assert words.start.is_monotonic_increasing
    assert (words.duration > 0).all()


def test_direct_text_events_punctuation_adds_timing_gap():
    events = DirectTextToEvents(text="One two. Three four.").get_events()
    words = events[events.type == "Word"].reset_index(drop=True)

    gap_with_sentence_pause = words.loc[2, "start"] - words.loc[1, "stop"]
    gap_without_sentence_pause = words.loc[1, "start"] - words.loc[0, "stop"]

    assert gap_with_sentence_pause > gap_without_sentence_pause
    assert words.loc[0, "sentence"] == "One two"
    assert words.loc[2, "sentence"] == "Three four"
