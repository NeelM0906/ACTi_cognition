import pandas as pd

from tribev2.eventstransforms import AssignSplitByEpisode


def test_assign_split_by_episode_holds_out_all_chunks_for_episode():
    events = pd.DataFrame(
        {
            "type": ["Fmri", "Word", "Fmri", "Word", "Fmri", "Word"],
            "movie": ["s01", "s01", "s01", "s01", "s02", "s02"],
            "chunk": ["e01a", "e01b", "e02a", "e02b", "e01a", "e01b"],
            "start": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "duration": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "timeline": ["a", "a", "b", "b", "c", "c"],
        }
    )
    out = AssignSplitByEpisode(val_episodes=["s01e01"])._run(events)

    assert set(out.loc[out["chunk"].isin(["e01a", "e01b"]), "split"]) == {
        "train",
        "val",
    }
    assert set(
        out.loc[(out["movie"] == "s01") & out["chunk"].isin(["e01a", "e01b"]), "split"]
    ) == {"val"}
    assert set(out.loc[out["movie"] == "s02", "split"]) == {"train"}


def test_assign_split_by_episode_accepts_episode_only_ids():
    events = pd.DataFrame(
        {
            "type": ["Fmri", "Fmri", "Fmri"],
            "movie": ["s01", "s01", "s01"],
            "chunk": ["e01a", "e01b", "e02a"],
            "start": [0.0, 0.0, 0.0],
            "duration": [1.0, 1.0, 1.0],
            "timeline": ["a", "a", "b"],
        }
    )
    out = AssignSplitByEpisode(val_episodes=["e01"])._run(events)

    assert set(out.loc[out["chunk"].isin(["e01a", "e01b"]), "split"]) == {"val"}
    assert set(out.loc[out["chunk"] == "e02a", "split"]) == {"train"}


def test_assign_split_by_episode_infers_missing_fields_from_timeline():
    events = pd.DataFrame(
        {
            "type": ["Fmri", "Sentence", "Word", "Fmri"],
            "movie": ["s01", None, "s01", "s01"],
            "chunk": ["e07a", None, "e07a", "e06a"],
            "start": [0.0, 0.0, 0.2, 0.0],
            "duration": [1.0, 0.5, 0.2, 1.0],
            "timeline": ["tl0", "tl0", "tl0", "tl1"],
        }
    )
    out = AssignSplitByEpisode(val_episodes=["s01e07"])._run(events)

    assert set(out.loc[out["timeline"] == "tl0", "split"]) == {"val"}
    assert set(out.loc[out["timeline"] == "tl1", "split"]) == {"train"}
