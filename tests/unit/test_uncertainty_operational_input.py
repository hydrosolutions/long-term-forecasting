"""Recent-input diagnostics must not reach inference without complete rows."""

import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from lt_forecasting.forecast_models.deep_models.uncertainty_mixture import (
    UncertaintyMixtureModel,
)


@pytest.mark.parametrize(
    "dates,features,expected",
    [
        (
            ["2026-10-01"] * 2,
            {"snow": [1, 2], "temperature": [3, 4]},
            "no recent input",
        ),
        (
            ["2026-10-29"] * 2,
            {"snow": [None, None], "temperature": [3, 4]},
            "all-null features: ['snow']",
        ),
        (["2026-10-29"] * 2, {"temperature": [3, 4]}, "absent columns: ['snow']"),
        (
            ["2026-10-29"] * 2,
            {"snow": [1, None], "temperature": [None, 4]},
            "no complete recent input",
        ),
    ],
)
def test_missing_recent_inputs(dates, features, expected, caplog):
    model = make_model(dates, features)
    result = UncertaintyMixtureModel.predict_operational(
        model, datetime.datetime(2026, 10, 30)
    )
    assert result.empty
    model.predict.assert_not_called()
    assert expected in caplog.text
    assert "skipped operational prediction" in caplog.text


def make_model(dates, features):
    return SimpleNamespace(
        name="MC_ALD",
        data=pd.DataFrame({"code": [1, 1], "date": pd.to_datetime(dates), **features}),
        features=["snow", "temperature"],
        general_config={"prediction_horizon": 30},
        __preprocess_data__=Mock(),
        load_model=Mock(),
        predict=Mock(return_value=pd.DataFrame()),
        model=object(),
        scaler=object(),
    )


def test_valid_recent_input_reaches_prediction():
    model = make_model(
        ["2026-10-28", "2026-10-29"],
        {"snow": [1, 2], "temperature": [3, 4]},
    )
    UncertaintyMixtureModel.predict_operational(model, datetime.datetime(2026, 10, 30))
    model.predict.assert_called_once()
    selected = model.predict.call_args.kwargs["df"]
    assert len(selected) == 1
    assert selected.iloc[0]["date"] == pd.Timestamp("2026-10-29")
