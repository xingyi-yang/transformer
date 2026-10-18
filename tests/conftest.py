from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from fixtures import messy_send  # noqa: E402

from send_bw.data.generate import generate_dataset  # noqa: E402
from send_bw.data.harmonize import harmonize  # noqa: E402
from send_bw.data.metadata import load_mapping  # noqa: E402
from send_bw.data.sources import LocalSendSource  # noqa: E402

MAPPING = ROOT / "configs" / "mappings" / "example_send.yaml"


@pytest.fixture(scope="session")
def mapping():
    return load_mapping(MAPPING)


@pytest.fixture(scope="session")
def messy(tmp_path_factory, mapping):
    root = tmp_path_factory.mktemp("messy")
    paths = messy_send.build(root)
    src = LocalSendSource(paths["send"], metadata_dir=paths["metadata"], domain_aliases=mapping.domain_aliases)
    tables, report = harmonize(src, mapping)
    return {"paths": paths, "source": src, "tables": tables, "report": report}


@pytest.fixture(scope="session")
def synth(tmp_path_factory, mapping):
    root = tmp_path_factory.mktemp("synth")
    info = generate_dataset({"n_studies": 10, "n_per_sex": [4, 5]}, root, seed=7)
    src = LocalSendSource(info["send_dir"], metadata_dir=info["metadata_dir"], domain_aliases=mapping.domain_aliases)
    tables, report = harmonize(src, mapping)
    return {"info": info, "root": root, "tables": tables, "report": report, "source": src}


@pytest.fixture(scope="session")
def trained(synth):
    """Small spline + transformer fitted on frozen study splits of the synthetic fixture."""
    from send_bw.data.features import FeatureSpec
    from send_bw.data.qc import CohortConfig, select_cohort
    from send_bw.data.splits import make_study_splits
    from send_bw.models.base import ModelData
    from send_bw.models.spline import SplineModel
    from send_bw.models.transformer import TransformerModel

    t = synth["tables"]
    c = select_cohort(t, CohortConfig(max_day=28))
    sc = t["study_context"][t["study_context"]["SPECIES"] == "RAT"]
    sp = make_study_splits(sc, {"validation": 0.2, "test": 0.2}, seed=3)
    tr_ids = sp[sp.SPLIT == "train"].STUDYID
    cref = t["compound_reference"]
    spec = FeatureSpec.fit(c.animals[c.animals.STUDYID.isin(tr_ids)], c.obs[c.obs.STUDYID.isin(tr_ids)], cref, "RAT")
    full = ModelData(c.animals, c.obs, cref, spec)
    data = {s: full.subset(sp[sp.SPLIT == s].STUDYID) for s in ("train", "validation", "test")}
    sm = SplineModel()
    sm.fit(data["train"], data["validation"], {"n_bootstrap": 3, "seed": 1})
    tm = TransformerModel()
    tm.fit(data["train"], data["validation"], {"shared_covariance": sm.cov, "n_members": 2, "max_epochs": 15,
                                               "patience": 5, "seed": 1})
    return {"spline": sm, "transformer": tm, "data": data, "splits": sp}
