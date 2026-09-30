#!/usr/bin/env python3
"""
Step 1 - Download run metadata and genus-level taxonomic profiles from GMrepo
(https://gmrepo.humangut.info) for shotgun-metagenomics CRC case-control cohorts.

Selection:
  * phenotype MeSH D015179 (Colorectal Neoplasms) vs D006262 (Health)
  * experiment_type = Metagenomics, GMrepo QCStatus = 1 (passed)
  * 8 cohorts that contain both CRC and healthy samples (see COHORTS)
  * one run per biological sample (first run of each sample_id)

Outputs: data/raw/samples.tsv and data/raw/profiles/<run_id>.json
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import requests

API = "https://gmrepo.humangut.info/api/"
COHORTS = {
    "PRJEB6070":   "Zeller 2014, France",
    "PRJEB7774":   "Feng 2015, Austria",
    "PRJEB10878":  "Yu 2017, China (Hong Kong)",
    "PRJEB12449":  "Vogtmann 2016, USA",
    "PRJEB27928":  "Wirbel 2019, Germany",
    "PRJNA447983": "Thomas 2019, Italy",
    "PRJNA731589": "China cohort (2021)",
    "PRJNA763023": "Young-onset CRC, China (2021)",
}
MESH = {"CRC": "D015179", "Healthy": "D006262"}
OUT = "data/raw"


def post(endpoint, payload, tries=6):
    for i in range(tries):
        try:
            r = requests.post(API + endpoint + "/", json=payload, timeout=180)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # network hiccups / server 5xx
            print(f"  retry {endpoint}: {e}", file=sys.stderr)
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"failed: {endpoint} {payload}")


def all_runs(mesh_id, page=2000):
    runs, skip = [], 0
    while True:
        batch = post("getAssociatedRunsByPhenotypeMeshIDLimit", {"mesh_id": mesh_id, "skip": skip, "limit": page})
        runs += batch
        skip += len(batch)
        print(f"  {mesh_id}: {skip} runs", file=sys.stderr)
        if len(batch) < page:
            return runs


def main():
    os.makedirs(f"{OUT}/profiles", exist_ok=True)
    samples = {}
    for label, mesh in MESH.items():
        for x in all_runs(mesh):
            ok = (x["experiment_type"] or "").lower().startswith("metagenom") and x["QCStatus"] == 1
            if ok and x["project_id"] in COHORTS and x["sample_id"] not in samples:
                samples[x["sample_id"]] = dict(run_id=x["run_id"], sample_id=x["sample_id"],
                                               project_id=x["project_id"], country=x["country"], label=label)
    print(f"selected {len(samples)} samples", file=sys.stderr)

    with open(f"{OUT}/samples.tsv", "w") as f:
        f.write("run_id\tsample_id\tproject_id\tcountry\tlabel\n")
        for s in samples.values():
            f.write("\t".join(str(s[k]) for k in ("run_id", "sample_id", "project_id", "country", "label")) + "\n")

    def fetch(s):
        path = f"{OUT}/profiles/{s['run_id']}.json"
        if not os.path.exists(path):
            d = post("getFullTaxonomicProfileByRunID", {"run_id": s["run_id"]})
            json.dump({"genus": d.get("genus", [])}, open(path, "w"))

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(fetch, samples.values()))


if __name__ == "__main__":
    main()
