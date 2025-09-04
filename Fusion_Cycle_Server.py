import argparse
import copy
import logging
import os
import sys
import pandas as pd
import traceback
import uvicorn
import Fusion_Cycle
from datetime import datetime
from fastapi import FastAPI
from fastapi.routing import APIRouter
from pydantic import BaseModel
from rdkit import RDLogger
from typing import List

app = FastAPI()
router = APIRouter()

base_response = {
    "status": "FAIL",
    "error": "",
    "results": []
}


def parse_args():
    parser = argparse.ArgumentParser("Fusion_Cycle_server")
    parser.add_argument("--server_ip", help="Server IP to use", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", help="Server port to use", type=int, default=9601)
    parser.add_argument("--log_file", help="Log file", type=str, default="Fusion_Cycle_server")

    return parser.parse_args()


class RequestBody(BaseModel):
    solute_smiles: List[str]
    solvent_smiles: List[str]
    temperature: List[float]
    density: List[float]

@app.post("/Fusion_Cycle")
def fusion_cycle_service(request_json: RequestBody):
    response = copy.deepcopy(base_response)

    try:
        results = []
        for solute, solvent, T, rho in zip(request_json.solute_smiles,
                                   request_json.solvent_smiles,
                                   request_json.temperature,
                                   request_json.density):
            dp = pd.DataFrame({'solute_smiles_canonical':[solute],
                               'solvent_smiles_canonical':[solvent],
                               'Temperature [K]':[T],
                               'solvent_density':[rho]})
            results.append(fusion_cycle.calculate_solubility(dp))
        response["results"] = results
        response["status"] = "SUCCESS"
        return response

    except Exception:
        response["error"] = f"Error during fusion cycle calculation, traceback: " \
                            f"{traceback.format_exc()}"
        traceback.print_exc()

        return response


app.include_router(router)


if __name__ == "__main__":
    args = parse_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")

    os.makedirs(f"./logs", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"./logs/{args.log_file}.{dt}.log")
    sh = logging.StreamHandler(sys.stdout)
    fh.setLevel(logging.INFO)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # set up model
    fusion_cycle = Fusion_Cycle.model()

    # start running
    uvicorn.run(app, host=args.server_ip, port=args.server_port)
