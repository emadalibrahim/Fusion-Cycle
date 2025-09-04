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
import os
import torch

# Detect number of available CUDA devices
num_devices = torch.cuda.device_count()

if num_devices > 0:
    # If GPUs exist, pick device 0 by default
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"Using GPU device 0 (total available: {num_devices})")
else:
    # No GPUs → run on CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("No GPU detected, running on CPU")

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
        logging.info("Received request: %s", request_json.dict())
        results = []

        for solute, solvent, T, rho in zip(
            request_json.solute_smiles,
            request_json.solvent_smiles,
            request_json.temperature,
            request_json.density,
        ):
            dp = pd.DataFrame(
                {
                    "solute_smiles_canonical": [solute],
                    "solvent_smiles_canonical": [solvent],
                    "Temperature [K]": [T],
                    "solvent_density": [rho],
                }
            )
            logging.info("Input dataframe:\n%s", dp)

            try:
                result = fusion_cycle.calculate_solubility(dp)
                results.append(result)
            except Exception as inner_e:
                logging.error("Error in calculate_solubility: %s", traceback.format_exc())
                response["error"] = f"calculate_solubility failed: {inner_e}"
                return response

        response["results"] = results
        response["status"] = "SUCCESS"
        return response

    except Exception:
        logging.error("Error in fusion_cycle_service: %s", traceback.format_exc())
        response["error"] = f"Error during fusion cycle calculation, traceback: {traceback.format_exc()}"
        return response


app.include_router(router)


if __name__ == "__main__":
    args = parse_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")

    os.makedirs("./logs", exist_ok=True)
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
    try:
        fusion_cycle = Fusion_Cycle.model()
        logging.info("Fusion_Cycle model loaded: %s", type(fusion_cycle))
    except Exception:
        logging.error("Failed to initialize Fusion_Cycle model: %s", traceback.format_exc())
        sys.exit(1)

    # start running
    uvicorn.run(app, host=args.server_ip, port=args.server_port)
