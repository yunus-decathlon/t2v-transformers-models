import os
from logging import getLogger
from fastapi import FastAPI, Response, status
from transformer import Transformer, TextInput, TextsInput


app = FastAPI()
logger = getLogger('uvicorn')

@app.on_event("startup")
def startup_event():
    global encoder
    model_path = "/./models/trans" #we assume the model was already downloaded
    device = os.environ.get("DEVICE") # device can be any pytorch device. 
    # Current implementation should detect cuda devices automatically
    encoder = Transformer().load_model(model_path, device)

    cuda_env = os.getenv("ENABLE_CUDA")
    cuda_per_process_memory_fraction = 1.0
    if "CUDA_PER_PROCESS_MEMORY_FRACTION" in os.environ:
        try:
            cuda_per_process_memory_fraction = float(os.getenv("CUDA_PER_PROCESS_MEMORY_FRACTION"))
        except ValueError:
            logger.error(f"Invalid CUDA_PER_PROCESS_MEMORY_FRACTION (should be between 0.0-1.0)")
    if 0.0 <= cuda_per_process_memory_fraction <= 1.0:
        logger.info(f"CUDA_PER_PROCESS_MEMORY_FRACTION set to {cuda_per_process_memory_fraction}")
    cuda_support=False
    cuda_core=""
    mps_env = os.getenv("ENABLE_MPS")

    
@app.get("/.well-known/live", response_class=Response)
@app.get("/.well-known/ready", response_class=Response)
async def live_and_ready(response: Response):
    response.status_code = status.HTTP_204_NO_CONTENT


@app.post("/texts/") #future support for /medias/
# Example Input JSON
# '{"text": ["cats are better than dogs", "and better than humans"]}'
def read_items(item: TextsInput, response: Response):
    return read_item(item, response)

@app.post("/vectors")
@app.post("/vectors/")
# Example Input JSON
# '{"text": "cats are better than dogs"}'
def read_item(item: TextInput, response: Response):
    try:
        vector = encoder(item.text)
        return {"text": item.text, "vector": vector.tolist(), "dim": len(vector)}
    except Exception as e:
        logger.exception('Something went wrong while vectorizing data.')
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": str(e)}
