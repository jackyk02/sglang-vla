from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
import uvicorn
import numpy as np
import sglang as sgl
from token2action import TokenActionConverter, image_qa
from typing import List, Optional
import json_numpy as json

app = FastAPI()
converter = TokenActionConverter()

# Initialize sglang backend
sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

class BatchRequest(BaseModel):
    instruction: str
    image_path: str
    batch_size: Optional[int] = 4
    temperature: Optional[float] = 1.0

def process_batch(batch_size: int, temp: float, instruction: str, image_path: str) -> List[np.ndarray]:
    """Process a batch of robot actions."""
    arguments = [
        {
            "image_path": image_path,
            "question": f"In: What action should the robot take to {instruction}?\nOut:",
        }
    ] * batch_size
    
    states = image_qa.run_batch(
        arguments,
        max_new_tokens=7,
        temperature=temp
    )
    
    # Convert to numpy arrays instead of lists
    return [np.array(s.get_meta_info("action")["output_ids"])
            for s in states], [np.array(converter.token_to_action(s.get_meta_info("action")["output_ids"]))
            for s in states]

@app.get("/")
async def read_root():
    return {"message": "Batch processing server is running"}

@app.post("/batch")
async def process_batch_request(request: Request):
    try:
        body = await request.body()
        data = json.loads(body)
        
        # Validate required fields
        if not isinstance(data.get("instruction"), str):
            raise HTTPException(status_code=400, detail="Instruction must be a string")
        if not isinstance(data.get("image_path"), str):
            raise HTTPException(status_code=400, detail="Image path must be a string")
            
        # Get optional parameters with defaults
        batch_size = int(data.get("batch_size", 4))
        temperature = float(data.get("temperature", 1.0))
        
        # Process the batch
        output_ids, actions = process_batch(
            batch_size=batch_size,
            temp=temperature,
            instruction=data["instruction"],
            image_path=data["image_path"]
        )
        
        # Convert numpy arrays to json-serializable format using json_numpy
        response_data = {"output_ids": output_ids, "actions": actions}
        return Response(content=json.dumps(response_data), media_type="application/json")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3200)