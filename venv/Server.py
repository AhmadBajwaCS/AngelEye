import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.models import Response
from fastapi.responses import FileResponse
import Driver

# $ pip install fastAPI uvicorn python-multipart
#Run server: $ uvicron Server:app --reload

app = FastAPI() #Server instance

origin = [
    "http://localhost",
    "http://localhost:4200", "*"
]

#Dealing with server connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
'''
Request Body:
- Handle file upload
- This method saves the file as "input.jpg" 
'''
@app.post("/uploadfile/")
async def create_upload(file: UploadFile = File(...)):
    #Create path in directory to upload the file to
    directory = os.path.dirname(__file__)
    file_path = os.path.join(directory, "input.jpg")

    #Write uploaded image content to image.jpg
    with open(file_path, 'wb') as buffer:
        data = await file.read()
        buffer.write(data)
    
    print("Getting File")
    return {"filename": file.filename}

'''
Response Body:
- Handle Prediciton
- This method sends to Angular:
    * (Heatmap) Image
    * Model prediction
    * Model confidence
'''
@app.get("/image")
async def get_image(model_name: str):
    Driver.visualize_heatmap('input.jpg', '{model_name}.h5')
    return FileResponse(f"output.png")