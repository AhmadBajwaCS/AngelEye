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

#Create path in directory to upload the file to
directory = os.path.dirname(__file__)
file_path = os.path.join(directory, "input.jpg")

'''
Request Body (CREATE):
- Handle file upload
- This method saves the file as "input.jpg" 
'''
@app.post("/uploadfile/")
async def create_upload(file: UploadFile = File(...)):
    #Write uploaded image content to image.jpg
    with open(file_path, 'wb') as buffer:
        data = await file.read()
        buffer.write(data)
    
    print("Getting File")
    return {"filename": file.filename}

'''
Response Body (READ):
- Handle Prediciton
- This method sends to Angular:
    * (Heatmap) Image
    * Model prediction
    * Model confidence
'''
@app.get("/image")
async def get_image(model_name: str):
    Driver.visualize_heatmap('input.jpg', '{model_name}.h5')
    return FileResponse(f"output.jpg")

'''
Update Request (UPDATE):
- Handle file update
- This method updates the "input.jpg" file
'''
@app.put("/updateImage/")
async def update_image(file: UploadFile = File(...)):
    #Write uploaded image content to image.jpg
    with open(file_path, 'wb') as buffer:
        data = await file.read()
        buffer.write(data)
    
    print("Updating File")
    return {"filename": file.filename}

'''
Delete Request (DELETE):
- Handle image deletion
- This method deletes the "input.jpg" file
'''
@app.delete("/deleteImage/")
async def delete_image():
    try:
        os.remove(file_path)
        print("File Deleted")
        file_path = os.path.join(directory, "input.jpg") #Add blank input.jpg back
        return {"message": "Files deleted successfully"}
    except FileNotFoundError:
        print("File Missing")
        return {"message": "File not found"}