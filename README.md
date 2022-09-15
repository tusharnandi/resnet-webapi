# Object detection Webapi
This is very simple AI & ML application which detect object from image and return detection result in json format.

## How Run the Application
  
If you are using Anaconda then you can install all required
Python packages by running the following commands in a shell:

```
   conda create --name tf python=3
   activate tf
   pip install -r requirements.txt
```


now run the application as below:
```
python webapi.py
```
#### Note that you have to edit this file to select whether you want to install the CPU or GPU version of TensorFlow.

##Docker image

```
docker run --publish 5000:5000 tusharkantinandi/python-predection:v1.0.0
```
