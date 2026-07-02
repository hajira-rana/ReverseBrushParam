# Inverse Brush Parameter Determination
## Abstract
Brush engines compose of the core components of digital and graphic art software. Industry leads such as Adobe Photoshop, Clip Studio Paint, Paint Tool SAI, and Procreate maintain closed brush engines with their own systems for calculating how a brush stroke is displayed based on select parameters. The purpose of an individual engine's parameters is to create a complex, realistic brush, mimicking the texture and dynamics of how a real drawing utensil would interact with a physical medium. Here I propose a method of determining brush parameters from an image of a brush stroke.
## Intro
Brush engines work by repeatedly stamping an image, called a brush tip, over a path, called a stroke. Like drawing in physical mediums, factors such as pressure, speed and tilt of the user's input adjust how the stamp is calculated and displayed onto the canvas. Brush engines in different graphic software vary in parameters as well as how those parameters are used to calculate the final mark. My aim is to create a brush generator that can recreate brushes based on an image of the stroke drawn. This project will use machine learning to classify brush parameters based on an image of a brush stroke. Testing and training data will all be done digitally however, ideally, the user will be able to take a picture of marks made by a real drawing utensil and use the machine learning model to attain an accurate digital representation of the tool.
## Dataset
I was able to use Adobe Photoshop's Batchplay Api to iterate through each parameter value, apply the brush to a predetermined stroke, and export the image. While rendering the training data I saw how many of the brushes with very small pixel sizes looked both very similar, which and caused the model to struggle with determining values, as the features were obfuscated. I made the decision to filter out brushes in my dataset smaller than 5 pixels. Before filtering out smaller values I ended up with over 25,000 images. 
## Approach
Photoshop only requires one parameter for all brushes, the Brush Tip Shape. The tip shape has several sub-parameters such as size, roundness, hardness, angle and spacing. Photoshop also offers several optional parameters categories such as, Shape Dynamics, Scattering, Texture, Color Dynamics, and Transfer. These parameters applies to the brush itself. There are also general tool parameters, such as Opacity, Flow, and Color Blending mode that apply to the current tool being used and do not affect the brush's identity. The first level of implementation will only deal with brushes that use the standard round tip with base parameters. Currently this project supports prediction of Size(px), Hardness(\%0-100), Spacing(\%0-100), and roundness(\%0-100) of the brush.
I utilize three convolution blocks; Conv1 - 32, Conv2 - 64, Conv3 - 128. Connected to a Multi-Layer Perception model that outputs predicted values for the four sub-parameters. 
## Future Work and Limitations
Currently this project only has the ability to determine four parameters from a pre-determined stroke which greatly limits the functionality in real life applications. Before adding optional parameter determination I will implement a segmentation step that selects the best path and patch o stroke possible for accurate determination. 

After segmentation The next will be to add a generative step that can determine and recreate brush tips.

## Usage
To run a sample dataset:

Download dataset from [Google Drive](https://drive.google.com/drive/folders/1eVD8lLuPivvfH75dZJ0UsPeuSivRzFvH?usp=sharing) and move to project folder.

Install packages (preferably in a venv) using pip install -r requirements.txt

run Cnn.py

When prompted enter N to run 20 samples.
