2024_2025_Prog_Test


Chest X-Ray Graph Network and AI Development Test
This repository provides an educational and practical test designed for participants to learn and apply concepts in radiology, graph database structuring, and AI algorithm development. By working with anonymized chest X-ray images and corresponding radiologist reports, testers will gain hands-on experience in extracting, organizing, and analyzing medical data to develop AI systems.

Project Goals
1 Education: Teach participants how to interpret chest X-rays and extract key information from radiologists' reports.

2 Data Structuring: Develop a graph database from radiologists' reports, capturing essential details such as disease name, location, size, number of abnormalities, and more.

3 AI Development: Utilize the graph database and chest X-ray images to build multiple specialized AI models.

4 Feasibility Showcase: Demonstrate the potential of well-structured datasets and targeted AI models for medical applications.


Step by Step Approach

Step 1: Organize Radiologist Reports into a Graph Database

This is implemented with 'Step1.py'. The breakdown of the code is in test.ipynb. The code does the following:

(1) Load the original X-ray report data. 

(2) Extract medical terms from all the reports in the original data using the spacy "en_ner_bc5cdr_md" model.

(3) Define the medical terms for diseases, abnormalities, lesions, and locations by manually inspecting the terms extracted from all the reports.

(4) Extract the diseases, abnormalities, lesions, locations and sizes from each report by match the defined medical terms to each report.

(5) Construct the graphic network by defining the nodes and edges for each report. Save each graph to a single json file.

(6) Visualize the network graph and the corresponding X-ray image.



Step 2 Demonstration the usage of the graph database
The restructured graph database is stored as the file 'all_graphs.json'. By running the code in 'Step2.py'， the use extract the graph of any patient in the database and visualize the diseases, the abnormalities, the lesions, and the sizes, and locations of the abnormalities or the lesions. The x-ray image and the full report.
 
Take the patient_0472 as an example, all the information is stored and can be extracted from the 'all_graoh.json'. By running the 'Step2.py', the FULL_REPORT of this patient can be printed, and the network graph can be plotted. The graph shows that there is a mass lession in the right lower lobe, and this can seen in the X-ray image.

Step 3: Develop AI Algorithms
This is implemented with 'Step3.py'. An AI model for diagnose cancel was fitted. The breakdown of the code is also in test.ipynb. The code does the following:
(1) Organize the original X-rag images, copy the images with dancer and without cancer into two different directories, respectively. 
(2) Split both the images with and without cancer into training, validation and test dataset
(3) Fit the model
(4) Test the model
(5) 


Step 3: Document the Process (try to use Jupyter notebook)
The progress is documented in test.ipynb

Step 4: Showcase Feasibility
=======
>>>>>>> origin/main
