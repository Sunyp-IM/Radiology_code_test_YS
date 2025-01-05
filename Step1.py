##### Step 1. Organize Radiologist Reports into a Graph Database

import os
import pandas as pd
import re


#### Define the current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
cwd = os.getcwd()

#### Load the radiologist report file
report_path = path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + \
    '/ProgrammingTest_Data/Radiology Reports.pkl'
radiology_reports = pd.read_pickle(report_path)

#### Add a columns for abberiating the PatientID
patient_ids = [f'Patient_{i:04d}' for i in range(1, 2001)]
radiology_reports.insert(0, 'PatientID_A', patient_ids)

#### Define the terms to search for disease, abnornality, lesion, and locatio in the radiologist reports. To do that, Firstly the medical
#### terms are extracted from the Full_REPORT column using spacy "en_ner_bc5cdr_md" model. Then, the terms are manually inspected to decide
#### whether they are disease, abnornality, lesion, and location. To determine this, the following references are used:
#### (1) MESH terms for lung diseases (https://www.ncbi.nlm.nih.gov/mesh?Db=mesh&Cmd=DetailsSearch&Term=%22Lung+Diseases%22%5BMeSH+Terms%5D)
#### (2) Radiology Assistant webpage (https://radiologyassistant.nl/chest/chest-x-ray/lung-disease)

diseases = ['chronic obstructive pulmonary disease', 'COPD', 'pulmonary fibrosis', 'cystic fibrosis', 'pneumonia', 'bronchopneumonia', 
            'peripheral pneumonia', 'cancer', 'lung cancer', 'mesothelioma', 'carcinomatosis', 'primary lung neoplasm', 'AML', 
            'pulmonary artery hypertension', 'pulmonary hypertension', 'pulmonary arterial hypertension', 'pulmonary venous hypertension',
            'chronic cardiopulmonary disease', 'ARDS', 'Congestive Heart Failure','CHF', 'diastolic heart fail', 'lung abscess', 
            'radiation pneumonitis', 'bronchitis', 'degenerative disease', 'obstructive pneumonia', 'postobstructive pneumonia',
            'interstitial pulmonary fibrosis', 'marfan syndrome', 'bochdalek hernia', 'hiatal hernia']

abnormalities = ['opacity', 'opacities', 'opacification', 'nodular opacities', 'left basilar opacities', 'right bibasilar opacities', 
            'right basilar opacities', 'bibasilar patchy opacities', 'interstitial airspace opacities', 'bibasilar opacity', 
            'hazy basilar opacity', 'lower lobe opacities', 'bibasilar hazy', 'airspace opacification', 'left basilar opacity', 
            'streaky bibasilar opacities', 'interstitial opacities', 'interstitial opacifications', 'reticular nodular opacities', 
            'hazy bibasilar opacities', 'heterogeneous opacities', 'left hemidiaphragm', 'basilar patchy opacifications',
                   
            'enlargement of the right pulmonary artery','enlarged main pulmonary artery', 'enlarged cardiac silhouette', 
            'right thyroid enlargement', 'enlargement of cardiomediastinal silhouette', 'enlargement of the cardiac', 
            'fullness of the superior mediastinum', 'right hilar enlargement', 'enlarged central pulmonary arteries', 'patulous esophagus', 
            'thyroid enlargement', 'aneurysmal dilatation', 'enlargement of the aorta', 'Cardiac enlargement', 'enlarged thoracic aorta', 
            'enlargement of the RIGHT', 'enlargement of the cardiac mediastinal silhouette', 'enlargement the cardiac silhouette', 
            'enlargement of the cardiomediastinal', 'enlargement of the cardiac silhouette status post', 'enlargement of the cardia', 
            'left hilar adenopathy', 'perihilar fullness',
                   
                   
           'atelectasis','bibasilar segmental atelectasis', 'bibasilar relaxation atelectasis', 'lower lobe subsegmental atelectasis', 
           'volume loss', 'left bibasilar atelectasis', 'middle lobe atelectasis', 'basilar subsegmental atelectasis', 'lingular atelectasis',
           'subsegmental atelectasis', 'bibasal atelectasis', 'postoperative atelectasis', 'discoid atelectasis', 'retrocardiac atelectasis', 
           'left retrocardiac atelectasis', 'right basilar atelectasis', 'lower lobe atelectasis', 'bibasilar subsegmental atelectasis', 
           'basilar discoid atelectatic', 'lucency' 'left basilar nodular atelectasis', 'low lung volumes', 'decreased crescentic lucency', 
           'nonvisualization of the left', 'relaxation atelectasis', 'acute airspace disease', 'lower lobe airspace disease',

            'interstitial disease', 'interstitial lung disease', 'interstitial abnormality', 'pulmonary interstitial edema','alveolar edema', 
            'pulmonary edema', 'airspace pulmonary edema', 'basilar airspace disease', 'upper lobe emphysema','interstitial edema', 
            'intra-alveolar pulmonary edema', 'fluid overload','asymmetric edema', 'asymmetric alveolar edema','asymmetric pulmonary edema', 
            'interstitial pulmonary edema', 'interstitial and alveolar pulmonary edema',
                   
            'emphysema', 'emphysematous change', 'chest wall emphysema', 'Subcutaneous venous emphysema', 'hyperinflated', 'hyperlucent', 
            'hyperinflated lungs', 'lungs right greater', 'pneumothorax', 'hydropneumothorax', 'upper lobe hyperinflation',
                   
            'hemorrhage', 'pulmonary hemorrhage', 'hemothorax', 'alveolar damage',
                   
            'pleural effusion', 'left and small right pleural effusion', 'small pleural effusions', 'pleural effusions right', 
            'pleural effusions blunt', 'bibasilar pleural effusions', 'right effusion', 'left-sided small pleural effusion', 
            'subpulmonic pleural effusion',
                   
            'fibrosis', 'calculus', 'degenerative changes', 'thoracic degenerative', 'osteopenia', 'hiatus hernia', 'thoracic adenopathy', 
            'nipple shadows', 'depressed', 'infection', 'aspiration', 'ectasia', 'airway obstruction', 'obstruction', 'pseudoobstruction', 
            'diverticulitis', 'pleural sinus tract', 'cardiomegaly', 'interlobular septal thickening', 'substernal goiter', 'loop of bowel', 
            'lymphadenopathy',
                   
            'calcification', 'calcifications', 'Atherosclerotic', 'atherosclerotic calcifications', 'artery calcific', 
            'atherosclerosis calcifications', 'coarse atherosclerotic calcifications', 'aortic atherosclerotic calcifications', 
            'vascular calcifications', 'calcifications of the thoracic aorta', 'atherosclerotic vascular calcifications', 
            'atherosclerotic ossifications', 'therosclerotic aortic calcifications', 'coronary artery calcifications', 'atherosclerosis'
            'atherosclerotic aortic calcifications', 'pleural calcifications', 'calcification of the aorta', 'pericardial calcifications', 
                   
            'aortic tortuosity', 'tortuous thoracic aorta', 'innominate artery tortuosity', 'obscuration of hilar vasculature', 
            'azygos fissure', 'pulmonary vascular fullness', 'thoracic aortic dissection', 'loss of pulmonary vascular', 
            'crowding of pulmonary vascular', 'aneurysmal', 'pulmonary artery shadows', 'thoracic aortic tortuosity','vascular congestion', 
            'perihilar congestion', 'pulmonary vascular congestion', 'pulmonary venous engorgement',
                   
            'deformity', 'heterogeneous pleural thickening', 'posttraumatic deformity', 'left-sided rib fractures', 'levoscoliosis',
            'acute osseous abnormality', 'fracture deformities', 'rib fracture deformities', 'rib fracture', 'rib fractures',
            'left clavicular fracture deformity', 'postoperative contusion', 'thoracic compression', 'kyphoscoliosis', 'kyphosis', 
            'thoracic kyphosis', 'pectus carinatum deformity', 'right anterior shoulder dislocation', 
            'thoracic vertebral compression fracture', 'compression deformities of thoracic spine', 'Compression deformities', 
            'dextroscoliosis', 'dextroconvex thoracic curvature', 'compression', 'glenohumeral subluxation', 
            'thoracic compression deformities', 'Thoracolumbar spine scoliosis', 'pectus excavatum deformity', 'fractures', 
            'left rib fracture deformities', 'sternal fracture', 'subluxation', 'displaced rib fractures', 'traumatic deformity',
            'thoracic dextroscoliosis', 'humeral head sclerosis', 'demineralization of the skeleton', 'thoracolumbar scoliosis',
            'DJD', 'rotator cuff injury', 'trauma', 'thoracic trauma', 'osteopenia'
                   ]

lesions = ['cyst', 'cystic change', 'nodule', 'nodules', 'nodularity', 'multiple pulmonary nodules', 'granuloma', 'calcified granuloma', 
            'ulcer', 'abscess', 'lesion', 'cavitary lesion', 'tumor', 'mass']

locations = ['right lung', 'left lung', 'upper', 'middle', 'lower', 'left upper', 'right upper', 'apical', 'basal', 'lingula', 'left basilar', 
            'right basilar', 'bibasilar',
            'anterior segment', 'posterior segment', 'lateral segment', 'medial segment', 'superior segment', 'inferior segment', 'left apex', 'right apex'
            'midlung', 'pleural', 'pleural space', 'pleural cavity', 'parietal pleura', 'visceral pleura',
            'mediastinum', 'mediastinumal', 'anterior mediastinum', 'middle mediastinum', 'posterior mediastinum', 'anterior mediastinumal', 'middle mediastinumal', 
            'posterior mediastinumal', 'suprahilar/mediastinal',
            'right hemidiaphragm', 'left hemidiaphragm',
            'cardiac silhouette', 'aorta', 'aortic arch', 'pulmonary artery', 'left pulmonary artery', 'right pulmonary artery', 'superior vena cava', 
            'inferior vena cava', 'hilar region', 'left hilar', 'right hilar'
            'thoracic spine', 'right hilum', 'left hilum', 'costophrenic angle', 'cardiophrenic angle', 'retrocardiac area', 'perihilar region', 
            'rib', 'sternum', 'clavicles', 'thoracic Spine',
            'subpulmonic space', 
            'adjacent to the descending branch of the left pulmonary artery', 'right glenohumeral joint']

negative_phrases = ["no evidence of", "no evidence for", "no", "without", "negative for", "free of", "not", "absence of"]


### Extracting information and construct the graphic network for the the report

import networkx as nx
import json
from networkx.readwrite import json_graph

# Function to split the text into sentences
def split_into_sentences(text):
    return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

# Function to find target_terms (diseases, abnormalities, lesions, or locations) using regular expressions in a individual sentence in a report.
def find_target_term(target_term, sentence_text):
    # \b is a word boundary, ensuring that we match whole words
    pattern = rf'\b{re.escape(target_term)}\b'
    return re.search(pattern, sentence_text)

# Function to exclude target terms in negative context
def is_negative_context(sentence_text, target_term):
    for negative_phrase in negative_phrases:
        if re.search(rf'\b{re.escape(negative_phrase)}\b', sentence_text, re.IGNORECASE):
            return True
    return False

# Function to remove general term (abnormality,location, etc.) if more specific term exist
def find_specific_terms(term_list):
    # Sort the terms by length in descending order
    term_list.sort(key=len, reverse=True)    
    filtered_terms = []
    for i, term in enumerate(term_list):
        is_substring = False
        for j in range(len(term_list)):
            if i != j and term in term_list[j]:
                is_substring = True
                break
        if not is_substring:
            filtered_terms.append(term)    
    return filtered_terms
    
# Function to find size
def find_size(sentence_text):
    # Define a regular expression pattern to match sizes
    size_pattern = re.compile(r'\b(\d+(\.\d+)?)\s*(cm|mm|millimeter|millimeters|centimeter|centimeters)\b')
    # Search for sizes in the report text
    sizes = size_pattern.findall(sentence_text)   
    # Extract the sizes and units
    extracted_sizes = [f"{match[0]} {match[2]}" for match in sizes]
     # Define additional size descriptors
    additional_sizes = ['small', 'medium', 'large']
    # Search for additional size descriptors in the sentence text
    for size in additional_sizes:
        if re.search(rf'\b{size}\b', sentence_text, re.IGNORECASE):
            extracted_sizes.append(size)
    return extracted_sizes
    
# Function to join list elements into a single string
def join_list_elements(lst):
    return ', '.join(lst) if lst else None

# Function to add a node if the value is not None or empty
def add_node_if_not_empty(graph, node_value, node_type):
    if node_value:  # Check if the value is not None or empty
        graph.add_node(node_value, type=node_type)

# Function to add a edge if the both valuea are not None or empty
def add_edge_if_non_empty(graph, node1_value, node2_value, nodes_relationship):
    if node1_value and node2_value:  # Check if the two node values are not None or empty
        graph.add_edge(node1_value, node2_value, relationship=nodes_relationship)



### Extract the Patient ID and Full_REPORT from a row in radiology_reports; split the Full_REPORT in to individual stentents; extract diseases, abnormalities,
### locations, and sizes from the stetence; make a DataFrame for the extracted information. 
## define an empty DataFrame to store all the extracted information
columns = ['PatientID_A', 'Disease', 'Abnormality', 'Abnormality_size', 'Lesion', 'Lesion_location', 'Lesion_size']
df0 = pd.DataFrame(columns=columns)

## Define a list to store network graphs of indivial patients
graph_path_list = []

## Extract information from each individual patient reports and construct network graphs for them 
for index, row in radiology_reports.iterrows():
    patient_id = row['PatientID_A']
    report_text = row['FULL_REPORT'] 
    # Normalize the report text to lower case for case-insensitive matching
    normalized_report_text = report_text.lower()
    
    # Split the text into sentences
    sentences = split_into_sentences(normalized_report_text)
    
    # define an empty DataFrame to store the extracted information of an individul row (patient) of constructing the graphic network
    df = pd.DataFrame(columns=columns)
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        # Find disease in the sentence
        found_diseases = []
        for disease in diseases:
            disease_lower = disease.lower()
            if find_target_term(disease_lower, sentence_lower):
                if not is_negative_context(sentence_lower, disease_lower):
                   found_diseases.append(disease)
    
        # Find abnormalities in sentence
        found_abnormalities = []
        for abnormality in abnormalities:
            abnormality_lower = abnormality.lower()
            if find_target_term(abnormality_lower, sentence_lower):
                if not is_negative_context(sentence_lower, abnormality_lower):
                    found_abnormalities.append(abnormality)
    
        # Remove the more general abnormalities if there is more specific abnormalilites 
        filtered_abnormalities = find_specific_terms(found_abnormalities)
        
        # Find abnormality size
        abnormality_size = []
        if filtered_abnormalities:
            abnormality_size = find_size(sentence_lower)
    
        # Find lesions in sentence
        found_lesions= []
        for lesion in lesions:
            lesion_lower = lesion.lower()
            if find_target_term(lesion_lower, sentence_lower):          
                if not is_negative_context(sentence_lower, lesion_lower):
                    found_lesions.append(lesion)
    
        # Find lesion locations 
        found_lesion_locations = []
        if found_lesions:        
            for location in locations:
                location_lower = location.lower()
                if find_target_term(location_lower, sentence_lower):
                    found_lesion_locations.append(location)
                    
        # Remove the more general locations if there is more specific locations 
        filtered_lesion_locations = find_specific_terms(found_lesion_locations)
    
        # Find lesion size
        lesion_size = []
        if found_lesions:
            lesion_size = find_size(sentence_lower)
        
        
        # Join list elements into a single string for each extract information
        if found_diseases or filtered_abnormalities or found_lesions:
            joined_diseases = join_list_elements(found_diseases)
            joined_abnormalities = join_list_elements(filtered_abnormalities)
            joined_abnormality_size = join_list_elements(abnormality_size)
            joined_lesions = join_list_elements(found_lesions)
            joined_lesion_locations = join_list_elements(filtered_lesion_locations)
            joined_lesion_size = join_list_elements(lesion_size)
        
            # Create a new row as a DataFrame
            new_row = pd.DataFrame({
                'PatientID_A': [patient_id],
                'Disease': [joined_diseases],
                'Abnormality': [joined_abnormalities],
                'Abnormality_size': [joined_abnormality_size],
                'Lesion': [joined_lesions],
                'Lesion_location': [joined_lesion_locations],
                'Lesion_size': [joined_lesion_size]
            })
        
            # Concatenate the new row with the original DataFrame
            df = pd.concat([df, new_row], ignore_index=True)

    ## Append df to df0
    df0 = pd.concat([df0, df])

    ## Dave df to file
    df0.to_csv('Extracted_info.csv')

    
    ## Construct the graphic network
    # Create an empty graph
    G = nx.Graph()    
    for i, row_df in df.iterrows(): 
        # Add notes
        add_node_if_not_empty(G, df.loc[i, 'PatientID_A'], 'patient')
        add_node_if_not_empty(G, df.loc[i, 'Disease'], 'disease')
        add_node_if_not_empty(G, df.loc[i, 'Abnormality'], 'abnormality')
        add_node_if_not_empty(G, df.loc[i, 'Abnormality_size'], 'Abnormality_size')
        add_node_if_not_empty(G, df.loc[i, 'Lesion'], 'lesion')
        add_node_if_not_empty(G, df.loc[i, 'Lesion_location'], 'lesion_location')
        add_node_if_not_empty(G, df.loc[i, 'Lesion_size'], 'lesion_size')
        
        # Add relationships (edges)
        if df.loc[i, 'Disease']:
            add_edge_if_non_empty(G, df.loc[i, 'PatientID_A'], df.loc[i, 'Disease'], 'HAS_DIAGNOSIS')
        elif not df.loc[i, 'Disease'] and df.loc[i, 'Abnormality']:
            add_edge_if_non_empty(G, df.loc[i, 'PatientID_A'], df.loc[i, 'Abnormality'], 'HAS_ABNORMALITY')
        elif not df.loc[i, 'Disease'] and not df.loc[i, 'Abnormality'] and df.loc[i, 'Lesion']:
            add_edge_if_non_empty(G, df.loc[i, 'PatientID_A'], df.loc[i, 'Lesion'], 'HAS_LESION')
        elif not df.loc[i, 'Disease'] and df.loc[i, 'Abnormality'] and df.loc[i, 'Lesion']:
            add_edge_if_non_empty(G, df.loc[i, 'PatientID_A'], df.loc[i, 'Abnormality'], 'HAS_ABNORMALITY')
            add_edge_if_non_empty(G, df.loc[i, 'PatientID_A'], df.loc[i, 'Lesion'], 'HAS_LESION')
        add_edge_if_non_empty(G, df.loc[i, 'Disease'], df.loc[i, 'Abnormality'], 'HAS_ABNORMALITY')
        add_edge_if_non_empty(G, df.loc[i, 'Abnormality'], df.loc[i, 'Abnormality_size'], 'HAS_SIZE')
        add_edge_if_non_empty(G, df.loc[i, 'Disease'], df.loc[i, 'Lesion'], 'HAS_Lesion')
        add_edge_if_non_empty(G, df.loc[i, 'Lesion'], df.loc[i, 'Lesion_location'], 'LOCATED_IN')
        add_edge_if_non_empty(G, df.loc[i, 'Lesion'], df.loc[i, 'Lesion_size'], 'HAS_SIZE')
        
    ## Add the graph path the the graph_list
    graph_path_list.append(G) 
    
    ## Save the graphic network to json file
    # Convert the graph to a JSON-compatible dictionary
    data = json_graph.node_link_data(G)
    file_path = 'graphs/' + patient_id + '_graph.json'
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

### Add a new column for graph_list to radiology_reports
radiology_reports['Graph_path'] = graph_path_list


#### Visualize the network graph and x-ray image
from textwrap import wrap
from textwrap import fill
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

## Specify the patient
patient_id_a = 'Patient_0100'
patient_id = radiology_reports.loc[radiology_reports['PatientID_A'] == patient_id_a, 'PatientID'].values[0]
report = radiology_reports.loc[radiology_reports['PatientID_A'] == patient_id_a, 'FULL_REPORT'].values[0]
print(f'Report: \n{report}\n\n')

## Load a network graph
# Load the graph from the JSON file
graph_file_path = 'graphs/Patient_0100_graph.json'
with open(graph_file_path, 'r') as f:
    data = json.load(f)

# Convert the JSON data to a networkx graph
G = nx.node_link_graph(data)

# define the network graph
subgraph_nodes = list(G.nodes)[:20]  # Take first 20 nodes for visualization
subgraph = G.subgraph(subgraph_nodes)
edge_labels = nx.get_edge_attributes(subgraph, 'relationship')

# Define the position layout for the nodes
pos = nx.spring_layout(subgraph)


## Load the X-ray image
path = os.path.abspath(os.path.join(cwd, os.pardir))
xray_image_filename = radiology_reports.loc[radiology_reports['PatientID_A'] == patient_id_a, 'PATH'].values[0]
xray_image_path = path + '/ProgrammingTest_Data/png/' + xray_image_filename
xray_image = mpimg.imread(xray_image_path)


## Create a figure with subplots
fig, axs = plt.subplots(1, 2, figsize=(20, 10))  # 1 row, 2 columns

## Create the dynamic title for the plot
figure_title = f"The network graph and the X-ray image for {patient_id_a} ({patient_id})"
wrapped_figure_title = "\n".join(wrap(figure_title, 100))  # Wrap the title text to 100 characters per line

## Set the super title of the figure
fig.suptitle(wrapped_figure_title, fontsize=14)

## Plot the network graph in the first subplot
# Function to wrap text
def wrap_labels(labels, width=20):
    return {node: fill(label, width) for node, label in labels.items()}

# Wrap the node labels
wrapped_labels = wrap_labels({node: node for node in subgraph.nodes})

#Plot the network graph
axs[0].set_title("Network Graph")
nx.draw(subgraph, pos, with_labels=True, labels=wrapped_labels, node_size=3000, node_color='lightblue', font_size=10, ax=axs[0])
nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, ax=axs[0])


## Plot the X-ray image in the second subplot
axs[1].imshow(xray_image)
axs[1].set_title("X-ray Image")
axs[1].axis('off')  # Hide the axes for the image

# Adjust layout to make room for the suptitle and ensure titles are visible
plt.tight_layout()
fig.subplots_adjust(top=0.85)

print(lesions)