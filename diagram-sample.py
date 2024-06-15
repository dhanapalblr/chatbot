from graphviz import Digraph

# Create a new Digraph
dot = Digraph(comment='AI/ML Workflow')

# Data Ingestion Group
with dot.subgraph(name='cluster_data_ingestion') as c:
    c.attr(color='blue')
    c.node_attr.update(style='filled', color='lightblue')
    c.attr(label='Data Ingestion', icon='database')
    c.node('CSV File', 'data.csv', icon='file')
    c.node('Load Data', icon='upload')

# Data Preprocessing Group
with dot.subgraph(name='cluster_data_preprocessing') as c:
    c.attr(color='green')
    c.node_attr.update(style='filled', color='lightgreen')
    c.attr(label='Data Preprocessing', icon='settings')
    c.node('Anonymize Data', icon='shield')
    c.node('Handle Missing Values', icon='eraser')
    c.node('Preprocess Data', icon='filter')
    with c.subgraph(name='cluster_preprocess_data') as sub_c:
        sub_c.node('Encode Categorical Variables', icon='code')
        sub_c.node('Scale Numerical Features', icon='bar-chart')
        sub_c.node('Generate Polynomial Features', icon='functions')
        sub_c.node('Perform Feature Selection', icon='select')

# Model Training Group (Vertical Layout)
with dot.subgraph(name='cluster_model_training') as c:
    c.attr(color='orange')
    c.node_attr.update(style='filled', color='lightcoral')
    c.attr(label='Model Training', icon='cpu')
    c.node('Split Data into Training and Testing Sets', icon='split')
    c.node('Reshape Data for LSTM Input', icon='layers')
    c.node('Augment Data for Imbalance', icon='balance-scale')
    c.node('Train Model with Grid Search', icon='search')
    c.node('Cross-Validate Model', icon='crosshairs')
    c.node('Evaluate Model', icon='check-circle')
    c.node('Hyperparameter Tuning', icon='sliders')
    c.node('Save Best Model', icon='save')

# GitHub Integration Group
with dot.subgraph(name='cluster_github_integration') as c:
    c.attr(color='purple')
    c.node_attr.update(style='filled', color='lightpink')
    c.attr(label='GitHub Integration', icon='github')
    c.node('Create GitHub Issue with Processed Data', icon='issue-opened')

# Deployment and Automation Group
with dot.subgraph(name='cluster_deployment_automation') as c:
    c.attr(color='red')
    c.node_attr.update(style='filled', color='lightcoral')
    c.attr(label='Deployment and Automation', icon='play')
    c.node('Run main.py Script', icon='terminal')

# Connections
dot.edge('CSV File', 'Load Data')
dot.edge('Load Data', 'Anonymize Data')
dot.edge('Anonymize Data', 'Handle Missing Values')
dot.edge('Handle Missing Values', 'Preprocess Data')
dot.edge('Preprocess Data', 'Encode Categorical Variables')
dot.edge('Preprocess Data', 'Scale Numerical Features')
dot.edge('Encode Categorical Variables', 'Generate Polynomial Features')
dot.edge('Scale Numerical Features', 'Generate Polynomial Features')
dot.edge('Generate Polynomial Features', 'Perform Feature Selection')
dot.edge('Perform Feature Selection', 'Split Data into Training and Testing Sets')
dot.edge('Split Data into Training and Testing Sets', 'Reshape Data for LSTM Input')
dot.edge('Reshape Data for LSTM Input', 'Augment Data for Imbalance')
dot.edge('Augment Data for Imbalance', 'Train Model with Grid Search')
dot.edge('Train Model with Grid Search', 'Cross-Validate Model')
dot.edge('Cross-Validate Model', 'Evaluate Model')
dot.edge('Evaluate Model', 'Hyperparameter Tuning')
dot.edge('Hyperparameter Tuning', 'Save Best Model')
dot.edge('Save Best Model', 'Create GitHub Issue with Processed Data')
dot.edge('Create GitHub Issue with Processed Data', 'Run main.py Script')

# Render and save the graph
dot.render('ai_ml_workflow', format='png', view=True)
