import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from sklearn.cluster import DBSCAN
from PIL import Image
import numpy as np
from scipy.stats import gaussian_kde


#set title
st.title('Live to Win!')
st.text('Analyzing Common Causes of Death within the World of Warcraft Hardcore Challenge')

#displaying the image on streamlit app

placeholder=st.image("https://boosting.pro/wp-content/uploads/2019/08/WoW-Classic-Vanilla-Map.jpg")

##upload file
st.sidebar.title('Navigation')
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv('clean_data.csv')###change path

##layout
options = st.sidebar.radio('Pages',
                           options = ['Home',
                                      'Data Stats',
                                      'Data Header',
                                      'Summary Plot',
                                      'Top 10',
                                      'Deaths by Class & Level',
                                      'Players Alive by Level',
                                      'NPC Level versus Player Level',
                                      'Map plot',
                                      'Machine learning plot'])

class_colors = {
    'Druid': '#FF7D0A',     # Orange
    'Hunter': '#ABD473',    # Green
    'Mage': '#69CCF0',      # Light blue
    'Paladin': '#F58CBA',   # Pink
    'Priest': '#FFFFFF',    # White
    'Rogue': '#FFF569',     # Light yellow
    'Shaman': '#0070DE',    # Blue
    'Warlock': '#9482C9',   # Purple
    'Warrior': '#C79C6E',   # Tan
    'Total':  '#000000'     # Black
    }

#sidebar functions

def stats(df):
    placeholder.empty()
    st.header('Data Stats')
    st.write(df.describe())
    
def header(df):
    placeholder.empty()
    st.header('Data Header')
    st.write(df.head())

def summary_plot(df):
    #@title Show Race Summary Chart
    placeholder.empty()
    st.header('')
    # Create a new DataFrame with race_name counts
    summary_df = df['race_name'].value_counts().reset_index()
    summary_df.columns = ['race_name', 'count']

    # Calculate the percentage of total records
    total_records = len(df)
    summary_df['percentage'] = (summary_df['count'] / total_records) * 100

    # define color
    race_colors = {
    'Human': '#006699',     # Blue
    'Orc': '#FF4400',       # Orange
    'Dwarf': '#FFCC33',     # Yellow
    'Night Elf': '#9933FF',  # Purple
    'Undead': '#660066',    # Dark Purple
    'Tauren': '#C79C6E',    # Tan
    'Gnome': '#66FFCC',     # Cyan
    'Troll': '#336600'      # Green
    }
    
    class_colors = {
    'Druid': '#FF7D0A',     # Orange
    'Hunter': '#ABD473',    # Green
    'Mage': '#69CCF0',      # Light blue
    'Paladin': '#F58CBA',   # Pink
    'Priest': '#FFFFFF',    # White
    'Rogue': '#FFF569',     # Light yellow
    'Shaman': '#0070DE',    # Blue
    'Warlock': '#9482C9',   # Purple
    'Warrior': '#C79C6E',   # Tan
    'Total':  '#000000'     # Black
    }
    
    # Create the bar chart using Plotly
    race_fig = go.Figure(data=[
        go.Bar(
            x=summary_df['race_name'],
            y=(summary_df['count'] / total_records),  # Set y-axis as percentage
            marker_color=[race_colors.get(c, '#000000') for c in summary_df['race_name']],
            text=[f"{count} ({percentage:.2f}%)" for count, percentage in zip(summary_df['count'], summary_df['percentage'])],  # Add count and percentage as text
            textposition='auto',  # Set text position to be automatically determined
            hovertemplate='%{text}',  # Customize the hover template
            texttemplate='%{text}'  # Display the text label on the bars
        )
    ])

    race_fig.update_layout(
        title='Total Deaths Recorded by Race',
        xaxis_title='Race Name',
        yaxis_title='Percentage of Total Records',
        yaxis_tickformat='.0f%',  # Format y-axis tick labels as percentage with no decimal places and "%" symbol
        plot_bgcolor='rgba(0, 0, 0, 0)'  # Set plot background color to transparent
    )
    st.plotly_chart(race_fig)
    
    
    #@title Show Class Summary Chart

    # Create a new DataFrame with class_name counts
    summary_df = df['class_name'].value_counts().reset_index()
    summary_df.columns = ['class_name', 'count']

    # Calculate the percentage of total records
    total_records = len(df)
    summary_df['percentage'] = (summary_df['count'] / total_records) * 100

    # Create the bar chart using Plotly
    class_fig = go.Figure(data=[
        go.Bar(
            x=summary_df['class_name'],
            y=(summary_df['count'] / total_records),  # Set y-axis as percentage
            marker_color=[class_colors.get(c, '#000000') for c in summary_df['class_name']],
            text=[f"{count} ({percentage:.2f}%)" for count, percentage in zip(summary_df['count'], summary_df['percentage'])],  # Add count and percentage as text
            textposition='auto',  # Set text position to be automatically determined
            hovertemplate='%{text}',  # Customize the hover template
            texttemplate='%{text}'  # Display the text label on the bars
        )
    ])

    class_fig.update_layout(
        title='Total Deaths Recorded by Class',
        xaxis_title='Class Name',
        yaxis_title='Percentage of Total Records',
        yaxis_tickformat='.0f%',  # Format y-axis tick labels as percentage with no decimal places and "%" symbol
        plot_bgcolor='rgba(0, 0, 0, 0)'  # Set plot background color to transparent
    )

    st.plotly_chart(class_fig)
    
    #@title Show Level Summary Chart

    # Create a new DataFrame with level counts
    summary_df = df['level'].value_counts().reset_index()
    summary_df.columns = ['level', 'count']

    # Calculate the percentage of total records
    total_records = len(df)
    summary_df['percentage'] = (summary_df['count'] / total_records) * 100

    level_fig = go.Figure(data=[
        go.Bar(
            x=summary_df['level'],
            y=(summary_df['count'] / total_records),  # Set y-axis as percentage
            text=[f"Level {level}: {count} ({percentage:.2f}%)" for level, count, percentage in zip(summary_df['level'], summary_df['count'], summary_df['percentage'])],  # Add level, count, and percentage as text
            textposition='auto',  # Set text position to be automatically determined
            hovertemplate='%{text}',  # Customize the hover template
            texttemplate='%{text}',  # Display the text label on the bars
            marker_color='#D3D3D3'  # Set the bar color as light gray
        )
    ])

    level_fig.update_layout(
        title='Total Deaths Recorded by Level',
        xaxis=dict(
            title='Level',
            tickmode='linear',  # Set tick mode to linear
            dtick=1,  # Set tick interval to 1
            tickfont=dict(size=10)  # Set tick font size
        ),
        xaxis_title='Level',
        yaxis_title='Percentage of Total Records',
        yaxis_tickformat='.0f%',  # Format y-axis tick labels as percentage with no decimal places and "%" symbol
        plot_bgcolor='rgba(0, 0, 0, 0)'  # Set plot background color to transparent
    )

    st.plotly_chart(level_fig)    
    
    #@title Show Zone Summary Chart

    # Create a new DataFrame with map_name counts
    summary_df = df['map_name'].value_counts().reset_index()
    summary_df.columns = ['map_name', 'count']

    # Calculate the percentage of total records
    total_records = len(df)
    summary_df['percentage'] = (summary_df['count'] / total_records) * 100

    zone_fig = go.Figure(data=[
        go.Bar(
            x=summary_df['map_name'],
            y=(summary_df['count'] / total_records),  # Set y-axis as percentage
            text=[f"Map: {name}\nCount: {count} ({percentage:.2f}%)" for name, count, percentage in zip(summary_df['map_name'], summary_df['count'], summary_df['percentage'])],  # Add map name, count, and percentage as text
            textposition='auto',  # Set text position to be automatically determined
            hovertemplate='%{text}',  # Customize the hover template
            texttemplate='%{text}',  # Display the text label on the bars
            marker_color='#D3D3D3'  # Set the bar color as light gray
        )
    ])

    zone_fig.update_layout(
        title='Total Deaths Recorded by Map',
        xaxis_title='Map Name',
        yaxis_title='Percentage of Total Records',
        yaxis_tickformat='.0f%',  # Format y-axis tick labels as percentage with no decimal places and "%" symbol
        plot_bgcolor='rgba(0, 0, 0, 0)'  # Set plot background color to transparent
    )

    st.plotly_chart(zone_fig)    
    
def top_ten(df):
    
    placeholder.empty()
    #@title Top 10 Death Sources by Class
    
    class_colors = {
    'Druid': '#FF7D0A',     # Orange
    'Hunter': '#ABD473',    # Green
    'Mage': '#69CCF0',      # Light blue
    'Paladin': '#F58CBA',   # Pink
    'Priest': '#FFFFFF',    # White
    'Rogue': '#FFF569',     # Light yellow
    'Shaman': '#0070DE',    # Blue
    'Warlock': '#9482C9',   # Purple
    'Warrior': '#C79C6E',   # Tan
    'Total':  '#000000'     # Black
    }
    
    # Get the top 10 npc_names per class
    top_npc_names = df.groupby(['class_name', 'npc_name']).size().groupby(level=0).nlargest(10).reset_index(level=0, drop=True).reset_index(name='count')

    # Create a new DataFrame with npc_name counts by class
    summary_df = top_npc_names.pivot(index='npc_name', columns='class_name', values='count').fillna(0).reset_index()

    # Sort the summary_df by the count for each class separately
    sorted_dfs = {}
    for column in summary_df.columns[1:]:
        sorted_dfs[column] = summary_df[['npc_name', column]].sort_values(column, ascending=False)

    # Create subplots for each class
    fig = make_subplots(rows=len(summary_df.columns[1:]), cols=1, subplot_titles=summary_df.columns[1:], shared_xaxes=True, vertical_spacing=0.02)

    for i, column in enumerate(summary_df.columns[1:], start=1):
        sorted_df = sorted_dfs[column]
        non_zero_df = sorted_df[sorted_df[column] > 0]  # Filter out NPCs with zero records

        fig.add_trace(
            go.Bar(
                x=non_zero_df[column],
                y=non_zero_df['npc_name'],
                orientation='h',
                name=column,
                text=[f'{npc_name}<br>Total Records: {count}' for npc_name, count in zip(non_zero_df['npc_name'], non_zero_df[column])],
                hovertemplate='%{text}',
                marker_color=class_colors.get(column, '#000000')  # Color code the bars based on class_colors
            ),
            row=i,
            col=1
        )

        fig.update_yaxes(title_text='NPC Name', row=i, col=1)
        fig.update_xaxes(title_text='Total Records', row=i, col=1)

    fig.update_layout(
        title='Top 10 Death Sources by Class',
        height=400 * len(summary_df.columns[1:]),  # Adjust the height based on the number of classes
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Set plot background color to transparent
        showlegend=False,
    )

    st.plotly_chart(fig)    

def deaths_by_class_Level(df):
    
    placeholder.empty()
    
    count_table = pd.pivot_table(df, index='class_name', columns='level', aggfunc='size', fill_value=0)
    total_row = count_table.sum(axis=0)
    total_row.name = 'Total'
    count_table = count_table._append(total_row)
    #count_table = pd.concat([count_table, total_row])

    #@title Show Deaths by Class & Level Chart:

    # Create a list to hold the traces
    traces = []

    # Iterate over each class
    for class_name in count_table.index:
        # Get the data for the current class
        data = count_table.loc[class_name]

        # Create a trace for the current class
        trace = go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            name=class_name,
            line=dict(color=class_colors[class_name]),
            hovertemplate='Class: ' + class_name + '<br>Level: %{x}<br>Count: %{y}<extra></extra>'
        )

        # Add the trace to the list
        traces.append(trace)

    # Create the layout
    layout = go.Layout(
        title='Total Deaths Recorded by Class by Level',
        xaxis=dict(title='Level'),
        yaxis=dict(title='Count'),
        hovermode='closest',
        legend=dict(title='Class'),
        showlegend=True
    )

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)

    # Show the figure
    st.plotly_chart(fig)  
    
    
    
    #@title Analysis of what percentage of all deaths occur at each level (1-60):

    # Calculate the percentage of the sum of row counts for each cell
    percentage_table = count_table.div(count_table.sum(axis=1), axis=0) * 100

    # Create a copy of the percentage_table with formatted percentages
    formatted_table = percentage_table.copy()
    formatted_table = formatted_table.applymap(lambda x: f'{x:.2f}%')

    # Display the formatted table
    #@title Show Percentage of Deaths by Class & Level Chart:

    # Create a list to hold the traces
    traces = []

    # Iterate over each class
    for class_name in percentage_table.index:
        # Get the data for the current class
        data = percentage_table.loc[class_name]

        # Create a trace for the current class
        trace = go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            name=class_name,
            line=dict(color=class_colors[class_name], width=2 if class_name == 'Total' else 1),
            hovertemplate='Class: ' + class_name + '<br>Level: %{x}<br>Percentage: %{y:.2f}%<extra></extra>'
        )

        # Add the trace to the list
        traces.append(trace)

    # Create the layout
    layout = go.Layout(
        title='Percentage of Deaths by Class by Level',
        xaxis=dict(title='Level'),
        yaxis=dict(title='Percentage'),
        hovermode='closest',
        legend=dict(title='Class'),
        showlegend=True
    )

    # Create the figure
    ratio_fig = go.Figure(data=traces, layout=layout)

    # Show the figure
    st.plotly_chart(ratio_fig) 

def player_alive(df):
    st.header('How many players reach key levels (1-60)')
    #@title Analysis of how many players reach all levels (1-60):
    count_table = pd.pivot_table(df, index='class_name', columns='level', aggfunc='size', fill_value=0)
    total_row = count_table.sum(axis=0)
    total_row.name = 'Total'
    count_table = count_table._append(total_row)
    
    
    percent_table = count_table.copy()
    percent_table = percent_table.div(percent_table.sum(axis=1), axis=0).mul(100)
    percent_table = 100-percent_table.cumsum(axis=1).round(2)

    #@title Analysis of how many players reach key levels (1-60):

    selected_columns = [10, 20, 30, 40, 50, 60]
    selected_fields = percent_table[selected_columns]
    #selected_fields
    st.table(data=selected_fields)
    
    #@title Show Characters alive at each level Chart:

    # Convert the percent_table DataFrame to a long format
    df = percent_table.reset_index().melt(id_vars='class_name', var_name='level', value_name='percentage')

    # Create a list to hold the traces
    traces = []

    # Iterate over each class
    for class_name in df['class_name'].unique():
        # Filter the data for the current class
        filtered_data = df[df['class_name'] == class_name]

        # Create a trace for the current class
        trace = go.Scatter(
            x=filtered_data['level'],
            y=filtered_data['percentage'],
            mode='lines',
            name=class_name,
            line=dict(color=class_colors[class_name], width=2),
            hovertemplate='Class: ' + class_name + '<br>Level: %{x}<br>Percentage: %{y:.2f}%<extra></extra>'
        )

        # Add the trace to the list
        traces.append(trace)

    # Create the layout
    layout = go.Layout(
        title='Percentage of Players Alive by Level',
        xaxis=dict(title='Level'),
        yaxis=dict(title='Percentage'),
        hovermode='closest',
        legend=dict(title='Class'),
        showlegend=True
    )

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)

    # Show the figure
    st.plotly_chart(fig)
    
def npc_player(df):
    
    count_table_char_vs_npc_level = pd.pivot_table(df, values='Character ID', index='class_name', columns='npc_less_player_level', aggfunc='size', fill_value=0)
    count_table_char_vs_npc_level.loc['Total'] = count_table_char_vs_npc_level.sum()   
    
    #@title Show NPC Level versus Player Level Chart

    # Convert the count_table_char_vs_npc_level DataFrame to a long format
    df = count_table_char_vs_npc_level.reset_index().melt(id_vars='class_name', var_name='npc_less_player_level', value_name='count')

    # Create a list to hold the traces
    traces = []

    # Iterate over each class
    for class_name in df['class_name'].unique():
        # Filter the data for the current class
        filtered_data = df[df['class_name'] == class_name]

        # Create a trace for the current class
        trace = go.Scatter(
            x=filtered_data['npc_less_player_level'],
            y=filtered_data['count'],
            mode='lines',
            name=class_name,
            line=dict(color=class_colors[class_name], width=2),
            hovertemplate='Class: ' + class_name + '<br>NPC Less Player Level: %{x}<br>Count: %{y}<extra></extra>'
        )

        # Add the trace to the list
        traces.append(trace)

    # Create the layout
    layout = go.Layout(
        title='Count of Deaths at each NPC Less Player Level',
        xaxis=dict(title='NPC Less Player Level'),
        yaxis=dict(title='Count'),
        hovermode='closest',
        legend=dict(title='Class'),
        showlegend=True
    )

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)

    # Show the figure
    st.plotly_chart(fig)

def map_plot(df):
    #map dictionary
    map_dict = {947: 'Azeroth',
                1411: 'Durotar',
                1412: 'Mulgore',
                1413: 'The Barrens',
                1414: 'Kalimdor',
                1415: 'Eastern Kingdoms',
                1416: 'Alterac Mountains',
                1417: 'Arathi Highlands',
                1418: 'Badlands',
                1419: 'Blasted Lands',
                1420: 'Tirisfal Glades',
                1421: 'Silverpine Forest',
                1422: 'Western Plaguelands',
                1423: 'Eastern Plaguelands',
                1424: 'Hillsbrad Foothills',
                1425: 'The Hinterlands',
                1426: 'Dun Morogh',
                1427: 'Searing Gorge',
                1428: 'Burning Steppes',
                1429: 'Elwynn Forest',
                1430: 'Deadwind Pass',
                1431: 'Duskwood',
                1432: 'Loch Modan',
                1433: 'Redridge Mountains',
                1434: 'Stranglethorn Vale',
                1435: 'Swamp of Sorrows',
                1436: 'Westfall',
                1437: 'Wetlands',
                1438: 'Teldrassil',
                1439: 'Darkshore',
                1440: 'Ashenvale',
                1441: 'Thousand Needles',
                1442: 'Stonetalon Mountains',
                1443: 'Desolace',
                1444: 'Feralas',
                1445: 'Dustwallow Marsh',
                1446: 'Tanaris',
                1447: 'Azshara',
                1448: 'Felwood',
                1449: "Un'Goro Crater",
                1450: 'Moonglade',
                1451: 'Silithus',
                1452: 'Winterspring'}
    
    
    df["map_id"] = df["map_id"].astype(int)
    df["map_name"] = df["map_id"].map(map_dict)
    map_list = df['map_name'].unique().tolist()

    map_name  = st.selectbox('Select Map', options = map_list)
    st.write(map_name)
    # Iterate over unique map names in the DataFrame
    #unique_map_names = df['map_name'].unique()
    
    # Read the image file and encode it to base64
    file_path = f'Maps/{map_name}.jpg'#####change path
   

    with open(file_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    
    # Get the image dimensions
    with Image.open(image_file.name) as img:
        image_width, image_height = img.size

    # Filter the DataFrame based on the map name
    filtered_df = df[df['map_name'] == map_name]
    
        # Perform DBSCAN clustering on the scaled coordinates (x, y)
    X = filtered_df[['x', 'y']]
    eps = 0.02  # Adjust the eps value as desired
    min_samples_percent = 0.0175  # Adjust the percentage as desired
    min_samples = max(int(len(X) * min_samples_percent), 2)  # Ensure a minimum of 2 samples

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X)

# Add the cluster labels as a column in the DataFrame
    filtered_df['cluster_label'] = cluster_labels
    # Calculate the cluster sizes
    cluster_sizes = filtered_df['cluster_label'].value_counts()
    
    # Filter out the smaller clusters (less than 1% of total records)
    min_cluster_size = int(len(filtered_df) * 0.01)
    small_clusters = cluster_sizes[cluster_sizes < min_cluster_size].index
    filtered_df = filtered_df[~filtered_df['cluster_label'].isin(small_clusters)]

    # Calculate the average distance of each point within a cluster to the cluster center
    cluster_centers = filtered_df.groupby('cluster_label')[['x', 'y']].mean()
    cluster_distances = np.linalg.norm(filtered_df[['x', 'y']] - cluster_centers.loc[filtered_df['cluster_label']].values, axis=1)
    filtered_df['avg_distance_to_center'] = cluster_distances

    # Find the cluster label with the longest average distance
    longest_distance_cluster = filtered_df.groupby('cluster_label')['avg_distance_to_center'].mean().idxmax()
    
    def create_main_map_figure(encoded_image, filtered_df, image_width, image_height, cluster_sizes, min_cluster_size, longest_distance_cluster):
        
        # Define some decent primary colors
        primary_colors = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']

        # Create the main map figure
        main_fig = go.Figure()

        # Add the main map image as a layout image
        main_fig.add_layout_image(
            source='data:image/jpeg;base64,' + encoded_image,
            x=0,
            y=1,
            sizex=1,
            sizey=1,
            sizing='stretch',
            opacity=1,
            layer='below'
        )

        # Get unique cluster labels including -1
        unique_cluster_labels = np.unique(np.concatenate((filtered_df['cluster_label'].unique(), [-1])))

        # Iterate over unique cluster labels
        for idx, cluster_label in enumerate(unique_cluster_labels):
            # Filter the DataFrame for the current cluster label
            cluster_df = filtered_df[filtered_df['cluster_label'] == cluster_label]

            # Scale the x and y coordinates based on the image dimensions
            scaled_x = cluster_df['x'] * image_width
            scaled_y = cluster_df['y'] * image_height

            # Determine the opacity and color for the clusters
            if cluster_label == -1:
                opacity = 0.1  # Light and transparent for cluster label -1
                color = 'gray'
            else:
                opacity = 0.1 if cluster_label in cluster_sizes[cluster_sizes < min_cluster_size].index or cluster_label == longest_distance_cluster else 0.5
                color = primary_colors[idx % len(primary_colors)]

            # Create hover text with cluster details for clusters not labeled as -1
            hover_text = (
                f"Cluster Label: {cluster_label}<br>"
                f"Total Records: {len(cluster_df)}<br>"
                f"Average Character Level: {cluster_df['level'].mean():.2f}<br>"
                f"Median Character Level: {cluster_df['level'].median():.2f}<br>"
                f"Mode Class: {cluster_df['class_name'].mode().values[0]}<br>"
                f"Mode Race: {cluster_df['race_name'].mode().values[0]}<br>"
                f"Mode NPC Name: {cluster_df['npc_name'].mode().values[0]}<br>"
                f"Average NPC Average Level: {cluster_df['npc_avg_level'].mean():.2f}<br>"
                f"Median NPC Average Level: {cluster_df['npc_avg_level'].median():.2f}<br>"
                f"Mode NPC Type: {cluster_df['npc_type'].mode().values[0]}<br>"
                f"Mode NPC Elite Status: {cluster_df['npc_elite_status'].mode().values[0]}<br>"
                f"Mode NPC Rare Status: {cluster_df['npc_rare_status'].mode().values[0]}<br>"
                f"Mode NPC Boss Status: {cluster_df['npc_boss_status'].mode().values[0]}<br>"
            ) if cluster_label != -1 else None

            # Add the clustered records as a scatter plot with hover text
            main_fig.add_trace(go.Scatter(
                x=scaled_x,
                y=scaled_y,
                mode='markers',
                marker=dict(
                    size=10,
                    opacity=opacity,
                    color=color,
                ),
                text=hover_text,
                hoverinfo='text',
                showlegend=False
            ))

            # Add the mode NPC name as text annotation at the center of the cluster
            if cluster_label != -1:
                center_x = scaled_x.mean()
                center_y = scaled_y.mean()
                mode_npc_name = cluster_df['npc_name'].mode().values[0]
                character_level = cluster_df['level'].mean()
                main_fig.add_trace(go.Scatter(
                    x=[center_x],
                    y=[center_y],
                    mode='text',
                    text=[f"{mode_npc_name}<br>Level: {character_level:.2f}"],
                    textposition='bottom center',
                    textfont=dict(size=10, color='black'),
                    showlegend=False
                ))

        # Configure the layout with adjusted axes ranges
        main_fig.update_layout(
            width=800,
            height=600,
            xaxis=dict(visible=False, range=[0, image_width]),
            yaxis=dict(visible=False, range=[image_height, 0]),  # Invert the y-axis range
            margin=dict(l=0, r=0, t=0, b=0)  # Set the margins to 0 on all sides
        )
        return main_fig
    
    try:
        main_fig = create_main_map_figure(encoded_image, filtered_df, image_width, image_height, cluster_sizes, min_cluster_size, longest_distance_cluster)
        st.plotly_chart(main_fig)
    except:
        st.write('No map found')
    # Show the figure
    #st.table(data=filtered_df.head())
    
    def create_pdf_line_chart(filtered_df, class_colors, image_width, image_height):
        try:
            # Create a line chart for the PDF of death level by class
            line_chart_data = filtered_df.groupby(['class_name', 'level']).size().reset_index(name='count')
            line_chart_data['PDF'] = line_chart_data.groupby('class_name')['count'].apply(lambda x: x / x.sum() * 100)

            if 'class_name' not in line_chart_data:
                print("No data available for class. Skipping KDE calculation.")
                return None

            # Smooth the PDF curves using KDE
            kde_line_chart_data = pd.DataFrame()
            for class_name, group in line_chart_data.groupby('class_name'):
                if len(group) > 1:
                    kde = gaussian_kde(group['level'])
                    x_vals = np.linspace(group['level'].min(), group['level'].max(), num=100)
                    kde_vals = kde(x_vals)
                    kde_group = pd.DataFrame({'level': x_vals, 'PDF': kde_vals, 'class_name': class_name})
                    kde_line_chart_data = kde_line_chart_data._append(kde_group)
                else:
                    print(f"Insufficient data for class: {class_name}. Skipping KDE calculation.")

            # Create line traces for each class
            line_traces = []
            for class_name, color in class_colors.items():
                class_data = kde_line_chart_data[kde_line_chart_data['class_name'] == class_name]
                line_trace = go.Scatter(
                    x=class_data['level'],
                    y=class_data['PDF'],
                    mode='lines',
                    name=class_name,
                    line=dict(color=color)
                )
                line_traces.append(line_trace)

            # Create the line chart figure
            line_chart_fig = go.Figure(data=line_traces)

            # Configure the layout for the line chart figure
            line_chart_fig.update_layout(
                xaxis=dict(title='Death Level'),
                yaxis=dict(title='PDF'),
                width=image_width,
                height=image_height,
                margin=dict(l=0, r=0, t=0, b=0)  # Set the margins to 0 on all sides
            )

            return line_chart_fig

        except:
            return None
        
    line_chart_fig = create_pdf_line_chart(filtered_df, class_colors, image_width, image_height)
    if line_chart_fig:
        # Show the line chart figure
        st.plotly_chart(main_fig)
    else:
        st.write('No line chart found')
        
        
    #@title Class Statistics per Region Function

    def create_bar_chart(class_data, image_width, table_height):
        # Create the table figure
        bar_chart = go.Figure(data=[
            go.Table(
                header=dict(values=['Class Name', 'Record Count', 'Percentage', 'Average Level'],
                            fill_color='lightgray',
                            align='left'),
                cells=dict(values=[class_data['Class Name'],
                                class_data['Record Count'],
                                class_data['Percentage'],
                                class_data['Average Level'].apply(lambda x: round(x, 2))],
                        #fill=dict(color=[class_colors.get(class_name, 'white') for class_name in class_data['Class Name']]),
                        align='left'))
        ])

        # Configure the layout for the table figure
        bar_chart.update_layout(
            title='Class Statistics',
            width=image_width,
            height=table_height,
            margin=dict(l=0, r=0, t=0, b=0)  # Set the margins to 0 on all sides
        )

        return bar_chart
    
        # Create bar chart data for each class
    
    
    table_height = 250
    class_data = filtered_df.groupby('class_name').agg({
        'level': ['count', 'mean']
    }).reset_index()

    class_data.columns = ['Class Name', 'Record Count', 'Average Level']
    class_data['Percentage'] = class_data['Record Count'] / len(filtered_df) * 100
    class_data['Percentage'] = class_data['Percentage'].round(2)

    # Sort the table by descending record count
    class_data = class_data.sort_values('Record Count', ascending=False)

    # Add the aggregate row "Total"
    total_record_count = class_data['Record Count'].sum()
    total_average_level = class_data['Average Level'].mean()
    total_percentage = class_data['Percentage'].astype(float).sum().round(2)
    class_data = class_data._append({'Class Name': 'Total', 'Record Count': total_record_count, 'Average Level': total_average_level, 'Percentage': str(total_percentage)}, ignore_index=True)

    # Create the table figure
    table_fig = create_bar_chart(class_data, image_width, table_height)
    if table_fig:
        # Show the line chart figure
        st.plotly_chart(table_fig)
    else:
        st.write('No bar chart created')    
        
    
    #@title NPC Statistics per Region Function

    def create_npc_table(top_npcs, image_width, table_height):
        # Create the table figure for top NPCs
        npc_table_fig = go.Figure(data=[
            go.Table(
                header=dict(values=['NPC Name', 'Record Count', 'Average Level', 'Type'],
                            fill_color='lightgray',
                            align='left'),
                cells=dict(values=[top_npcs['NPC Name'],
                                top_npcs['Record Count'],
                                top_npcs['Average Level'].apply(lambda x: round(x, 2)),
                                top_npcs['Type']],
                        align='left'))
        ])

        # Configure the layout for the NPC table figure
        npc_table_fig.update_layout(
            title='Top 10 NPCs with Highest Records',
            width=image_width,
            height=table_height,
            margin=dict(l=0, r=0, t=0, b=0)  # Set the margins to 0 on all sides
        )

        return npc_table_fig
        
    # Create a table showing the top 10 NPCs with highest records
    top_npcs = filtered_df.groupby('npc_name').agg({
        'npc_avg_level': ['count', 'mean'],
        'npc_type': 'first'
    }).reset_index()

    top_npcs.columns = ['NPC Name', 'Record Count', 'Average Level', 'Type']
    top_npcs = top_npcs.sort_values('Record Count', ascending=False).head(10)

    # Create the table figure for top NPCs
    npc_table_fig = create_npc_table(top_npcs, image_width, table_height)       
    if npc_table_fig:
        # Show the line chart figure
        st.plotly_chart(npc_table_fig)
    else:
        st.write('No bar chart created')          

def machine_learning_plot(df):
    st.header('Under construction')
    
    return None


#sidebar list
if options == 'Data Stats':
    stats(df)
elif options == 'Data Header':
    header(df)
elif options == 'Summary Plot':
    summary_plot(df)
elif options == 'Top 10':
    top_ten(df)    
elif options == 'Deaths by Class & Level':    
    deaths_by_class_Level(df)
elif options == 'Players Alive by Level':
    placeholder.empty()
    player_alive(df)
elif options == 'NPC Level versus Player Level':
    placeholder.empty()
    npc_player(df)
elif options == 'Map plot':
    placeholder.empty()
    map_plot(df)
elif options == 'Machine learning plot':
    placeholder.empty()
    machine_learning_plot(df)

























