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

abstract_text = '''
The Hardcore Challenge has emerged as a prominent phenomenon within the World of Warcraft (WoW) community, wherein players voluntarily subject themselves to a unique and demanding gameplay mode. Whereas players within this online role-playing environment typically quest within the lands of Azeroth without substantial long-lasting consequences, the Hardcore Challenge reverses this paradigm by forcing players to restart their journey from scratch upon character death. 


This concept of 'one-life' or 'death=delete' has illuminated the game's more formidable aspects, bringing attention to encounters the community finds particularly challenging.  This report delves into the intricate landscape of the Hardcore Challenge, exploring the many causes of death players encounter as they navigate Azeroth's perilous realms.


The core of this report comprises a detailed analysis of various WoW leveling zones. Each region undergoes meticulous examination, illuminating distinctive trials such as treacherous caverns, densely populated regions teeming with hostile entities, and encounters with formidable adversaries. The outcome is an informed evaluation of the prevalent reasons for player mortality, providing valuable guidance for those embarking on the Hardcore Challenge.'''
#st.text()
abstract = st.markdown('''### Abstract:  ''' + abstract_text)

##upload file
st.sidebar.title('Navigation')
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv('clean.csv')

##layout
options = st.sidebar.radio('Pages',
                           options = ['Home',
                                      'Data Stats',
                                      'Data Header',
                                      'Summary Plot',
                                      'Top 10',
                                      'Deaths by Class & Level',
                                      'Pre Clustering Heatmap',
                                      'Clustering Heatmap'])


# @title Function: define_class_colors()

# Purpose:
# Define Class Colors


def define_class_colors():

    class_colors = {
        'Druid': '#FF7D0A',      # Orange
        'Hunter': '#ABD473',     # Green
        'Mage': '#69CCF0',       # Light blue
        'Paladin': '#F58CBA',    # Pink
        'Priest': '#FFFFFF',     # White
        'Rogue': '#FFF569',      # Light yellow
        'Shaman': '#0070DE',     # Blue
        'Warlock': '#9482C9',    # Purple
        'Warrior': '#C79C6E',    # Tan
        'Total':  '#000000'      # Black
    }
    return class_colors

# @title Function: define_race_colors()

# Purpose:
# Define Race Colors

def define_race_colors():

    race_colors = {
        'Human': '#006699',      # Blue
        'Orc': '#FF4400',        # Orange
        'Dwarf': '#FFCC33',      # Yellow
        'Night Elf': '#9933FF',  # Purple
        'Undead': '#660066',     # Dark Purple
        'Tauren': '#C79C6E',     # Tan
        'Gnome': '#66FFCC',      # Cyan
        'Troll': '#336600'       # Green
    }
    return race_colors

# Usage example:
# race_colors = define_race_colors()

# @title Function: create_count_table()

# Purpose:
# Create Count Table for Visuals

def create_count_table(Main_df):

    count_table = pd.pivot_table(Main_df, index='class_name', columns='level', aggfunc='size', fill_value=0)
    total_row = count_table.sum(axis=0)
    total_row.name = 'Total'
    count_table = count_table._append(total_row)

    return count_table

# Usage example:
# count_table = create_count_table(Main_df)
# @title Function: create_players_alive_by_level_df()

def create_players_alive_by_level_df(count_table):

    percent_table = count_table.copy()
    percent_table = percent_table.div(percent_table.sum(axis=1), axis=0).mul(100)
    percent_table = 100 - percent_table.cumsum(axis=1).round(2)

    players_alive_by_level_df = percent_table

    return players_alive_by_level_df

# Usage example:
# players_alive_by_level_df = create_players_alive_by_level_df(count_table)

# @title Visual: race_summary_chart()

# Purpose:
# Create Character Race Summary Chart

def race_summary_chart(Main_df, race_colors):

    # Create a new DataFrame with race_name counts
    summary_df = Main_df['race_name'].value_counts().reset_index()
    summary_df.columns = ['race_name', 'count']

    # Calculate the percentage of total records
    total_records = len(Main_df)
    summary_df['percentage'] = (summary_df['count'] / total_records) * 100

    # Create the bar chart using Plotly
    fig = go.Figure(data=[
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

    fig.update_layout(
        title='Total Deaths Recorded by Race',
        xaxis_title='Race Name',
        yaxis_title='Percentage of Total Records',
        yaxis_tickformat='.0f%',  # Format y-axis tick labels as percentage with no decimal places and "%" symbol
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Set plot background color to transparent
        width=700,
        height=500,
        
    )

    # Show the figure
    st.plotly_chart(fig)

# Usage example:
# race_summary_chart(Main_df, race_colors)

# @title Visual: class_summary_chart()

# Purpose:
# Create Character Class Summary Chart

def class_summary_chart(Main_df, class_colors):

    # Create a new DataFrame with class_name counts
    summary_df = Main_df['class_name'].value_counts().reset_index()
    summary_df.columns = ['class_name', 'count']

    # Calculate the percentage of total records
    total_records = len(Main_df)
    summary_df['percentage'] = (summary_df['count'] / total_records) * 100

    # Create the bar chart using Plotly
    fig = go.Figure(data=[
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

    fig.update_layout(
        title='Total Deaths Recorded by Class',
        xaxis_title='Class Name',
        yaxis_title='Percentage of Total Records',
        yaxis_tickformat='.0f%',  # Format y-axis tick labels as percentage with no decimal places and "%" symbol
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Set plot background color to transparent
        width=700,
        height=500,
    )

    # Show the figure
    st.plotly_chart(fig)

# Usage example:
# class_summary_chart(Main_df, class_colors)

# @title Visual: level_summary_chart()

# Purpose:
# Create Character Level Summary Chart

def level_summary_chart(Main_df):

    # Create a new DataFrame with level counts
    summary_df = Main_df['level'].value_counts().reset_index()
    summary_df.columns = ['level', 'count']

    # Calculate the percentage of total records
    total_records = len(Main_df)
    summary_df['percentage'] = (summary_df['count'] / total_records) * 100

    fig = go.Figure(data=[
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

    fig.update_layout(
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
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Set plot background color to transparent
        width=700,
        height=500,
    )

    # Show the figure
    st.plotly_chart(fig)

# Usage example:
# level_summary_chart(Main_df)

# @title Visual: zone_summary_chart()

# Purpose:
# Create Zone Summary Chart

def zone_summary_chart(Main_df):

    # Create a new DataFrame with map_name counts
    summary_df = Main_df['map_name'].value_counts().reset_index()
    summary_df.columns = ['map_name', 'count']

    # Calculate the percentage of total records
    total_records = len(Main_df)
    summary_df['percentage'] = (summary_df['count'] / total_records) * 100

    fig = go.Figure(data=[
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

    fig.update_layout(
        title='Total Deaths Recorded by Zone',
        xaxis_title='Map Name',
        yaxis_title='Percentage of Total Records',
        yaxis_tickformat='.0f%',  # Format y-axis tick labels as percentage with no decimal places and "%" symbol
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Set plot background color to transparent
        width=700,
        height=500,
    )

    # Show the figure
    st.plotly_chart(fig)

# Usage example:
# zone_summary_chart(Main_df)

# @title Visual: top_death_sources_by_class()

# Purpose:
# Create Top Death Sources by Class Chart

def top_death_sources_by_class(Main_df, class_colors, top_n):

    # Get the top npc_names per class
    top_npc_names = Main_df.groupby(['class_name', 'npc_name']).size().groupby(level=0).nlargest(top_n).reset_index(level=0, drop=True).reset_index(name='count')

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
        title='Top Death Sources by Class',
        height=200 * len(summary_df.columns[1:]),  # Adjust the height based on the number of classes
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Set plot background color to transparent
        showlegend=False,
    )

    # Show the figure
    st.plotly_chart(fig)

# Usage example:
# top_death_sources_by_class(Main_df, class_colors, 3)

# @title Visual: deaths_by_class_and_level_chart()

# Purpose:
# Create Deaths by Class & Level Chart

def deaths_by_class_and_level_chart(count_table, class_colors):

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
        title='Total Deaths Recorded by Class and Level',
        xaxis=dict(title='Level'),
        yaxis=dict(title='Count'),
        hovermode='closest',
        legend=dict(title='Class'),
        showlegend=True,
        width=700,
        height=500,
    )

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)

    # Show the figure
    st.plotly_chart(fig)

# Usage example:
# deaths_by_class_and_level_chart(create_count_table(Main_df), class_colors)

# @title Visual: percentage_players_alive_by_level()

# Purpose:
# Create Percentage of Players Alive by Level Chart

def percentage_players_alive_by_level(players_alive_by_level_df, class_colors):

    # Convert the percent_table DataFrame to a long format
    df = players_alive_by_level_df.reset_index().melt(id_vars='class_name', var_name='level', value_name='percentage')

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
        showlegend=True,
        width=700,
        height=500,
    )

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)

    # Show the figure
    st.plotly_chart(fig)

# Usage example:
# percentage_players_alive_by_level(players_alive_by_level_df, class_colors)

# @title Visual: pre_clustering_heatmap_visual()

# Purpose:
# Create Pre-Clustering Heatmap Visual

def pre_clustering_heatmap_visual(Main_df,map_dict,map_path):
    # Read the image file and encode it to base64
    Main_df["map_id"] = Main_df["map_id"].astype(int)
    Main_df["map_name"] = Main_df["map_id"].map(map_dict)
    map_list = Main_df['map_name'].unique().tolist()

    map_name  = st.selectbox('Select Map', options = map_list)
    st.write(map_name)
    # Iterate over unique map names in the DataFrame
    #unique_map_names = df['map_name'].unique()
    
    # Read the image file and encode it to base64    
    with open(f'{map_path}/{map_name}.jpg', 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    # Get the image dimensions
    with Image.open(image_file.name) as img:
        image_width, image_height = img.size

    # Filter the DataFrame based on the map name
    filtered_df = Main_df[Main_df['map_name'] == map_name]

    # Create the main map figure
    fig = go.Figure()

    # Add the main map image as a layout image
    fig.add_layout_image(
        source='data:image/jpeg;base64,' + encoded_image,
        x=0,
        y=1,
        sizex=1,
        sizey=1,
        sizing='stretch',
        opacity=1,
        layer='below'
    )

    # Scale the x and y coordinates based on the image dimensions
    scaled_x = filtered_df['x'] * image_width
    scaled_y = (1 - filtered_df['y']) * image_height

    # Add all the points as a scatter plot
    fig.add_trace(go.Scatter(
        x=scaled_x,
        y=scaled_y,
        mode='markers',
        marker=dict(
            size=10,
            opacity=0.5,
        ),
        showlegend=False
    ))

    # Configure the layout with adjusted axes ranges
    fig.update_layout(
        width=800,
        height=600,
        xaxis=dict(visible=False, range=[0, image_width]),
        yaxis=dict(visible=False, range=[image_height, 0]),  # Invert the y-axis range
        margin=dict(l=0, r=0, t=0, b=0)  # Set the margins to 0 on all sides
    )

    # Show the figure
    st.plotly_chart(fig)

# Usage example:
# pre_clustering_heatmap_visual(input("Enter the map name: "), Main_df)

# @title Visual: create_heatmap()

# Purpose:
# Create Heatmap for Region Visual

def create_heatmap(Main_df, map_dict, map_path, class_colors, map_name):
    
    try:
        # Read the image file and encode it to base64
        with open(f'{map_path}/{map_name}.jpg', 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()

        # Get the image dimensions
        with Image.open(image_file.name) as img:
            image_width, image_height = img.size

        # Filter the DataFrame based on the map name
        filtered_df = Main_df[Main_df['map_name'] == map_name]

        # Perform DBSCAN clustering on the scaled coordinates (x, y)
        X = filtered_df[['x', 'y']]
        eps = 0.02
        min_samples_percent = 0.0175
        min_samples = max(int(len(X) * min_samples_percent), 2)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(X)

        # Add the cluster labels as a column in the DataFrame
        filtered_df['cluster_label'] = cluster_labels

        # Create the main map figure
        fig = go.Figure()

        # Add the main map image as a layout image
        fig.add_layout_image(
            source='data:image/jpeg;base64,' + encoded_image,
            x=0,
            y=1,
            sizex=1,
            sizey=1,
            sizing='stretch',
            opacity=1,
            layer='below'
        )

        # Iterate over unique cluster labels
        for cluster_label in filtered_df['cluster_label'].unique():
            if cluster_label == -1:
                continue

            # Filter the DataFrame for the current cluster label
            cluster_df = filtered_df[filtered_df['cluster_label'] == cluster_label]

            # Scale the x and y coordinates based on the image dimensions
            scaled_x = cluster_df['x'] * image_width
            scaled_y = (1 - cluster_df['y']) * image_height

            # Determine the opacity and color for the clusters
            opacity = 0.1 if cluster_label >= 0 else 0.5
            color = 'lightgray' if cluster_label == -1 else None

            # Calculate cluster center
            center_x = scaled_x.mean()
            center_y = scaled_y.mean()

            # Create hover text with cluster details
            hover_text = (
                f"<b>{cluster_df['npc_name'].mode().values[0]}</b><br>"
                f"<b>lvl: {cluster_df['level'].median():.0f}</b><br>"
            )

            # Add the clustered records as a scatter plot with hover text
            fig.add_trace(go.Scatter(
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

            # Add label for cluster center
            fig.add_annotation(
                x=center_x,
                y=center_y,
                xref='x',
                yref='y',
                text=hover_text,
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-30
            )

        # Configure the layout with adjusted axes ranges
        fig.update_layout(
            width=800,
            height=600,
            xaxis=dict(visible=False, range=[0, image_width]),
            yaxis=dict(visible=False, range=[image_height, 0]),
            margin=dict(l=0, r=0, t=0, b=0)
        )

        # Show the figure
        st.plotly_chart(fig)
        
        #return  

    except FileNotFoundError:
        print("Map file not found. Skipping...\n")
        
# @title Visual: create_pdf_line_chart()

# Purpose:
# Create PDF chart for region visual

def create_pdf_line_chart(Main_df, map_path, map_name, class_colors): 

    filtered_df = Main_df[Main_df['map_name'] == map_name]

    with open(f'{map_path}/{map_name}.jpg', 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    # Get the image dimensions
    with Image.open(image_file.name) as img:
        image_width, image_height = img.size

    #try:
        # Create a line chart for the PDF of death level by class
    line_chart_data = filtered_df.groupby(['class_name', 'level']).size().reset_index(name='count')
    #line_chart_data['PDF'] = line_chart_data.groupby('class_name')['count'].apply(lambda x: x / x.sum() * 100)
    line_chart_data['PDF'] = line_chart_data.groupby('class_name')['count'].transform(lambda x: x / x.sum() * 100)

    # Smooth the PDF curves using KDE
    kde_line_chart_data = pd.DataFrame()
    for class_name, group in line_chart_data.groupby('class_name'):
        kde = gaussian_kde(group['level'])
        x_vals = np.linspace(group['level'].min(), group['level'].max(), num=100)
        kde_vals = kde(x_vals)
        kde_group = pd.DataFrame({'level': x_vals, 'PDF': kde_vals, 'class_name': class_name})
        kde_line_chart_data = kde_line_chart_data._append(kde_group)

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
        margin=dict(l=0, r=0, t=0, b=0)
    )

    st.plotly_chart(line_chart_fig)
        
    return 

    #except Exception as e:
        #print("An error occurred:", str(e))
        #return None

# @title Visual: create_class_table()

# Purpose:
# Create Class Data Bar Chart for Region Visual

def create_class_table(Main_df, map_data_path, map_name):

    filtered_df = Main_df[Main_df['map_name'] == map_name]

    with open(f'{map_path}/{map_name}.jpg', 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    # Get the image dimensions
    with Image.open(image_file.name) as img:
        image_width, image_height = img.size


    # Create bar chart data for each class
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
    table_fig = go.Figure(data=[
        go.Table(
            header=dict(values=['Class Name', 'Record Count', 'Percentage', 'Average Level'],
                        fill_color='lightgray',
                        align='left'),
            cells=dict(values=[class_data['Class Name'],
                               class_data['Record Count'],
                               class_data['Percentage'],
                               class_data['Average Level'].apply(lambda x: round(x, 2))],
                       align='left'))
    ])

    # Configure the layout for the table figure
    table_fig.update_layout(
        title='Class Statistics',
        width=image_width,
        height=250,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    st.plotly_chart(table_fig)
    
    return 

# @title Visual: create_npcs_table()

# Purpose:
# Create NPC table for region visual

def create_top_npcs_table(Main_df, map_data_path, map_name):

    filtered_df = Main_df[Main_df['map_name'] == map_name]

    # Read the image file and encode it to base64
    with open(f'{map_path}/{map_name}.jpg', 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    # Get the image dimensions
    with Image.open(image_file.name) as img:
        image_width, image_height = img.size

    # Create a table showing the top 10 NPCs with highest records
    top_npcs = filtered_df.groupby('npc_name').agg({
        'npc_avg_level': ['count', 'mean'],
        'npc_type': 'first'
    }).reset_index()

    top_npcs.columns = ['NPC Name', 'Record Count', 'Average Level', 'Type']
    top_npcs = top_npcs.sort_values('Record Count', ascending=False).head(10)

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
        height=250,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(npc_table_fig)
    return 

# Example usage:
# create_top_npcs_table(Main_df, map_data_path, map_name)

# @title Visual: region_analysis()

def region_analysis(region_analysis_df, map_name):
    try:
        df = region_analysis_df

        # Filter the DataFrame based on the map name
        notes = df[df['Zone'] == map_name]['Notes'].values

        if len(notes) > 0:
            st.markdown(notes[0])
            return
        else:
            return None

    except Exception as e:
        print("An error occurred:", str(e))
        return None

# @title Function: create_count_table()

# Purpose:
# Create Count Table for Visuals

def create_count_table(Main_df):

    count_table = pd.pivot_table(Main_df, index='class_name', columns='level', aggfunc='size', fill_value=0)
    total_row = count_table.sum(axis=0)
    total_row.name = 'Total'
    count_table = count_table._append(total_row)

    return count_table

# Usage example:
# count_table = create_count_table(Main_df)

# @title Function: load_region_analysis()

# Purpose:
# Loads region analysis into model

def load_region_analysis(region_summary_path):

  data = pd.read_excel(region_summary_path, sheet_name='Zone Summary')

  return data

# Usage example:
# load_region_analysis("your_data_path.xlsx")

#sidebar functions

def stats(df):
    placeholder.empty()
    st.header('Data Stats')
    st.write(df.describe())
    
def header(df):
    placeholder.empty()
    st.header('Data Header')
    st.write(df.head())


# Define Class & Race Colors
class_colors = define_class_colors()                                                                                # Create class_colors dictionary
race_colors = define_race_colors()   

# Create Visual Aid Dataframes
count_table = create_count_table(df)                                                                           # Create count_table Count of Deaths by Class & Level df
players_alive_by_level_df = create_players_alive_by_level_df(count_table)    
#
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

map_path = 'data/Maps'
region_summary_path = 'data/Regional Analysis.xlsx' 

region_analysis_df = load_region_analysis(region_summary_path)
print(region_analysis_df)
#sidebar list
if options == 'Data Stats':
    abstract.empty()
    placeholder.empty()
    stats(df)
elif options == 'Data Header':
    abstract.empty()
    placeholder.empty()
    header(df)
elif options == 'Summary Plot':
    abstract.empty()
    placeholder.empty()
    race_summary_chart(df, race_colors)
    class_summary_chart(df, class_colors)
    level_summary_chart(df)
    zone_summary_chart(df)
    
elif options == 'Top 10':
    abstract.empty()
    placeholder.empty()
    top_death_sources_by_class(df, class_colors, 3)   
elif options == 'Deaths by Class & Level': 
    abstract.empty()   
    placeholder.empty()
    deaths_by_class_and_level_chart(count_table, class_colors)
    level_summary_chart(df)


elif options == 'Pre Clustering Heatmap':
    abstract.empty()
    placeholder.empty()
    pre_clustering_heatmap_visual(df,map_dict,map_path)
elif options == 'Clustering Heatmap':
    abstract.empty()
    placeholder.empty()
    
    # Read the image file and encode it to base64
    df["map_id"] = df["map_id"].astype(int)
    df["map_name"] = df["map_id"].map(map_dict)
    map_list = df['map_name'].unique().tolist()

    map_name  = st.selectbox('Select Map', options = map_list)

    st.markdown("#### Heatmap")
    create_heatmap(df, map_dict, map_path, class_colors, map_name)

    st.markdown("#### Class Table")
    create_class_table(df, map_path, map_name)

    st.markdown("#### Top Mobs Table")
    create_top_npcs_table(df, map_path, map_name)

    st.markdown("#### Class Deaths by Level")
    create_pdf_line_chart(df, map_path, map_name, class_colors)

    st.markdown("#### Interpretation")
    region_analysis(region_analysis_df, map_name)
