import pandas as pd 
import plotly as plty 


def create_dt(file):
    # Read the CSV file into a DataFrame.
    df = pd.read_csv(file)
    return df

def plot_data(df, column, title, x_axis_label, y_axis_label):
    fig = plty.Figure(data=[plty.Bar(x=df['Date'], y=df[column])])
    fig.update_layout(title=title, xaxis_title=x_axis_label, yaxis_title=y_axis_label)
    fig.show()
    