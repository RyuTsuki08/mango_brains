from flask import Blueprint, render_template
import plotly
import json
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# Para extraer la data del csv - modulo "data_analizer"
from data_analizer.data import X, y , df,General_df, df_pca, df_negative, df_neutral, df_positive, Needed_columns, Sample_df, Sample_df_0, Sample_df_1
import plotly.express as px

plots = Blueprint('plots', __name__)

@plots.route('/plots')
def index():
    # Graph 1
    fig1 = px.line(Sample_df, title="Sample EEG Signal")
    fig1.update_xaxes(title_text="Time")
    fig1.update_yaxes(title_text="Amplitude")
    
    fig2 = make_subplots(rows=3, cols=1, subplot_titles=("Neutral - FFT", "Negative - FFT", "Positive - FFT"))

    fig2.append_trace(go.Scatter(
    x=df_neutral.columns,
    y=df.loc[1463, 'fft_0_b':'fft_749_b'[:50]],
    name="Neutral"
    ), row=1, col=1)

    fig2.append_trace(go.Scatter(
        x=df_negative.columns,
        y=df.loc[713, 'fft_0_b':'fft_749_b'[:50]],
        name="Negative"
    ), row=2, col=1)

    fig2.append_trace(go.Scatter(
        x=df_positive.columns,
        y=df.loc[940, 'fft_0_b':'fft_749_b'[:50]],
        name="Positive"
    ), row=3, col=1)


    fig2.update_layout(height=900, title_text="EEG - FFT - 3 samples")
    
    fig3 = go.Figure()
    
    fig4 = px.bar(Sample_df_0, x=Sample_df_0["label"], y="entropy", color="label")
    
    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    
    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    
    graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
    
    
    data = {
        'graph1JSON': graph1JSON,
        'graph2JSON': graph2JSON,
        'graph3JSON': graph3JSON,
        'graph4JSON': graph4JSON,
        'title': 'Graph 1'
    }

    return render_template('plots.html', data=data)