import gradio as gr
from gradio import outputs, inputs
import numpy as np
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta

poly = pickle.load(open('model_1.pkl', 'rb'))

history = dict()

history['Aanntpl datumtijd'] = []
history['Prognose datumtijd'] = []
history['Prognose duur'] = []
history['Voorspelling datumtijd'] = []
history['Voorspelling duur'] = []


dash_beschrijving = '''
Als input verwacht het dashboard: de prognose van de aannemer en het tijdstip wanneer de aannemer ter plaatse was.
Om de reparatie duur prognose te berekenen en uiteindelijk als input te gebruiken voor het model.

RMSE waardes:
    Prognose van de aannemer: RMSE is 45.59.
    Gewijzigde prognose door model: RMSE is 35.55.


    wat is RMSE:
    RMSE duid de gemiddelde kwaliteit van voorspellingen aan.
    De RMSE geeft aan hoe groot de fout gemiddeld was van een groep voorspellingen, hoe lager de RMSE hoe beter de voorspellingen.
'''


def predict(aanntpl_ddt, prog_ddt):
    # parse feature vars
    aanntpl_ddt = datetime.strptime(aanntpl_ddt, '%d/%m/%Y %H:%M:%S')
    prog_ddt = datetime.strptime(prog_ddt, '%d/%m/%Y %H:%M:%S')

    # mins
    reparatie_prog_duur = int((prog_ddt - aanntpl_ddt).total_seconds() / 60)

    # model
    X = np.array(reparatie_prog_duur).reshape(-1,1)
    pred = round(poly.predict(X)[0])

    # ddt van voorspelling = melding + prognose
    voorspelling_ddt = aanntpl_ddt + timedelta(minutes=pred)

    #Gegevens worden toegevoegd aan het dataframe voor de historybox
    history['Aanntpl datumtijd'].insert(0, aanntpl_ddt)
    history['Prognose datumtijd'].insert(0, prog_ddt)
    history['Prognose duur'].insert(0, reparatie_prog_duur)
    history['Voorspelling datumtijd'].insert(0, voorspelling_ddt)
    history['Voorspelling duur'].insert(0, pred)

    # plot
    line = np.linspace(1, 360, 2).reshape(-1,1)
    plt.xlabel("Prognose")
    plt.ylabel("Reparatie duur")
    plt.title("Lineair regressie model")
    plt.plot(line, poly.predict(line), '-r', linewidth=3)

    #Van te voren berekende RMSE's
    prog_aannemer = 45.59
    prog_model = 35.55

    return pred, plt, prog_aannemer, prog_model, pd.DataFrame(history), "nostra.png"


iface = gr.Interface(
    fn=predict,
    title="Nostradamus",
    allow_flagging=False,
    allow_screenshot=False,
    theme='darkgrass',
    description=dash_beschrijving,
    inputs = [inputs.Textbox(lines=1, placeholder=None, default="11/01/2018 17:26:56", label='Datum-tijd Aannemer ter plaatse'),
              inputs.Textbox(lines=1, placeholder=None, default="11/01/2018 20:00:01", label='Datum-tijd Prognose')],
    outputs= [outputs.Textbox(type="number", label="Voorspelling reparatieduur (minuten):"),
              outputs.Image(type="plot", label="Grafiek model"),
              outputs.Textbox(type="number", label="RMSE Prognose aannemer:"),
              outputs.Textbox(type="number", label="RMSE Voorspelling Nostradamus:"),
              outputs.Dataframe(type="pandas", label='Voorgaande voorspellingen'),
              outputs.Image(type="file", label="Michel de Nostredame"),],

    examples=[["D/M/Y H:M:S", "D/M/Y H:M:S"],
              ["25/05/2017 17:15:00", "25/05/2017 17:50:00"]]
    ).launch()


