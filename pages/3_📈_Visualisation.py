import streamlit as st

st.set_page_config(
    page_title="Visualisation des données",
    layout="wide",
    menu_items={}
)

import uuid
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns

from Libraries.Data import vgsales_cleaned_df

st.markdown("""
        <style>
               .block-container {
                    padding-top: 3rem;
                    padding-bottom: 2rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

df = vgsales_cleaned_df.copy()
li_salon = ['Wii', 'NES', 'X360', 'PS3', 'PS2', 'SNES', 'PS4', 'N64', 'PS', 'XB', 'PC', '2600', 'XOne', 'GC', 'GEN',
            'DC', 'SAT', 'SCD', 'NG', 'TG16', '3DO', 'PCFX']
li_portable = ['GB', 'DS', 'GBA', '3DS', 'PSP', 'WiiU', 'PSV', 'WS', 'GG']

df['Type'] = np.where(df['Platform'].isin(li_salon), 'Salon', 'Portable')

df['Year'] = df['Year'].astype(int)

st.image("Images/waves_band.png")
st.title("Visualisation des informations")
st.write("### Evolution du nombre de jeux sortis par année")

fig = go.Figure()
fig.add_trace(go.Histogram(x=df.Year,
                           marker=dict(color='darkorange'),
                           marker_line=dict(width=2, color='black')))
fig.update_layout(bargap=0.2, title='Évolution du nombre de jeux sortis par année',
                xaxis_title = "Année de sortie",
                yaxis_title = "Nombre de jeux",
                height = 500)
st.plotly_chart(fig, key = uuid.uuid4())

st.write("### Comparaison du nombre de jeux sortis et celui du nombre de ventes médian par année de sortie")
fig, ax = plt.subplots(figsize=(13, 6))
sns.set_style("whitegrid", {'axes.grid' : False})
sns.lineplot(x='Year', y='Global_Sales', data=df, ax=ax, label='Nombre de ventes médian des jeux\n(en million)\nAvec la répartition inter-percentiles\n(2.5%-97.5%)', errorbar="pi", estimator="median")
sns.move_legend(ax, "upper left")
ax.set_xlabel('Année de sortie des jeux', labelpad = 15, fontsize = 16)
ax.set_ylabel('Nombre de ventes\n(en million)', labelpad = 15, fontsize = 16)
ax.set_title('Nombre de ventes médian des jeux par année de sortie\nNombre de jeux sortis par année', fontsize = 16)
ax2 = ax.twinx()
game_counts = df.Year.value_counts().sort_index()
# sns.lineplot(x=game_counts.index, y=game_counts.values, ax = ax2, label='Nombre de jeux sortis')
df.Year.value_counts().sort_index().plot(ax=ax2, color='orange', kind='area', alpha=0.2, legend ='Nombre de jeux sortis')
# print(data_no_nan.Year.value_counts().sort_index())
# sns.countplot(x='Year', data=data_no_nan, ax=ax2, color ='red', alpha=0.2, edgecolor='black', label='Nombre de jeux')
# ax.sharex(ax2)
ax2.set_xlabel('')
ax2.set_ylabel('Nombre de jeux',  fontsize = 16)
ax2.legend(['Nombre de jeux sortis'], fontsize = 9)
st.pyplot(fig)

################## GRAPH TOP 10
st.write("### Top 10 des jeux par régions")

with st.expander("Afficher TOP 10 des jeux"):
    top10_EU = df[['Name', 'EU_Sales', 'Publisher', 'Genre']].sort_values(by='EU_Sales').tail(10)
    top10_JP = df[['Name', 'JP_Sales', 'Publisher', 'Genre']].sort_values(by='JP_Sales').tail(10)
    top10_NA = df[['Name', 'NA_Sales', 'Publisher', 'Genre']].sort_values(by='NA_Sales').tail(10)
    top10_Other = df[['Name', 'Other_Sales', 'Publisher', 'Genre']].sort_values(by='Other_Sales').tail(10)
    top10_Gl = df[['Name', 'Global_Sales', 'Publisher', 'Genre']].sort_values(by='Global_Sales').tail(10)

    fig = make_subplots(rows=5, cols=1,
                        subplot_titles=("Marché Européen",
                                        "Marché Japonais",
                                        "Marché Nord Américain",
                                        "Marché autres région",
                                        "Marché globale"))
    fig.add_trace(
        go.Bar(y=top10_EU["Name"],
              x=top10_EU["EU_Sales"],
              orientation='h',
              text=top10_EU["Publisher"],
              name='Europe'),
        row=1, col=1,
                )
    fig.add_trace(
        go.Bar(y=top10_JP["Name"],
              x=top10_JP["JP_Sales"],
              orientation='h',
              text=top10_JP["Publisher"],
              name='Japon'),
        row=2, col=1
                )
    fig.add_trace(
        go.Bar(y=top10_NA["Name"],
              x=top10_NA["NA_Sales"],
              orientation='h',
              text=top10_NA["Publisher"],
              name='Amérique du Nord'),
        row=3, col=1
                )
    fig.add_trace(
        go.Bar(y=top10_Other["Name"],
              x=top10_Other["Other_Sales"],
              orientation='h',
              text=top10_Other["Publisher"],
              name='Autres régions'),
        row=4, col=1
                )
    fig.add_trace(
        go.Bar(y=top10_Gl["Name"],
              x=top10_Gl["Global_Sales"],
              orientation='h',
              text=top10_Gl["Publisher"],
              name='Monde'),
        row=5, col=1
                )
    fig.update_xaxes(title_text="Nombre de ventes (en million)", row=1, col=1)
    fig.update_xaxes(title_text="Nombre de ventes (en million)", row=2, col=1)
    fig.update_xaxes(title_text="Nombre de ventes (en million)", row=3, col=1)
    fig.update_xaxes(title_text="Nombre de ventes (en million)", row=4, col=1)
    fig.update_xaxes(title_text="Nombre de ventes (en million)", row=5, col=1)

    fig.update_layout(title="Top 10 des jeux par nombre de ventes",
                      xaxis_title="Nombre de ventes (en million)",
                      height=2400,width=800)
    st.plotly_chart(fig, key = uuid.uuid4())
st.write("### Top 5 des éditeurs par régions")

with st.expander("Afficher TOP 5 des éditeurs"):
    pie_data_gl = df.groupby('Publisher').sum().sort_values('Global_Sales', ascending=False).reset_index().head()
    pie_data_na = df.groupby('Publisher').sum().sort_values('NA_Sales', ascending=False).reset_index().head()
    pie_data_eu = df.groupby('Publisher').sum().sort_values('EU_Sales', ascending=False).reset_index().head()
    pie_data_jp = df.groupby('Publisher').sum().sort_values('JP_Sales', ascending=False).reset_index().head()
    pie_data_other = df.groupby('Publisher').sum().sort_values('Other_Sales', ascending=False).reset_index().head()

    fig = make_subplots(rows=3, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]])

    fig.add_trace(go.Pie(values=pie_data_na['NA_Sales'],
                        labels=pie_data_na['Publisher'],
                        pull=[0.15,0,0,0,0], ),
                  1, 1)
    fig.add_trace(go.Pie(values=pie_data_eu['EU_Sales'],
                        labels=pie_data_eu['Publisher'],
                        pull=[0.15,0,0,0,0], ),
                  1, 2)
    fig.add_trace(go.Pie(values=pie_data_jp['JP_Sales'],
                        labels=pie_data_jp['Publisher'],
                        pull=[0.15,0,0,0,0], ),
                  2, 1)
    fig.add_trace(go.Pie(values=pie_data_other['Other_Sales'],
                        labels=pie_data_other['Publisher'],
                        pull=[0.15,0,0,0,0], ),
                  2, 2)
    fig.add_trace(go.Pie(values=pie_data_gl['Global_Sales'],
                        labels=pie_data_gl['Publisher'],
                        pull=[0.15,0,0,0,0], ),
                  3, 1)

    fig.update_traces(hole=.3, hoverinfo="label+percent+name")

    fig.update_layout(
        title_text="Répartition du top 5 des éditeurs par region",
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='NA', x=sum(fig.get_subplot(1, 1).x) / 2, y=(sum(fig.get_subplot(1, 1).y))*1.012 / 2,
                          font_size=20, showarrow=False, xanchor="center"),
                    dict(text='EU', x=sum(fig.get_subplot(1, 2).x) / 2, y=(sum(fig.get_subplot(1, 2).y))*1.012 / 2,
                          font_size=20, showarrow=False, xanchor="center"),
                    dict(text='JP', x=sum(fig.get_subplot(2, 1).x) / 2, y=sum(fig.get_subplot(2, 1).y) / 2,
                          font_size=20, showarrow=False, xanchor="center"),
                    dict(text='Other', x=sum(fig.get_subplot(2, 2).x) / 2, y=sum(fig.get_subplot(2, 2).y) / 2,
                          font_size=20, showarrow=False, xanchor="center"),
                    dict(text='Global', x=sum(fig.get_subplot(3, 1).x) / 2, y=(sum(fig.get_subplot(3, 1).y))*0.98 / 2,
                          font_size=20, showarrow=False, xanchor="center")],
        height=1300, width=800)
    st.plotly_chart(fig, key = uuid.uuid4())
st.write("### Top 5 des plateformes par régions")

with st.expander("Afficher TOP 5 des plateformes"):
    pie_data_gl = df.groupby('Platform').sum().sort_values('Global_Sales', ascending=False).reset_index().head()
    pie_data_na = df.groupby('Platform').sum().sort_values('NA_Sales', ascending=False).reset_index().head()
    pie_data_eu = df.groupby('Platform').sum().sort_values('EU_Sales', ascending=False).reset_index().head()
    pie_data_jp = df.groupby('Platform').sum().sort_values('JP_Sales', ascending=False).reset_index().head()
    pie_data_other = df.groupby('Platform').sum().sort_values('Other_Sales', ascending=False).reset_index().head()

    fig = make_subplots(rows=3, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]])

    fig.add_trace(go.Pie(values=pie_data_na['NA_Sales'],
                        labels=pie_data_na['Platform'],
                        pull=[0.15,0,0,0,0], ),
                  1, 1)
    fig.add_trace(go.Pie(values=pie_data_eu['EU_Sales'],
                        labels=pie_data_eu['Platform'],
                        pull=[0.15,0,0,0,0], ),
                  1, 2)
    fig.add_trace(go.Pie(values=pie_data_jp['JP_Sales'],
                        labels=pie_data_jp['Platform'],
                        pull=[0.15,0,0,0,0], ),
                  2, 1)
    fig.add_trace(go.Pie(values=pie_data_other['Other_Sales'],
                        labels=pie_data_other['Platform'],
                        pull=[0.15,0,0,0,0], ),
                  2, 2)
    fig.add_trace(go.Pie(values=pie_data_gl['Global_Sales'],
                        labels=pie_data_gl['Platform'],
                        pull=[0.15,0,0,0,0], ),
                  3, 1)

    fig.update_traces(hole=.3, hoverinfo="label+percent+name")

    fig.update_layout(
        title_text="Répartition du top 5 des platforme par région",
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='NA', x=sum(fig.get_subplot(1, 1).x) / 2, y=(sum(fig.get_subplot(1, 1).y))*1.012 / 2,
                          font_size=20, showarrow=False, xanchor="center"),
                    dict(text='EU', x=sum(fig.get_subplot(1, 2).x) / 2, y=(sum(fig.get_subplot(1, 2).y))*1.012 / 2,
                          font_size=20, showarrow=False, xanchor="center"),
                    dict(text='JP', x=sum(fig.get_subplot(2, 1).x) / 2, y=sum(fig.get_subplot(2, 1).y) / 2,
                          font_size=20, showarrow=False, xanchor="center"),
                    dict(text='Other', x=sum(fig.get_subplot(2, 2).x) / 2, y=sum(fig.get_subplot(2, 2).y) / 2,
                          font_size=20, showarrow=False, xanchor="center"),
                    dict(text='Global', x=sum(fig.get_subplot(3, 1).x) / 2, y=(sum(fig.get_subplot(3, 1).y))*0.98 / 2,
                          font_size=20, showarrow=False, xanchor="center")],
        height=1300, width=800)
    st.plotly_chart(fig, key = uuid.uuid4())
st.write("### Ventes globales (distinction du type de plateforme)")

with st.expander("Afficher les ventes globales par type de plateforme"):
    df['Platform'].unique()
    platform_count = df['Platform'].value_counts()
    platform_count.columns = ['Platform','Count']
    valides_platform = platform_count[platform_count >= 34].index

    df_vgchartz_filter_platform = df[df['Platform'].isin(valides_platform)]

    df_plat_type_global_sales = df_vgchartz_filter_platform.groupby(['Type','Platform'])['Global_Sales'].sum().reset_index().sort_values(by='Global_Sales', ascending=False)

    df_salon_sales = df_plat_type_global_sales.loc[df_plat_type_global_sales['Type'] == "Salon"]
    df_portable_sales = df_plat_type_global_sales.loc[df_plat_type_global_sales['Type'] == "Portable"]

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Ventes mondiales",))

    fig.add_trace(go.Bar(y = df_salon_sales["Global_Sales"], x = df_salon_sales["Platform"],name="Consoles de salon",text=round(df_salon_sales["Global_Sales"],2), textposition='auto'), row=1, col=1)
    fig.add_trace(go.Bar(y = df_portable_sales["Global_Sales"], x = df_portable_sales["Platform"],name="Consoles portables",text=round(df_portable_sales["Global_Sales"],2), textposition='auto'), row=2, col=1)
    fig.update_layout(width=800,height=800,title_text="Ventes globales de jeux par type d'équipment (en millions de copies vendues)")
    st.plotly_chart(fig, key = uuid.uuid4())
st.write("### Ventes régionales de jeux par type de support")

with st.expander("Afficher les ventes régionales de jeux par type de support"):
    platform_count = df['Platform'].value_counts()
    platform_count.columns = ['Platform','Count']
    valides_platform = platform_count[platform_count >= 34].index

    df_vgchartz_filter_platform = df[df['Platform'].isin(valides_platform)]
    df_plat_na_sales = df_vgchartz_filter_platform.groupby(['Type','Platform'])['NA_Sales'].sum().reset_index().sort_values(by='NA_Sales', ascending=True)
    df_plat_eu_sales = df_vgchartz_filter_platform.groupby(['Type','Platform'])['EU_Sales'].sum().reset_index().sort_values(by='EU_Sales', ascending=True)
    df_plat_jp_sales = df_vgchartz_filter_platform.groupby(['Type','Platform'])['JP_Sales'].sum().reset_index().sort_values(by='JP_Sales', ascending=True)
    df_plat_other_sales = df_vgchartz_filter_platform.groupby(['Type','Platform'])['Other_Sales'].sum().reset_index().sort_values(by='Other_Sales', ascending=True)

    df_salon_na_sales = df_plat_na_sales.loc[df_plat_na_sales['Type'] == "Salon"]
    df_portable_na_sales = df_plat_na_sales.loc[df_plat_na_sales['Type'] == "Portable"]

    df_salon_eu_sales = df_plat_eu_sales.loc[df_plat_eu_sales['Type'] == "Salon"]
    df_portable_eu_sales = df_plat_eu_sales.loc[df_plat_eu_sales['Type'] == "Portable"]

    df_salon_jp_sales = df_plat_jp_sales.loc[df_plat_jp_sales['Type'] == "Salon"]
    df_portable_jp_sales = df_plat_jp_sales.loc[df_plat_jp_sales['Type'] == "Portable"]

    df_salon_other_sales = df_plat_other_sales.loc[df_plat_other_sales['Type'] == "Salon"]
    df_portable_other_sales = df_plat_other_sales.loc[df_plat_other_sales['Type'] == "Portable"]

    fig = make_subplots(rows=4, cols=1, subplot_titles=("Ventes en Amérique du Nord", "Ventes en Europe", "Ventes au Japon", "Ventes dans les autres régions"))

    fig.add_trace(go.Bar(y = df_salon_na_sales["NA_Sales"], x = df_salon_na_sales["Platform"],name="Ventes NA",
             text=round(df_salon_na_sales["NA_Sales"],2), textposition='auto'), row=1, col=1)
    fig.add_trace(go.Bar(y = df_salon_eu_sales["EU_Sales"], x = df_salon_eu_sales["Platform"],name="Ventes EU",
              text=round(df_salon_eu_sales["EU_Sales"],2), textposition='auto'), row=2, col=1)
    fig.add_trace(go.Bar(y = df_salon_jp_sales["JP_Sales"], x = df_salon_jp_sales["Platform"],name="Ventes JP",
              text=round(df_salon_jp_sales["JP_Sales"],2), textposition='auto'), row=3, col=1)
    fig.add_trace(go.Bar(y = df_salon_other_sales["Other_Sales"], x = df_salon_other_sales["Platform"],name="Ventes autres régions",
              text=round(df_salon_other_sales["Other_Sales"],2), textposition='auto'), row=4, col=1)

    fig.update_layout(width=800,height=1200,title_text="Ventes régionales de jeux sur équipments de salons (en millions de copies vendues)")
    st.plotly_chart(fig, key = uuid.uuid4())

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Ventes en Amérique du Nord", "Ventes en Europe", "Ventes au Japon", "Ventes dans les autres régions"))

    fig.add_trace(go.Bar(y = df_portable_na_sales["NA_Sales"], x = df_portable_na_sales["Platform"],name="Ventes NA",text=round(df_portable_na_sales["NA_Sales"],2), textposition='auto'), row=1, col=1)
    fig.add_trace(go.Bar(y = df_portable_eu_sales["EU_Sales"], x = df_portable_eu_sales["Platform"],name="Ventes EU",text=round(df_portable_eu_sales["EU_Sales"],2), textposition='auto'), row=1, col=2)
    fig.add_trace(go.Bar(y = df_portable_jp_sales["JP_Sales"], x = df_portable_jp_sales["Platform"],name="Ventes JP",text=round(df_portable_jp_sales["JP_Sales"],2), textposition='auto'), row=2, col=1)
    fig.add_trace(go.Bar(y = df_portable_other_sales["Other_Sales"], x = df_portable_other_sales["Platform"],name="Ventes autres régions",text=round(df_portable_other_sales["Other_Sales"],2), textposition='auto'), row=2, col=2)

    fig.update_layout(width=800,height=800,title_text="Ventes régionales de jeux sur équipements portables (en millions de copies vendues)")
    st.plotly_chart(fig, key = uuid.uuid4())
st.markdown("### Volume de jeux édités par plateforme au fil du temps pour le type Salon")

with st.expander("Afficher les volumes de jeux édités par plateforme au fil du temps (Salon)"):

    platform_count = df['Platform'].value_counts()
    platform_count.columns = ['Platform','Count']
    valides_platform = platform_count[platform_count >= 34].index

    df_vgchartz_filter_platform = df[df['Platform'].isin(valides_platform)]
    df_plat_year_count = df_vgchartz_filter_platform.groupby(['Type','Platform', 'Year']).size().reset_index(name='Count').sort_values(by='Count', ascending=True)

    df_salon_sales = df_plat_year_count.loc[df_plat_year_count['Type'] == "Salon"].sort_values(by='Year', ascending=True)

    #vu le nombre de plateformes, il faut rajouter plusieurs palettes de couleurs
    #color_sequence = px.colors.qualitative.Dark2 + px.colors.qualitative.Vivid

    fig = px.bar(df_salon_sales,
                y='Platform', x='Count',
                orientation="h",
                hover_data=['Platform', 'Year'], color='Year',
                labels={'Platform'}, height=800, width=800,
                color_continuous_scale=px.colors.sequential.Inferno)


    fig.update_layout(
        title='Volume de jeux édités par plateforme au fil du temps pour le type Salon',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title=dict(
                text="Plateformes de salon",
                font=dict(
                    size=16
                    )
                ),
            tickfont_size=14,
            ),
        legend=dict(
            x=0,
            y=1,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
            ),
        xaxis=dict(
            title=dict(
                text="Nbre de Jeux publiés",
                font=dict(
                    size=16
                    )
                ),
            tickfont_size=14,
            ),

        )
    st.plotly_chart(fig, key = uuid.uuid4())
st.write("### Volume de jeux édités par plateforme au fil du temps pour le type Portable")

with st.expander("Afficher les volumes de jeux édités par plateforme au fil du temps (Portable)"):
    platform_count = df['Platform'].value_counts()
    platform_count.columns = ['Platform','Count']
    valides_platform = platform_count[platform_count >= 34].index

    df_vgchartz_filter_platform = df[df['Platform'].isin(valides_platform)]
    df_plat_year_count = df_vgchartz_filter_platform.groupby(['Type','Platform', 'Year']).size().reset_index(name='Count').sort_values(by='Count', ascending=False)

    df_portable_sales = df_plat_year_count.loc[df_plat_year_count['Type'] == "Portable"].sort_values(by='Year', ascending=True)

    #vu le nombre de plateformes, il faut rajouter plusieurs palettes de couleurs
    #color_sequence = px.colors.qualitative.Dark2

    fig = px.bar(df_portable_sales,
                y='Platform', x='Count',
                hover_data=['Platform', 'Year'], color='Year',
                labels={'Platform'}, height=800, width=800,orientation="h",
                color_continuous_scale=px.colors.sequential.Inferno)


    fig.update_layout(
        title='Volume de jeux édités par plateforme au fil du temps pour le type portable',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title=dict(
                text="Plateformes portables",
                font=dict(
                    size=16
                    )
                ),
            tickfont_size=14,
            ),
        legend=dict(
            x=0,
            y=1,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
            ),
        xaxis=dict(
            title=dict(
                text="Nbre de Jeux publiés",
                font=dict(
                    size=16
                    )
                ),
            tickfont_size=14,
            ),

        )
    st.plotly_chart(fig, key = uuid.uuid4())
st.write("### Durée moyenne du marché de développement d'un jeu sur une plateforme")

with st.expander("Durée moyenne du marché de développement d'un jeu"):
    platform_count = df['Platform'].value_counts()
    platform_count.columns = ['Platform','Count']
    valides_platform = platform_count[platform_count >= 34].index

    df_vgchartz_filter_platform = df[df['Platform'].isin(valides_platform)]
    df_plat_year_count = df_vgchartz_filter_platform.groupby(['Type','Platform', 'Year']).size().reset_index(name='Count').sort_values(by='Count', ascending=False)
    df_salon_sales = df_plat_year_count.loc[df_plat_year_count['Type'] == "Salon"].sort_values(by='Year', ascending=False)

    #on va compter le nombre d'années par Platform
    df_salon_year_count = df_salon_sales.groupby(['Platform']).size().reset_index(name='Count').sort_values(by='Count', ascending=False)

    fig = px.bar(df_salon_year_count,
                y='Platform', x='Count',
                hover_data=['Platform', 'Count'],
                color='Count',
                labels={'Platform'}, height=800, width=800,orientation="h",
                color_continuous_scale=px.colors.sequential.Inferno)


    fig.update_layout(
        title='durée moyenne du marché de développement d\'un jeu sur une plateforme',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title=dict(
                text="Plateformes de salon",
                font=dict(
                    size=16
                    )
                ),
            tickfont_size=14,
            ),
        legend=dict(
            x=0,
            y=1,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
            ),
        xaxis=dict(
            title=dict(
                text="Nbre années",
                font=dict(
                    size=16
                    )
                ),
            tickfont_size=14,
            ),

        )
    st.plotly_chart(fig, key = uuid.uuid4())
    df_plat_year_count = df.groupby(['Type', 'Platform', 'Year']).size().reset_index(name='Count').sort_values(by='Year', ascending=False)

    df_portable_sales = df_plat_year_count.loc[df_plat_year_count['Type'] == "Portable"].sort_values(by='Year', ascending=False)

    #on va compter le nombre d'années par Platform
    df_portable_year_count = df_portable_sales.groupby(['Platform']).size().reset_index(name='Count').sort_values(by='Count', ascending=False)

    fig = px.bar(df_portable_year_count,
                y='Platform', x='Count',
                hover_data=['Platform', 'Count'],
                color='Count',
                labels={'Platform'}, height=800, width=800,orientation="h",
                color_continuous_scale=px.colors.sequential.Inferno)


    fig.update_layout(
        title='durée moyenne du marché de développement d\'un jeu sur une plateforme',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title=dict(
                text="Plateformes Portables",
                font=dict(
                    size=16
                    )
                ),
            tickfont_size=14,
            ),
        legend=dict(
            x=0,
            y=1,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
            ),
        xaxis=dict(
            title=dict(
                text="Nbre années",
                font=dict(
                    size=16
                    )
                ),
            tickfont_size=14,
            ),

        )
    st.plotly_chart(fig, key = uuid.uuid4())
