import streamlit as st
from plotly import graph_objects as go
from predictors import predict_ddg_dynamut
from collections import defaultdict

def group_features_by_ontology(features):
    """
    Groups UniProt features into categories: domain, site, bond, other.
    """
    categories = defaultdict(list)

    for feat in features:
        ftype = feat["type"].lower()
        start = feat["location"]["start"]["value"]
        end   = feat["location"]["end"]["value"]
        desc  = feat.get("description", "").strip()

        entry = {
            "label": desc or f"{ftype.title()} {start}-{end}",
            "start": start,
            "end": end,
        }

        # Group based on feature type logic
        if "domain" in ftype or "region" in ftype or "repeat" in ftype or "motif" in ftype:
            categories["domain"].append(entry)
        elif "site" in ftype or "binding" in ftype or "metal" in ftype:
            categories["site"].append(entry)
        elif "bond" in ftype or "disulfide" in ftype or "cross-link" in ftype:
            categories["bond"].append(entry)
        else:
            categories["other"].append(entry)

    return categories



def plot_domains(features: list[dict], seq_length: int) -> go.Figure:
    grouped = group_features_by_ontology(features)
    fig = go.Figure()

    # Better colors and y-labels
    category_settings = {
        "domain": {"color": "#4F81BD", "y": 0.8, "label": "Domain/Region"},
        "site":   {"color": "#C0504D", "y": 0.6, "label": "Sites (Active, Binding, etc)"},
        "bond":   {"color": "#9BBB59", "y": 0.4, "label": "Bonds"},
        "other":  {"color": "#7F7F7F", "y": 0.2, "label": "Other Features"}
    }

    for category, feats in grouped.items():
        for feat in feats:
            x0 = feat["start"]
            x1 = feat["end"]
            y = category_settings[category]["y"]
            label = feat["label"]
            color = category_settings[category]["color"]

            if category == "bond":
                # Show as a single horizontal line
                fig.add_trace(go.Scatter(
                    x=[x0, x1],
                    y=[y, y],
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(color=color, size=8),
                    hovertemplate=f"{label}<br>{x0} - {x1}",
                    name=category_settings[category]["label"],
                    showlegend=False
                ))
            else:
                # Show as a rectangle (domain, site, other)
                fig.add_trace(go.Bar(
                    x=[max(1, x1 - x0)],
                    y=[y],
                    base=x0,
                    width=0.1,
                    orientation="h",
                    name=category_settings[category]["label"],
                    marker=dict(color=color),
                    hovertemplate=f"{label}<br>{x0} - {x1}",
                    showlegend=False
                ))

    fig.update_layout(
        title="🧬 Protein Domain and Feature Map (UniProt)",
        xaxis_title="Amino Acid Position",
        yaxis=dict(
            tickvals=[v["y"] for v in category_settings.values()],
            ticktext=[v["label"] for v in category_settings.values()],
            range=[0, 1],
            showgrid=False
        ),
        xaxis=dict(range=[0, seq_length], showgrid=True),
        plot_bgcolor="#FAFAFA",
        height=350,
        margin=dict(t=40, l=40, r=20, b=40),
        showlegend=False
    )
    return fig


def plot_conservation(scores: list[float]) -> go.Figure:
    """
    Plots per-residue conservation as a line chart.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(scores) + 1)),
        y=scores,
        mode='lines+markers',
        name='Conservation Score'
    ))
    fig.update_layout(
        xaxis_title='Residue Number',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1])
    )
    return fig


def show_mutation_form():
    """
    Renders a form for users to input a mutation and see DynaMut predictions.
    """
    with st.form('mutate_form'):
        site = st.text_input('Residue (e.g. A123)')
        mutation = st.text_input('Mutation (e.g. A123C)')
        submitted = st.form_submit_button('Predict ΔΔG')
        if submitted:
            try:
                chain = site[0]
                resnum = int(site[1:])
                result = predict_ddg_dynamut('temp.pdb', chain, resnum, mutation)
                st.success(f"Predicted ΔΔG: {result.get('ddg')} kCal/mol")
            except Exception as e:
                st.error(f"Error: {e}")