import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_offer_utilization(df_offer):
    """
    Make a plot for distribution of offer utilization.

    Parameters
    ----------
    df_offer: pandas.DataFrame
        The data set of offer.

    Returns
    -------
    None
    """

    offer_use = df_offer.groupby(['person', 'is_offer_used']).count()['offer_id'].unstack().reset_index().fillna(0)
    offer_use.columns = ['person', 'not_used', 'used']
    offer_use['utilization'] = offer_use['used'] / (offer_use['not_used'] + offer_use['used'])
    offer_use['utilization'].hist(bins=60)
    plt.xlabel('Utilization')
    plt.ylabel('Count')
    plt.title('Offer Utilization Distribution')
    plt.show()


def plot_offer_utilization_by_group(df_offer, df_profile, group):
    """
    Make a plot for distribution of offer utilization by group.

    Parameters
    ----------
    df_offer: pandas.DataFrame
        The data set of offer.
    df_profile: pandas.DataFrame
        The data set of user profile.
    group: str
        Grouping variable.

    Returns
    -------
    None
    """

    offer_use = df_offer.groupby(['person', 'is_offer_used']).count()['offer_id'].unstack().reset_index().fillna(0)
    offer_use.columns = ['person', 'not_used', 'used']
    offer_use['utilization'] = offer_use['used'] / (offer_use['not_used'] + offer_use['used'])

    merged = offer_use.merge(df_profile, left_on='person', right_on='id')
    merged = merged[[group, 'utilization']]

    vals, labels = [], []
    for g in np.sort(merged[group].unique()):
        vals.append(list(merged.loc[merged[group] == g, 'utilization']))
        labels.append(g)
    plt.hist(vals, alpha=0.5, density=False, histtype='bar', stacked=True, label=labels, bins=20)

    plt.xlabel('Utilization')
    plt.ylabel('Count')
    plt.title(f'Offer Utilization Distribution by {group}')
    plt.legend()
    plt.show()


def plot_funnel(df_offer_time, stages=['time_received', 'time_viewed', 'time_completed']):
    """
    Make a funnel plot for the offer utilization.

    Parameters
    ----------
    df_offer_time: pandas.DataFrame
        The data set of offer time.
    stages: list
        Stages of offer use.

    Returns
    -------
    None
    """

    from plotly import graph_objects as go

    vals = []
    for s in stages:
        vals.append(df_offer_time[s].notnull().sum())
    fig = go.Figure(go.Funnel(
    y = [s.split('_')[1] for s in stages],
    x = vals,
    textinfo = "value+percent initial"))

    fig.update_layout(
        title={
            'text': "Offer",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'}
    )

    fig.show()


def plot_funnel_by_group(df_offer_time, df_profile, group, stages=['time_received', 'time_viewed', 'time_completed']):
    """
    Make a funnel plot for the offer utilization by group.

    Parameters
    ----------
    df_offer_time: pandas.DataFrame
        The data set of offer time.
    df_profile: pandas.DataFrame
        The data set of user profile.
    group: str
        Grouping variable.
    stages: list
        Stages of offer use.

    Returns
    -------
    None
    """

    from plotly import graph_objects as go

    fig = go.Figure()
    merged = df_offer_time.merge(df_profile, left_on='person', right_on='id')
    merged = merged[[group] + stages]
    
    for g in np.sort(merged[group].unique()):
        vals = []
        for s in stages:
            vals.append(merged.loc[merged[group] == g, s].notnull().sum())
        fig.add_trace(go.Funnel(
            name = str(g),
            orientation = "h",
            y = [s.split('_')[1] for s in stages],
            x = vals,
            textinfo = "value+percent initial"))
    
    fig.update_layout(
        title={
            'text': f"Offer by {group}",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'}
    )

    fig.show()


def plot_average_amount(df_transaction, how):
    """
    Make a plot for transaction amount.

    Parameters
    ----------
    df_transaction: pandas.DataFrame
        The data set of transaction.
    how: str
        The method to aggregate the amount. The options are average and sum.

    Returns
    -------
    None
    """

    df_transaction['offer_use'] = np.where(df_transaction['offer_id'] == '', 'Not used', 'Used')
    if how == 'average':
        df_transaction.groupby('offer_use')['amount'].mean().plot(kind='bar')
    elif how == 'sum':
        df_transaction.groupby('offer_use')['amount'].sum().plot(kind='bar')
    plt.ylabel('Amount')
    plt.title(f'The {how} of amount by offer use')
    plt.show()


def plot_feature_importance(rf, feature_names):
    """
    Make a plot for feature importance.

    Parameters
    ----------
    rf: pandas.classifier
        The trained Random Forest model.
    feature_names: list
        Features to be included for calculation.

    Returns
    -------
    None
    """

    pd.Series(rf.best_estimator_[1].feature_importances_,
              index=feature_names).nlargest(10)\
                  .plot(kind='barh', fontsize=18)
