import matplotlib.pyplot as plt
import pandas as pd


def plot_offer_utilization(df_offer):
    # distribution of offer utilization 
    offer_use = df_offer.groupby(['person', 'is_offer_used']).count()['offer_id'].unstack().reset_index().fillna(0)
    offer_use.columns = ['person', 'not_used', 'used']
    offer_use['utilization'] = offer_use['used'] / (offer_use['not_used'] + offer_use['used'])
    offer_use['utilization'].hist(bins=60)
    plt.xlabel('Utilization')
    plt.ylabel('Count')
    plt.title('Offer Utilization Distribution')
    plt.show()

def plot_offer_utilization_by_group(df_offer, df_profile, group):
    offer_use = df_offer.groupby(['person', 'is_offer_used']).count()['offer_id'].unstack().reset_index().fillna(0)
    offer_use.columns = ['person', 'not_used', 'used']
    offer_use['utilization'] = offer_use['used'] / (offer_use['not_used'] + offer_use['used'])

    merged = offer_use.merge(df_profile, left_on='person', right_on='id')
    merged = merged[[group, 'utilization']]
    
    vals, labels = [], []
    print(merged[group].unique())
    for g in np.sort(merged[group].unique()):
        vals.append(np.array(merged.loc[merged[group] == g, 'utilization']))
        labels.append(g)
    plt.hist(vals, alpha=0.5, density=False, histtype='bar', stacked=True, label=labels, bins=20)

    plt.xlabel('Utilization')
    plt.ylabel('Count')
    plt.title(f'Offer Utilization by {group} Distribution')
    plt.legend()
    plt.show()
