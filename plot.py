import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import colorcet as cc


names = ['Prudhoe', 'Canning', 'Dalton', 'Kaktovik']
labels = ['(a)', '(b)', '(c)', '(d)']

dfs = [pd.read_csv(
    f'/Users/rbiessel/Documents/DIP_Project/output/{name}.csv') for name in names]

cmap = mpl.cm.get_cmap("cet_glasbey_hv")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
for i in range(len(names)):

    sns.kdeplot(x=dfs[i]['eccentricity'], weights=dfs[i]['area'], label=names[i],
                clip=(0, 1), color=cmap(int(i * 100/2)), bw_adjust=.4, linewidth=3, alpha=0.8, ax=ax)

    # Show the plot

ax.legend(loc='best', fancybox=True, framealpha=0.1)
ax.set_xlabel('Eccentricity')
ax.set_xlim([0, 1])
ax.set_title('(a)', loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.tight_layout()
plt.savefig('/Users/rbiessel/Documents/DIP_Project/figures/eccentricity.png',
            dpi=300, transparent=True)
plt.show()


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
maxarea = 175000 * 1e-6
for i in range(len(names)):

    sns.kdeplot(dfs[i]['area'] * 100 * 1e-6, label=names[i], color=cmap(int(i * 100/2)),
                bw_adjust=.4, linewidth=3, alpha=0.8, ax=ax, clip=(0, maxarea))

    total_area = np.sum(dfs[i]['area']) * 100 * 1e-6
    print(f'Area, {names[i]}: {total_area} km squared')

# ax.legend(loc='best')
ax.set_xlabel('Area (square km)')
ax.set_xlim([0, maxarea])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('(b)', loc='left')
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.tight_layout()
plt.savefig('/Users/rbiessel/Documents/DIP_Project/figures/areas.png',
            dpi=300, transparent=True)
plt.show()


for i in range(len(names)):

    bins = int(360/8)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 5))

    lakes = dfs[i]
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # Add pi/4 to everything to correct it
    lakes['orientation'] = lakes['orientation'] + np.pi/4

    hist_kwargs = {
        'bins': bins,
        'weights': lakes['area'] * 100 * 1e-6,
        'color': 'maroon',
        'alpha': 0.9,
        'density': False
    }

    ax.hist(lakes['orientation'], **hist_kwargs)
    ax.hist(np.pi + lakes['orientation'], **hist_kwargs)
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

    # ax.plot(lakes['orientation'], n)
    # ax.set_rmax(10)
    # ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    # ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    ax.set_title(f'{labels[i]} {names[i]}', loc='left', fontsize=12)
    plt.tight_layout(w_pad=-1, h_pad=2.0)

    plt.savefig(
        f'/Users/rbiessel/Documents/DIP_Project/figures/{names[i]}.rose.png', transparent=True, dpi=300)
    # plt.show()
