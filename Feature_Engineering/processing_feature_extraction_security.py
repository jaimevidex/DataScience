from pandas import read_csv, DataFrame, Series, Index, concat
from matplotlib.pyplot import figure, show, gca
from dslabs_functions import plot_multi_scatters_chart, plot_bar_chart
from sklearn.decomposition import PCA
from matplotlib.axes import Axes

data: DataFrame = read_csv("Data/class_ny_arrests_scaled_zscore.csv", index_col="ARREST_KEY")
target = "CLASS"
var1 = "PD_CD"
var2 = "Longitude"

figure()
plot_multi_scatters_chart(data, var1, var2, target)
show()

target_data: Series = data.pop(target)
index: Index = data.index
pca = PCA()
pca.fit(data)

xvalues: list[str] = [f"PC{i+1}" for i in range(len(pca.components_))]
figure()
ax: Axes = gca()
plot_bar_chart(
    xvalues,
    pca.explained_variance_ratio_,
    ax=ax,
    title="Explained variance ratio",
    xlabel="PC",
    ylabel="ratio",
    percentage=True,
)
ax.plot(pca.explained_variance_ratio_)
show()

transf: PCA = pca.transform(data)
data_trsf = DataFrame(transf, columns=xvalues, index=index)
data_trsf: DataFrame = concat([data_trsf, target_data], axis=1)

figure()
plot_multi_scatters_chart(data_trsf, "PC1", "PC2", target)
show()

