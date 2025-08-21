# Databricks notebook source
pip install shap

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import shap
import mlflow
import numpy as np

from scipy.stats import probplot
from itertools import product

# COMMAND ----------

def _generate_residuals_plot_figure(
    actual: pd.Series, 
    predicted: pd.Series, 
    color = '#007ba7', 
    alpha = 0.75,
    figsize = (10, 8),
    width_ratios = [5, 1.5, 1.5],
    scilimits=(4,4)
    ):
    """This method generate a residuals plot figure, with a plot of the prediction error, a histogram and a Q-Q plot. Heavily inspired by code from yellowbrick. (https://www.scikit-yb.org/en/latest/_modules/yellowbrick/regressor/residuals.html#ResidualsPlot)

    Args:
        actual (pd.Series): y true values 
        predicted (pd.Series): y predict values
        color (str, optional): color used in figure
        alpha (float, optional): transparency in figure. Defaults to 0.75.

    Returns:
        Figure: generated figure
    """    
    residuals = actual - predicted

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize,gridspec_kw={'width_ratios': width_ratios})
    fig.suptitle('Residuals Plot', fontsize=12)

    # Draw the residuals scatter plot
    ax1.scatter(predicted, residuals, c=color, alpha=alpha)
    ax1.ticklabel_format(axis='both', style='sci', scilimits=scilimits)
    ax1.set_title('Prediction Error', fontsize=10)
    ax1.set_xlabel('Predicted Value', fontsize=8)
    ax1.set_ylabel('Residuals', fontsize=8)
    ax1.axhline(y = 0, color = 'black', linestyle = '--')
    ax1.grid()
    
    # Add residuals histogram
    ax2.hist(residuals, bins=50, orientation="horizontal", color=color)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_title('Histogram', fontsize=10)
    ax2.set_xlabel('Distribution', fontsize=8)
    ax2.axhline(y = 0, color = 'black', linestyle = '--')
    ax2.grid()
    
    # Add residuals qqplot
    osm, osr = probplot(residuals, dist="norm", fit=False)
    ax3.scatter(osm, osr, c=color, alpha=alpha)
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.set_title('Q-Q plot', fontsize=10)
    ax3.set_xlabel('Theoretical Quantiles', fontsize=8)
    ax3.axhline(y = 0, color = 'black', linestyle = '--')
    ax3.grid()
    
    return fig

# COMMAND ----------

#------------------------ -----------------------------------SHAP Values Plot-------------------------------------------------------------------------#

# COMMAND ----------

def _generate_dependence_plot(
    shap_values_models: list[shap.Explanation],
    model_names: list[str],
    features: list[str] | None = None,
    figsize=(16,6),
    name_tag: str = "",
):
    """
    Create a SHAP dependence scatter plot, with the value of the feature on the x-axis and the SHAP value of the same feature on the y-axis. This plot shows how the model depends on the given feature.
        Inputs:
        - shap_values_models: list of shap_values for one model or more of models
        - model_names: list of names of one model or more models
        - features: a list of features to display the Shap dependence plot or None, default = all features
        - figsize: tuple with the figure size, default = (16,6)
        - name_tag: a description or tag to add to the name of the plot, default = '"" 
        Outputs:
        - Display a figure with the SHAP dependence plot of features for one or more model and log it as a mlflow artifact.
    """    
    if features is None:
        feature_list = shap_values_models[0].feature_names
    else:
        feature_list = features
    n_cols = len(feature_list)
    n_rows = len(shap_values_models)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize, squeeze=False)
    plot_name = name_tag
    for shap_values, name, row in zip(shap_values_models, model_names, range(n_rows)):
        plot_name = plot_name + '_' + name
        for feature, col in zip(feature_list, range(n_cols)):
            shap.plots.scatter(shap_values[:,feature], ax=axes[row, col], show=False, title=name.capitalize())
    plt.tight_layout()
    plt.show()
    file_name = f"shap_dependence_plot_model{plot_name.lower()}.png"
    mlflow.log_figure(fig, file_name)
    print(f"Displaying: {file_name}")

# COMMAND ----------

def _generate_interaction_plot(
    shap_values_models: list[shap.Explanation],
    model_names: list[str],
    features: list[str] | None = None,
    figsize=(16,6),
    name_tag: str = "",
):
    """
    Create a Shap Interaction plot to reveal how the interactions of two features impact on model predictions.
        Inputs:        
        - shap_values_models: list of shap_values for one model or more of models
        - model_names: list of names of one model or more models
        - features: list of features to display the Shap interaction plot or None, default = all features
        - figsize: tuple with the figure size, default = (16,6)
        - name_tag: a description or tag to add to the name of the plot, default = '""
        Outputs:
        - Display a figure with the SHAP interaction plot of two features for one or more model and log it as a mlflow artifact.
    """
    if features is None:
        feature_list = shap_values_models[0].feature_names
    else:
        feature_list = features
    n_cols = len(feature_list)
    n_rows = len(shap_values_models)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize, squeeze=False)
    plot_name = name_tag
    for shap_values, name, row in zip(shap_values_models, model_names, range(n_rows)):
        plot_name = plot_name + "_" + name
        for feature, col in zip(feature_list, range(n_cols)):
            shap.plots.scatter(shap_values[:,feature], ax=axes[row, col], show=False, title=name.capitalize(), color=shap_values)
    plt.tight_layout()
    plt.show()
    file_name = f"shap_interaction_plot_model_{plot_name.lower()}.png"
    mlflow.log_figure(fig, file_name)
    print(f"Displaying: {file_name}")

# COMMAND ----------

def _generate_waterfall_plot(    
    shap_values_models: list[shap.Explanation],
    model_names: list[str],
    instances: list[int],
    figsize=(20,8),
    name_tag: str = "",
):
    """
    Create a Shap waterfall plot, which s how the SHAP values (evidence) of each feature move the model output from the mean predicted value to the final model prediction to this instance given the evidence of all the features.
        Inputs:        
        - shap_values_models: list of shap_values for one model or more of models
        - model_names: list of names of one model or more models
        - instances: a list of instances to analize in the waterfall plot
        - figsize: size of the figure, default = (20,8)
        - name_tag: a description or tag to add to the name of the plot, default = '""
        Outputs:
        - Display a figure with the SHAP waterfall plot of each instance and log it as a mlflow artifact.
    """
    n_cols = len(shap_values_models)
    n_rows = len(instances)
    model_instance_combinations = list(product(shap_values_models, instances))
    name_instance_combinations = list(product(model_names, instances))
    models = ''
    point_names = name_tag
    fig = plt.figure()
    for n_plot, (combination_shap, combination_name) in enumerate(zip(model_instance_combinations, name_instance_combinations), start=1):
        plot = (n_rows*100)+(n_cols*10)+(n_plot)
        ax = fig.add_subplot(plot)
        shap_values = combination_shap[0]
        instance = combination_shap[1]
        name = combination_name[0]
        shap.plots.waterfall(shap_values[instance], show=False)
        ax.set_title(f'{name.capitalize()} Model, Instance {instance}')
        ax.set_ylabel(f'Feature Name and Value')
        ax.set_xlabel(f'Contribution from mean(E[f(X)]) to actual(f(x)) prediction')
    for name in model_names:
        models = models + '_' + name
    for point in instances:
        point_names = point_names + f'{point}_'
    plt.gcf().set_size_inches(figsize[0],figsize[1])
    plt.tight_layout()
    plt.show()
    file_name = f"shap_waterfall_plot{models.lower()}_{point_names}.png"
    mlflow.log_figure(fig, file_name)
    print(f"Displaying: {file_name}")

# COMMAND ----------

def _generate_global_importance_plot(    
    shap_values_models: list[shap.Explanation],
    model_names: list[str],
    figsize=(20,8),
    clustering: bool=False,
    X_train: pd.DataFrame | None=None,
    y_train: pd.Series | pd.DataFrame | None=None,
    clustering_cutoff:float=0.5,
    name_tag: str = "",
):
    """
    Create a Shap global importance plot for one or more models based on the average of the absolute SHAP values per feature accross the data.
        Inputs:        
        - shap_values_models: list of shap_values for one model or more of models
        - model_names: list of names of one model or more models
        - figsize: size of the figure, default = (20,8)
        - clustering: bool, default = False, if True, enable the feature clustering functionality, which allows identifying highly correlated variables
        - clustering_cutoff: the cutoff value for clustering, default = 0.5
        - name_tag: a tag to add to the name of the plot, default = ''
        Outputs:
        - Display a figure with the SHAP global importance plot of each instance and log it as a mlflow artifact.
    """
    if clustering is True:
        clustering = shap.utils.hclust(X_train, y_train)
    else:
        clustering=None
    n_cols = len(shap_values_models)
    n_rows = 1
    plot_name = name_tag
    fig = plt.figure()
    for n_plot, (shap_values, name) in enumerate(zip(shap_values_models, model_names), start=1):
        plot = (n_rows*100)+(n_cols*10)+(n_plot)
        ax = fig.add_subplot(plot)
        shap.plots.bar(shap_values, clustering=clustering, clustering_cutoff=clustering_cutoff, show=False)
        ax.set_title(f'{name.capitalize()} Model Global Importance')
        ax.set_ylabel(f'Features', fontsize=14)
        plot_name = plot_name + '_' + name
    plt.gcf().set_size_inches(figsize[0],figsize[1])
    plt.tight_layout()
    plt.show()
    file_name = f"shap_global_importance_{plot_name.lower()}_plot.png"
    mlflow.log_figure(fig, file_name)
    print(f"Displaying: {file_name}")

# COMMAND ----------

def _generate_summary_plot(    
    shap_values_models: list[shap.Explanation],
    model_names: list[str],
    figsize=(20,8),
    name_tag: str = "",
):
    """
    Create a Shap summary plot (also called a beeswarm plot), which graphically represents the SHAP values accross all features and numerous data points. 
        Inputs:        
        - shap_values_models: list of shap_values for one model or more of models
        - model_names: list of names of one model or more models
        - figsize: size of the figure, default = (20,8).
        - name_tag: a description or tag to add to the name of the plot, default = '""
        Outputs:
        - Display a figure with the SHAP summary plot and log it as a mlflow artifact.
    """
    n_cols = len(shap_values_models)
    n_rows = 1
    plot_name = name_tag
    models = ''
    fig = plt.figure()
    for n_plot, (shap_values, name) in enumerate(zip(shap_values_models, model_names), start=1):
        plot = (n_rows*100)+(n_cols*10)+(n_plot)
        ax = fig.add_subplot(plot)
        shap.plots.beeswarm(shap_values, show=False)
        ax.set_title(f'{name.capitalize()} Model Global Importance')
        ax.set_ylabel(f'Features', fontsize=14)
        plot_name = plot_name + '_' + name
    plt.gcf().set_size_inches(figsize[0],figsize[1])
    plt.tight_layout()
    plt.show()
    file_name = f"shap_beeswarm_plot_{plot_name.lower()}.png"
    mlflow.log_figure(fig, file_name)
    print(f"Displaying: {file_name}")

# COMMAND ----------

def _generate_feature_max_min_plot(
    X_test: pd.DataFrame, 
    features: list[str] | None = None, 
    figsize=(20,3),
    name_tag: str = "",
):
    """
    Create a plot with colorbars for each feature, which shows the min and max values of the feature using the default colormap of the shap summary plot.
        Inputs:        
        - X_test: a pandas DataFrame of the test data
        - features: list of features to display in the plot, default = all X_test features
        - figsize: size of the figure, default = (20,3)
        - name_tag: a tag to add to the name of the plot, default = ''
        Outputs:
        - Display a figure with colobars of max and min of each feature and log it as a mlflow artifact.
    """
    if features is None:
        feature_list = X_test.columns
    else:
        feature_list = features
    cmap = shap.plots.colors.red_blue
    # Create the figure and axis (reduced figsize)
    fig, ax = plt.subplots(figsize=figsize)  # Reduced the figure size
    # Remove the frame of the plot
    ax.axis('off')
    for i, feature in enumerate(feature_list):
        # Create ScalarMappable for this feature's range
        vmin = X_test[feature].min()
        vmax = X_test[feature].max()
        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        # Create a vertical colorbar for this feature
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        # Set the colorbar label to the feature name and set it to the left
        cbar.set_label(feature, labelpad=1)
        cbar.ax.yaxis.set_label_position('left')  # Move label to the left
        # Adjust position of the vertical colorbars
        cbar.ax.set_position([0.1 + i * 0.1, 0.18, 0.03, 0.7])  # Adjust horizontal placement for each colorbar
    plt.show()
    file_name = f"feature_max_min_plot_{name_tag.lower()}.png"
    mlflow.log_figure(fig, file_name)
    print(f"Displaying: {file_name}")

# COMMAND ----------

def _generate_individual_scatter_plot( 
    shap_values: shap.Explanation,
    feature: str,
    plot_title: str,
    color: str| shap.Explanation = '#1E88E5',
    x_lim: tuple[float, float] = None,
    y_lim: tuple[float, float] = None,
    figsize=(20,8),
    ):

    """
    Create a Shap individual scatter plot, which shows the SHAP values for a single feature. This plot allows the edition of of x-axis and y-axis to zoom in and out.
    If the color argument is a string, the graph is a dependence plot, if the color argument a shap.Explanation, then this plot is a feature interaction plot.
        Inputs:
            - shap_values_models: shap_values for one model
            - feature: feature to be exhibited in the plot
            - plot_title: title of the plot
            - color: a string of the color of the plot or a shap.Explanation object to reveal the interaction of the first feature with other feature
            - x_lim: x-axis limits of the plot
            - y_lim: y-axis limits of the plot
            - figsize: size of the figure, default = (20,8)
        Outputs:
            - Display a figure with the SHAP summary plot and log the figure as an artifact
    """
    shap.plots.scatter(shap_values[:,feature], show=False, color=color)
    plt.title(plot_title)
    if x_lim is not None:
        plt.xlim(x_lim[0], x_lim[1])
    else:
        plt.tight_layout()
    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1])
    else:
        plt.tight_layout()
    plot_name = plot_title.replace(" ", "_")
    if isinstance(color, shap.Explanation):
        file_name = f"shap_interaction_plot_{plot_name.lower()}.png"
    else:
        file_name = f"shap_dependence_plot_{plot_name.lower()}.png"
    fig = plt.gcf()
    plt.show()
    mlflow.log_figure(fig, file_name)
    print(f"Displaying: {file_name}")

