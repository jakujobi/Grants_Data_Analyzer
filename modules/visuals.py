import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any

# SDSU brand colors
SDSU_BLUE = "#0033A0"
SDSU_YELLOW = "#FFD100"
SDSU_WHITE = "#FFFFFF"

def set_sdsu_style():
    """Set the SDSU brand style for plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set SDSU colors
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[SDSU_BLUE, SDSU_YELLOW, '#4A90E2', '#F5A623', '#7ED321'])
    
    # Set font sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    # Set figure size
    plt.rcParams['figure.figsize'] = [12, 8]

def create_bar_chart(data: pd.DataFrame, x_col: str, y_col: str, title: str, 
                    x_label: Optional[str] = None, y_label: Optional[str] = None,
                    top_n: Optional[int] = None) -> plt.Figure:
    """
    Create a bar chart with SDSU styling.
    
    Args:
        data: DataFrame containing the data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Chart title
        x_label: Optional x-axis label
        y_label: Optional y-axis label
        top_n: Optional number of top items to show
        
    Returns:
        Matplotlib figure
    """
    # Validate input data
    if data.empty:
        print("Warning: Empty DataFrame provided")
        # Create empty figure
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig
    
    if x_col not in data.columns or y_col not in data.columns:
        print(f"Warning: Required columns missing. Available columns: {list(data.columns)}")
        # Create empty figure
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Missing required columns', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig
    
    # Set SDSU style
    set_sdsu_style()
    
    # Create figure
    fig, ax = plt.subplots()
    
    try:
        # Sort data by y_col in descending order
        sorted_data = data.sort_values(y_col, ascending=False)
        
        # Limit to top N if specified
        if top_n is not None:
            sorted_data = sorted_data.head(top_n)
        
        # Create bar chart
        bars = ax.bar(sorted_data[x_col], sorted_data[y_col], color=SDSU_BLUE)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel(x_label or x_col)
        ax.set_ylabel(y_label or y_col)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        ax.text(0.5, 0.5, 'Error creating visualization', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
    
    return fig

def create_pie_chart(data: pd.DataFrame, values_col: str, labels_col: str, title: str,
                    top_n: Optional[int] = None) -> plt.Figure:
    """
    Create a pie chart with SDSU styling.
    
    Args:
        data: DataFrame containing the data
        values_col: Column name for values
        labels_col: Column name for labels
        title: Chart title
        top_n: Optional number of top items to show
        
    Returns:
        Matplotlib figure
    """
    # Validate input data
    if data.empty:
        print("Warning: Empty DataFrame provided")
        # Create empty figure
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig
    
    if values_col not in data.columns or labels_col not in data.columns:
        print(f"Warning: Required columns missing. Available columns: {list(data.columns)}")
        # Create empty figure
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Missing required columns', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig
    
    # Set SDSU style
    set_sdsu_style()
    
    # Create figure
    fig, ax = plt.subplots()
    
    try:
        # Sort data by values_col in descending order
        sorted_data = data.sort_values(values_col, ascending=False)
        
        # Limit to top N if specified
        if top_n is not None:
            sorted_data = sorted_data.head(top_n)
            
            # Add "Others" category if there are more items
            if len(data) > top_n:
                others_value = data.iloc[top_n:][values_col].sum()
                others_row = pd.DataFrame({
                    values_col: [others_value],
                    labels_col: ['Others']
                })
                sorted_data = pd.concat([sorted_data, others_row])
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sorted_data[values_col],
            labels=sorted_data[labels_col],
            autopct='%1.1f%%',
            startangle=90,
            colors=[SDSU_BLUE, SDSU_YELLOW, '#4A90E2', '#F5A623', '#7ED321']
        )
        
        # Set title
        ax.set_title(title)
        
        # Adjust layout
        plt.tight_layout()
    except Exception as e:
        print(f"Error creating pie chart: {e}")
        ax.text(0.5, 0.5, 'Error creating visualization', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
    
    return fig

def create_line_chart(data: pd.DataFrame, x_col: str, y_col: str, title: str,
                     x_label: Optional[str] = None, y_label: Optional[str] = None) -> plt.Figure:
    """
    Create a line chart with SDSU styling.
    
    Args:
        data: DataFrame containing the data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Chart title
        x_label: Optional x-axis label
        y_label: Optional y-axis label
        
    Returns:
        Matplotlib figure
    """
    # Validate input data
    if data.empty:
        print("Warning: Empty DataFrame provided")
        # Create empty figure
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig
    
    if x_col not in data.columns or y_col not in data.columns:
        print(f"Warning: Required columns missing. Available columns: {list(data.columns)}")
        # Create empty figure
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Missing required columns', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig
    
    # Set SDSU style
    set_sdsu_style()
    
    # Create figure
    fig, ax = plt.subplots()
    
    try:
        # Sort data by x_col
        sorted_data = data.sort_values(x_col)
        
        # Create line chart
        ax.plot(sorted_data[x_col], sorted_data[y_col], marker='o', color=SDSU_BLUE, linewidth=2)
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel(x_label or x_col)
        ax.set_ylabel(y_label or y_col)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
    except Exception as e:
        print(f"Error creating line chart: {e}")
        ax.text(0.5, 0.5, 'Error creating visualization', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
    
    return fig

def create_summary_dashboard(summary_counts: Dict[str, int]) -> plt.Figure:
    """
    Create a summary dashboard with key metrics.
    
    Args:
        summary_counts: Dictionary with summary counts
        
    Returns:
        Matplotlib figure
    """
    # Validate input data
    if not summary_counts:
        print("Warning: Empty summary counts provided")
        # Create empty figure
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig
    
    # Set SDSU style
    set_sdsu_style()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    try:
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Create summary cards
        for i, (key, value) in enumerate(summary_counts.items()):
            ax = axes[i]
            
            # Remove axes
            ax.axis('off')
            
            # Create a text box
            ax.text(0.5, 0.5, str(value), 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=24, fontweight='bold',
                    color=SDSU_BLUE)
            
            # Add label
            label = key.replace('_', ' ').title()
            ax.text(0.5, 0.2, label, 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
        
        # Hide unused subplots
        for i in range(len(summary_counts), len(axes)):
            axes[i].axis('off')
        
        # Set title
        fig.suptitle('SDSU Research Summary', fontsize=20, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    except Exception as e:
        print(f"Error creating summary dashboard: {e}")
        for ax in axes:
            ax.axis('off')
        axes[0].text(0.5, 0.5, 'Error creating visualization', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[0].transAxes)
    
    return fig

def save_figure(fig: plt.Figure, filename: str, dpi: int = 300) -> None:
    """
    Save a figure to a file.
    
    Args:
        fig: Matplotlib figure
        filename: Output filename
        dpi: DPI for the output image
    """
    try:
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving figure: {e}")
    finally:
        plt.close(fig) 