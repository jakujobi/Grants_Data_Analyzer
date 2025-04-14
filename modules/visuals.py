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

def create_co_pi_analysis_chart(yearly_data: pd.DataFrame, metric: str = 'matching_projects') -> plt.Figure:
    """
    Create a visualization for Co-PI analysis showing yearly trends.
    
    Args:
        yearly_data: DataFrame with yearly statistics from get_projects_by_co_pi_count_yearly
        metric: Which metric to visualize ('matching_projects', 'matching_percentage', etc.)
        
    Returns:
        Matplotlib Figure object
    """
    if yearly_data.empty:
        # Create an empty figure with a message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by fiscal year
    data = yearly_data.sort_values('fiscal_year')
    
    # Set up labels and titles based on the metric
    if metric == 'matching_projects':
        y_label = 'Number of Projects'
        title = 'Projects Matching Co-PI Criteria by Year'
        color = 'steelblue'
    elif metric == 'matching_percentage':
        y_label = 'Percentage (%)'
        title = 'Percentage of Projects Matching Co-PI Criteria by Year'
        color = 'forestgreen'
    elif metric == 'total_co_pis':
        y_label = 'Number of Co-PIs'
        title = 'Total Co-PIs by Year'
        color = 'darkorange'
    else:
        y_label = 'Count'
        title = f'{metric.replace("_", " ").title()} by Year'
        color = 'steelblue'
    
    # Create bar chart
    bars = ax.bar(data['fiscal_year'].astype(str), data[metric], color=color, alpha=0.8)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}' if metric == 'matching_percentage' else f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Fiscal Year', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Customize x-axis labels
    plt.xticks(rotation=45)
    
    # Add totals as a text annotation
    total_projects = data['total_projects'].sum()
    total_matches = data['matching_projects'].sum()
    overall_percentage = (total_matches / total_projects * 100) if total_projects > 0 else 0
    
    ax.text(0.02, 0.97, 
            f"Total Projects: {total_projects}\n"
            f"Total Matching: {total_matches}\n"
            f"Overall: {overall_percentage:.1f}%",
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_co_pi_comparison_chart(yearly_data: pd.DataFrame) -> plt.Figure:
    """
    Create a comparison chart showing projects matching criteria vs total projects.
    
    Args:
        yearly_data: DataFrame with yearly statistics from get_projects_by_co_pi_count_yearly
        
    Returns:
        Matplotlib Figure object
    """
    if yearly_data.empty:
        # Create an empty figure with a message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by fiscal year
    data = yearly_data.sort_values('fiscal_year')
    
    # Set up x positions
    years = data['fiscal_year'].astype(str)
    x = np.arange(len(years))
    width = 0.35
    
    # Create grouped bar chart
    matched_bars = ax.bar(x - width/2, data['matching_projects'], width, label='Matching Projects', color='steelblue')
    total_bars = ax.bar(x + width/2, data['total_projects'], width, label='Total Projects', color='lightgray')
    
    # Add value labels on top of bars
    for bars in [matched_bars, total_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
    
    # Add percentage labels on matched bars
    for i, (bar, percentage) in enumerate(zip(matched_bars, data['matching_percentage'])):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                f'{percentage:.1f}%',
                ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('Fiscal Year', fontsize=12)
    ax.set_ylabel('Number of Projects', fontsize=12)
    ax.set_title('Matching Projects vs Total Projects by Year', fontsize=14, fontweight='bold')
    
    # Set x-axis ticks
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45)
    
    # Add grid and legend
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig 

def create_multi_college_trend_chart(yearly_stats: pd.DataFrame) -> plt.Figure:
    """
    Create a chart showing the trend of multi-college projects over time.
    
    Args:
        yearly_stats: DataFrame with yearly statistics from get_multi_college_project_yearly_stats
        
    Returns:
        Matplotlib Figure object
    """
    if yearly_stats.empty:
        # Create an empty figure with a message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()
    
    # Sort by fiscal year
    data = yearly_stats.sort_values('fiscal_year')
    years = data['fiscal_year'].astype(str)
    
    # Plot count bars
    bars = ax1.bar(years, data['multi_college_projects'], color='steelblue', alpha=0.7, 
                  label='Multi-College Projects')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    # Plot percentage line
    line = ax2.plot(years, data['multi_college_percentage'], color='crimson', marker='o', 
                   linestyle='-', linewidth=2, label='% of Total Projects')
    
    # Add percentage labels
    for i, pct in enumerate(data['multi_college_percentage']):
        ax2.annotate(f'{pct:.1f}%', 
                    (i, pct),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center', 
                    fontsize=9)
    
    # Set labels and title
    ax1.set_xlabel('Fiscal Year', fontsize=12)
    ax1.set_ylabel('Number of Projects', fontsize=12, color='steelblue')
    ax2.set_ylabel('Percentage of Total Projects', fontsize=12, color='crimson')
    plt.title('Multi-College Projects Trend Over Time', fontsize=14, fontweight='bold')
    
    # Set tick parameters
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='crimson')
    plt.xticks(rotation=45)
    
    # Add grid
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add dual legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_multi_vs_single_college_comparison_chart(comparison_data: Dict[str, Any]) -> plt.Figure:
    """
    Create a chart comparing multi-college and single-college projects.
    
    Args:
        comparison_data: Dictionary with comparison metrics from get_multi_college_vs_single_college_comparison
        
    Returns:
        Matplotlib Figure object
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Extract data
    multi_count = comparison_data['multi_college_projects']
    single_count = comparison_data['single_college_projects']
    
    # Left chart: Project counts
    labels = ['Multi-College', 'Single-College']
    counts = [multi_count, single_count]
    colors = ['#1f77b4', '#ff7f0e']
    
    # Create pie chart
    ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
           wedgeprops={'edgecolor': 'w', 'linewidth': 1, 'antialiased': True})
    ax1.set_title('Project Distribution', fontsize=14, fontweight='bold')
    
    # Add count annotations
    ax1.text(0, -1.2, f"Multi-College: {multi_count:,} projects", ha='center', fontsize=11)
    ax1.text(0, -1.4, f"Single-College: {single_count:,} projects", ha='center', fontsize=11)
    
    # Right chart: Award amount comparison (if available)
    if comparison_data['avg_award_multi'] is not None and comparison_data['avg_award_single'] is not None:
        # Bar chart for award amounts
        categories = ['Average Award Amount']
        multi_award = [comparison_data['avg_award_multi']]
        single_award = [comparison_data['avg_award_single']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax2.bar(x - width/2, multi_award, width, label='Multi-College', color='#1f77b4')
        ax2.bar(x + width/2, single_award, width, label='Single-College', color='#ff7f0e')
        
        # Add value labels
        ax2.text(x - width/2, multi_award[0] + 1000, f"${multi_award[0]:,.0f}", 
                ha='center', va='bottom', fontsize=10)
        ax2.text(x + width/2, single_award[0] + 1000, f"${single_award[0]:,.0f}", 
                ha='center', va='bottom', fontsize=10)
        
        # Add difference annotation
        diff_pct = comparison_data['award_difference_percentage']
        diff_text = f"Multi-college projects receive {abs(diff_pct):.1f}% "
        diff_text += "more" if diff_pct > 0 else "less"
        ax2.text(0, -0.15, diff_text, ha='center', transform=ax2.transAxes, fontsize=11)
        
        # Set labels and title
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.set_ylabel('Average Award Amount ($)', fontsize=12)
        ax2.set_title('Funding Comparison', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', linestyle='--', alpha=0.3)
    else:
        # No award data available
        ax2.text(0.5, 0.5, "No award amount data available", ha='center', va='center', fontsize=12)
        ax2.set_axis_off()
    
    # Add overall title
    plt.suptitle('Multi-College vs. Single-College Projects', fontsize=16, fontweight='bold', y=1.05)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_college_collaboration_chart(college_metrics: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """
    Create a chart showing colleges with the most multi-college collaborations.
    
    Args:
        college_metrics: DataFrame with college metrics from get_college_collaboration_metrics
        top_n: Number of top colleges to display
        
    Returns:
        Matplotlib Figure object
    """
    if college_metrics.empty:
        # Create an empty figure with a message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Sort by multi-college projects and take top N
    top_colleges = college_metrics.sort_values('multi_college_projects', ascending=False).head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar chart
    bars = ax.barh(top_colleges['college'], top_colleges['multi_college_projects'], 
                  color='steelblue', alpha=0.8)
    
    # Add value labels
    for i, (value, pct) in enumerate(zip(top_colleges['multi_college_projects'], 
                                        top_colleges['multi_college_percentage'])):
        ax.text(value + 0.1, i, f"{int(value)} ({pct:.1f}%)", va='center', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Number of Multi-College Projects', fontsize=12)
    ax.set_title('Top Colleges by Multi-College Project Participation', fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_avg_colleges_per_project_chart(yearly_stats: pd.DataFrame) -> plt.Figure:
    """
    Create a chart showing the average number of colleges per multi-college project over time.
    
    Args:
        yearly_stats: DataFrame with yearly statistics from get_multi_college_project_yearly_stats
        
    Returns:
        Matplotlib Figure object
    """
    if yearly_stats.empty:
        # Create an empty figure with a message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Sort by fiscal year
    data = yearly_stats.sort_values('fiscal_year')
    years = data['fiscal_year'].astype(str)
    
    # Create line chart
    line = ax.plot(years, data['avg_colleges_per_multi_project'], marker='o', 
                  linestyle='-', linewidth=2, color='forestgreen')
    
    # Add value labels
    for i, value in enumerate(data['avg_colleges_per_multi_project']):
        ax.annotate(f'{value:.2f}', 
                   (years.iloc[i], value),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center', 
                   fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Fiscal Year', fontsize=12)
    ax.set_ylabel('Average Number of Colleges', fontsize=12)
    ax.set_title('Average Number of Colleges per Multi-College Project', fontsize=14, fontweight='bold')
    
    # Set tick parameters
    plt.xticks(rotation=45)
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Calculate overall average
    overall_avg = data['avg_colleges_per_multi_project'].mean()
    
    # Add horizontal line for overall average
    ax.axhline(y=overall_avg, color='r', linestyle='--', alpha=0.7)
    ax.text(years.iloc[0], overall_avg + 0.05, f'Overall Average: {overall_avg:.2f}', 
           fontsize=10, color='r')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_multi_college_dashboard(yearly_stats: pd.DataFrame, comparison_data: Dict[str, Any]) -> plt.Figure:
    """
    Create a comprehensive dashboard for multi-college project analysis.
    
    Args:
        yearly_stats: DataFrame with yearly statistics from get_multi_college_project_yearly_stats
        comparison_data: Dictionary with comparison metrics from get_multi_college_vs_single_college_comparison
        
    Returns:
        Matplotlib Figure object
    """
    if yearly_stats.empty:
        # Create an empty figure with a message
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 4)
    
    # Summary metrics at the top
    ax_summary = fig.add_subplot(gs[0, :])
    ax_summary.axis('off')
    
    # Extract summary data
    total_projects = comparison_data['total_projects']
    multi_projects = comparison_data['multi_college_projects']
    multi_pct = comparison_data['multi_college_percentage']
    avg_colleges = comparison_data['avg_colleges_multi']
    
    # Display summary metrics
    summary_text = f"""
    Total Projects: {total_projects:,}
    Multi-College Projects: {multi_projects:,} ({multi_pct:.1f}%)
    Average Colleges per Multi-College Project: {avg_colleges:.2f}
    """
    
    # Add award data if available
    if comparison_data['avg_award_multi'] is not None and comparison_data['avg_award_single'] is not None:
        diff_pct = comparison_data['award_difference_percentage']
        diff_text = "more" if diff_pct > 0 else "less"
        
        summary_text += f"""
    Average Award for Multi-College Projects: ${comparison_data['avg_award_multi']:,.2f}
    Average Award for Single-College Projects: ${comparison_data['avg_award_single']:,.2f}
    Multi-College Projects receive {abs(diff_pct):.1f}% {diff_text} funding on average
    """
    
    ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Trend chart (top left)
    ax_trend = fig.add_subplot(gs[1, :2])
    
    # Sort by fiscal year
    data = yearly_stats.sort_values('fiscal_year')
    years = data['fiscal_year'].astype(str)
    
    # Plot count bars with second y-axis for percentage
    ax_trend_twin = ax_trend.twinx()
    bars = ax_trend.bar(years, data['multi_college_projects'], color='steelblue', alpha=0.7, 
                      label='Multi-College Projects')
    line = ax_trend_twin.plot(years, data['multi_college_percentage'], color='crimson', marker='o', 
                             linestyle='-', linewidth=2, label='% of Total Projects')
    
    # Set labels
    ax_trend.set_xlabel('Fiscal Year', fontsize=10)
    ax_trend.set_ylabel('Number of Projects', fontsize=10, color='steelblue')
    ax_trend_twin.set_ylabel('Percentage', fontsize=10, color='crimson')
    ax_trend.set_title('Multi-College Projects Trend', fontsize=12, fontweight='bold')
    ax_trend.tick_params(axis='y', labelcolor='steelblue')
    ax_trend_twin.tick_params(axis='y', labelcolor='crimson')
    plt.setp(ax_trend.xaxis.get_majorticklabels(), rotation=45)
    
    # Add legend
    lines1, labels1 = ax_trend.get_legend_handles_labels()
    lines2, labels2 = ax_trend_twin.get_legend_handles_labels()
    ax_trend.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    
    # Average colleges chart (top right)
    ax_avg = fig.add_subplot(gs[1, 2:])
    line = ax_avg.plot(years, data['avg_colleges_per_multi_project'], marker='o', 
                      linestyle='-', linewidth=2, color='forestgreen')
    ax_avg.set_xlabel('Fiscal Year', fontsize=10)
    ax_avg.set_ylabel('Average Number of Colleges', fontsize=10)
    ax_avg.set_title('Colleges per Multi-College Project', fontsize=12, fontweight='bold')
    plt.setp(ax_avg.xaxis.get_majorticklabels(), rotation=45)
    ax_avg.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Pie chart (bottom left)
    ax_pie = fig.add_subplot(gs[2, :2])
    labels = ['Multi-College', 'Single-College']
    sizes = [comparison_data['multi_college_projects'], comparison_data['single_college_projects']]
    colors = ['#1f77b4', '#ff7f0e']
    ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
              wedgeprops={'edgecolor': 'w', 'linewidth': 1})
    ax_pie.set_title('Project Distribution', fontsize=12, fontweight='bold')
    
    # Award comparison (bottom right)
    ax_award = fig.add_subplot(gs[2, 2:])
    
    if comparison_data['avg_award_multi'] is not None and comparison_data['avg_award_single'] is not None:
        categories = ['Average Award Amount']
        multi_award = [comparison_data['avg_award_multi']]
        single_award = [comparison_data['avg_award_single']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax_award.bar(x - width/2, multi_award, width, label='Multi-College', color='#1f77b4')
        ax_award.bar(x + width/2, single_award, width, label='Single-College', color='#ff7f0e')
        
        ax_award.set_xticks(x)
        ax_award.set_xticklabels(categories)
        ax_award.set_ylabel('Average Award Amount ($)', fontsize=10)
        ax_award.set_title('Funding Comparison', fontsize=12, fontweight='bold')
        ax_award.legend(fontsize=8)
        ax_award.grid(axis='y', linestyle='--', alpha=0.3)
    else:
        ax_award.text(0.5, 0.5, "No award amount data available", ha='center', va='center', fontsize=10)
        ax_award.set_axis_off()
    
    # Add title
    plt.suptitle('Multi-College Projects Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig 

def create_college_network_graph(collaboration_matrix: pd.DataFrame, centrality_metrics: pd.DataFrame = None) -> plt.Figure:
    """
    Create an interactive network graph visualization of college collaborations.
    
    Args:
        collaboration_matrix: DataFrame containing the pairwise college collaboration counts
        centrality_metrics: Optional DataFrame with centrality metrics to size nodes
        
    Returns:
        Matplotlib Figure object with the network graph
    """
    if collaboration_matrix.empty:
        # Create an empty figure with a message
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "No collaboration data available", ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    try:
        import networkx as nx
    except ImportError:
        # Fallback if networkx isn't available
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "NetworkX library required for network visualization", ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Create a new graph
    G = nx.Graph()
    
    # Add nodes (colleges)
    for college in collaboration_matrix.index:
        G.add_node(college)
    
    # Add edges (collaborations)
    for i, college1 in enumerate(collaboration_matrix.index):
        for college2 in collaboration_matrix.columns[i+1:]:  # only upper triangle to avoid duplicates
            weight = collaboration_matrix.loc[college1, college2]
            if weight > 0:
                G.add_edge(college1, college2, weight=weight)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Set node sizes based on centrality if provided
    if centrality_metrics is not None and not centrality_metrics.empty:
        node_sizes = {}
        for _, row in centrality_metrics.iterrows():
            # Scale the degree centrality for visualization
            node_sizes[row['college']] = 1000 * (0.1 + row['degree_centrality'])
    else:
        # Set uniform node sizes based on degree
        node_sizes = {node: 500 for node in G.nodes()}
    
    # Set edge widths based on weights
    edge_widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
    
    # Define SDSU colors for the plot
    SDSU_BLUE = "#0033A0"
    SDSU_YELLOW = "#FFD100"
    
    # Use a spring layout for the graph
    try:
        # Try using the 'kamada_kawai_layout' for better spacing if available
        pos = nx.kamada_kawai_layout(G)
    except:
        # Fall back to spring layout
        pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, 
                         node_size=[node_sizes[node] for node in G.nodes()],
                         node_color=SDSU_BLUE, 
                         alpha=0.8)
    
    nx.draw_networkx_edges(G, pos, 
                         width=edge_widths,
                         alpha=0.5, 
                         edge_color='gray')
    
    # Draw node labels with smaller font for readability
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='white')
    
    # Set title and remove axis
    ax.set_title('College Collaboration Network', fontsize=16, fontweight='bold')
    ax.set_axis_off()
    
    # Add legend explaining node size
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=SDSU_BLUE, 
                  markersize=10, label='College (node size = network centrality)'),
        plt.Line2D([0], [0], color='gray', lw=1, label='Collaboration (edge width = frequency)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add a note about interactivity
    plt.figtext(0.5, 0.01, "Note: The network layout may vary between runs.", 
               ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    return fig

def create_college_heatmap(collaboration_matrix: pd.DataFrame) -> plt.Figure:
    """
    Create a heatmap visualization of the college collaboration matrix.
    
    Args:
        collaboration_matrix: DataFrame containing the pairwise college collaboration counts
        
    Returns:
        Matplotlib Figure object with the heatmap
    """
    if collaboration_matrix.empty:
        # Create an empty figure with a message
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "No collaboration data available", ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define SDSU colors for the plot
    SDSU_BLUE = "#0033A0"
    
    # Create a mask for the diagonal (self-collaborations)
    mask = np.zeros_like(collaboration_matrix, dtype=bool)
    np.fill_diagonal(mask, True)
    
    # Apply the mask to the collaboration matrix
    masked_matrix = np.ma.array(collaboration_matrix, mask=mask)
    
    # Create the heatmap
    im = ax.imshow(masked_matrix, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Number of Collaborations', rotation=270, labelpad=20)
    
    # Set tick labels
    ax.set_xticks(np.arange(len(collaboration_matrix.columns)))
    ax.set_yticks(np.arange(len(collaboration_matrix.index)))
    ax.set_xticklabels(collaboration_matrix.columns, rotation=90)
    ax.set_yticklabels(collaboration_matrix.index)
    
    # Add text annotations in the cells
    for i in range(len(collaboration_matrix.index)):
        for j in range(len(collaboration_matrix.columns)):
            if i != j and collaboration_matrix.iloc[i, j] > 0:
                ax.text(j, i, str(int(collaboration_matrix.iloc[i, j])),
                       ha="center", va="center", 
                       color="white" if collaboration_matrix.iloc[i, j] > 3 else "black")
    
    # Set title
    ax.set_title('College Collaboration Heatmap', fontsize=16, fontweight='bold')
    
    # Add a note
    plt.figtext(0.5, 0.01, "Note: Diagonal values (self-collaborations) are masked.", 
               ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    return fig

def create_college_role_distribution_chart(role_distribution: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """
    Create a chart visualizing the distribution of PI vs. Co-PI roles across colleges.
    
    Args:
        role_distribution: DataFrame with role distribution metrics from analyze_college_role_distribution
        top_n: Number of top colleges to display
        
    Returns:
        Matplotlib Figure object with the role distribution chart
    """
    if role_distribution.empty:
        # Create an empty figure with a message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No role distribution data available", ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Sort by total projects and take top N
    top_colleges = role_distribution.sort_values('total_projects', ascending=False).head(top_n)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Define SDSU colors
    SDSU_BLUE = "#0033A0"
    SDSU_YELLOW = "#FFD100"
    
    # Bar chart showing PI vs. Co-PI project counts
    bar_width = 0.35
    x = np.arange(len(top_colleges))
    
    # PI bars
    ax1.bar(x - bar_width/2, top_colleges['pi_projects'], 
           width=bar_width, color=SDSU_BLUE, label='PI Projects')
    
    # Co-PI bars
    ax1.bar(x + bar_width/2, top_colleges['co_pi_projects'], 
           width=bar_width, color=SDSU_YELLOW, label='Co-PI Projects')
    
    # Set labels and title
    ax1.set_xlabel('College', fontsize=12)
    ax1.set_ylabel('Number of Projects', fontsize=12)
    ax1.set_title('PI vs. Co-PI Project Counts by College', fontsize=14, fontweight='bold')
    
    # Set x-ticks
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_colleges['college'], rotation=90)
    
    # Add grid and legend
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.legend()
    
    # Pie charts showing role percentage distribution
    for i, (_, college) in enumerate(top_colleges.iterrows()):
        # Skip colleges with no projects
        if college['total_projects'] == 0:
            continue
        
        # Calculate position in the grid (4 columns)
        row = i // 4
        col = i % 4
        
        # Create a small pie chart for this college
        size = 0.15  # Size of each pie chart
        margin = 0.05  # Margin between pie charts
        
        # Position the pie chart in the grid
        x_pos = col * (size + margin) + size/2
        y_pos = 1 - (row * (size + margin) + size/2)
        
        # Draw the pie chart
        wedges, texts, autotexts = ax2.pie(
            [college['pi_percentage'], college['co_pi_percentage']],
            colors=[SDSU_BLUE, SDSU_YELLOW],
            autopct='%1.0f%%',
            startangle=90,
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
            textprops={'fontsize': 8},
            radius=size,
            center=(x_pos, y_pos)
        )
        
        # Make the percentage text white for better visibility
        for autotext in autotexts:
            autotext.set_color('white')
        
        # Add college name as a title for each pie
        ax2.text(x_pos, y_pos + size + 0.01, college['college'],
                ha='center', va='bottom', fontsize=8)
    
    # Set title for the pie chart subplot
    ax2.set_title('PI vs. Co-PI Role Distribution by College', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Add legend for the pie charts
    ax2.legend(['PI Projects', 'Co-PI Projects'], loc='upper right')
    
    # Set overall title
    fig.suptitle('College Role Distribution Analysis', fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def create_college_diversity_chart(diversity_metrics: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """
    Create a chart visualizing the diversity of college collaborations.
    
    Args:
        diversity_metrics: DataFrame with diversity metrics from get_college_collaboration_diversity
        top_n: Number of top colleges to display
        
    Returns:
        Matplotlib Figure object with the diversity chart
    """
    if diversity_metrics.empty:
        # Create an empty figure with a message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No diversity metrics available", ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Sort by diversity index and take top N
    top_diverse = diversity_metrics.sort_values('collaboration_diversity', ascending=False).head(top_n)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Define SDSU colors
    SDSU_BLUE = "#0033A0"
    SDSU_YELLOW = "#FFD100"
    
    # Bar chart for diversity index
    ax1.barh(top_diverse['college'], top_diverse['collaboration_diversity'], 
            color=SDSU_BLUE, alpha=0.8)
    
    # Set labels and title
    ax1.set_xlabel('Diversity Index (0-1)', fontsize=12)
    ax1.set_title('Collaboration Diversity by College', fontsize=14, fontweight='bold')
    
    # Add grid
    ax1.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add a reference line at 0.5
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    ax1.text(0.51, len(top_diverse) - 0.5, 'More Diverse →', 
            color='red', fontsize=8, va='center')
    
    # Bar chart for partners vs. total collaborations
    x = np.arange(len(top_diverse))
    bar_width = 0.35
    
    # Unique partners
    ax2.bar(x - bar_width/2, top_diverse['unique_partners'], 
           width=bar_width, color=SDSU_BLUE, label='Unique Partners')
    
    # Effective partners (a measure of diversity)
    ax2.bar(x + bar_width/2, top_diverse['effective_partners'], 
           width=bar_width, color=SDSU_YELLOW, label='Effective Partners')
    
    # Set labels and title
    ax2.set_xlabel('College', fontsize=12)
    ax2.set_ylabel('Number of Partners', fontsize=12)
    ax2.set_title('Unique vs. Effective Partners by College', fontsize=14, fontweight='bold')
    
    # Set x-ticks
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_diverse['college'], rotation=90)
    
    # Add grid and legend
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.legend()
    
    # Set overall title
    fig.suptitle('College Collaboration Diversity Analysis', fontsize=16, fontweight='bold')
    
    # Add note explaining metrics
    plt.figtext(0.5, 0.01, 
               "Note: Diversity Index ranges from 0 (concentrated collaborations) to 1 (diverse collaborations).\n" +
               "Effective Partners represents the diversity-weighted number of collaborating colleges.",
               ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def create_college_collaboration_dashboard(
    collaboration_matrix: pd.DataFrame, 
    centrality_metrics: pd.DataFrame, 
    role_distribution: pd.DataFrame, 
    diversity_metrics: pd.DataFrame
) -> plt.Figure:
    """
    Create a comprehensive dashboard for college collaboration analysis.
    
    Args:
        collaboration_matrix: DataFrame containing the pairwise college collaboration counts
        centrality_metrics: DataFrame with centrality metrics for each college
        role_distribution: DataFrame with role distribution metrics
        diversity_metrics: DataFrame with collaboration diversity metrics
        
    Returns:
        Matplotlib Figure object with the comprehensive dashboard
    """
    # Check if we have any data to display
    if (collaboration_matrix.empty and centrality_metrics.empty and 
        role_distribution.empty and diversity_metrics.empty):
        # Create an empty figure with a message
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, "No college collaboration data available", ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Create figure with subplots grid
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 4)
    
    # Define SDSU colors
    SDSU_BLUE = "#0033A0"
    SDSU_YELLOW = "#FFD100"
    
    # Top section: Title and key metrics
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    
    # Display key metrics if available
    if not centrality_metrics.empty:
        # Get total collaborations
        total_collaborations = int(centrality_metrics['total_collaborations'].sum() / 2)  # Divide by 2 as each collaboration is counted twice
        
        # Get top collaborative college
        top_college = centrality_metrics.iloc[0]['college']
        top_college_collaborations = int(centrality_metrics.iloc[0]['total_collaborations'])
        
        # Get number of colleges in the network
        num_colleges = len(centrality_metrics)
        
        metrics_text = f"""
        Total Collaborations: {total_collaborations}
        Colleges in Network: {num_colleges}
        Top Collaborative College: {top_college} ({top_college_collaborations} collaborations)
        """
    else:
        metrics_text = "No collaboration metrics available"
    
    ax_title.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=12, 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Network visualization (top-left)
    if not collaboration_matrix.empty:
        ax_network = fig.add_subplot(gs[1:, :2])
        
        # Create a smaller network visualization for the dashboard
        try:
            import networkx as nx
            
            # Create network
            G = nx.Graph()
            
            # Add nodes and edges
            for i, college1 in enumerate(collaboration_matrix.index):
                G.add_node(college1)
                for college2 in collaboration_matrix.columns[i+1:]:
                    weight = collaboration_matrix.loc[college1, college2]
                    if weight > 0:
                        G.add_edge(college1, college2, weight=weight)
            
            # Set node sizes based on centrality
            if not centrality_metrics.empty:
                node_sizes = {}
                for _, row in centrality_metrics.iterrows():
                    node_sizes[row['college']] = 500 * (0.1 + row['degree_centrality'])
            else:
                node_sizes = {node: 300 for node in G.nodes()}
            
            # Set edge widths
            edge_widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
            
            # Layout
            try:
                pos = nx.kamada_kawai_layout(G)
            except:
                pos = nx.spring_layout(G, k=0.3, iterations=50)
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, 
                                 node_size=[node_sizes[node] for node in G.nodes()],
                                 node_color=SDSU_BLUE, 
                                 alpha=0.8,
                                 ax=ax_network)
            
            nx.draw_networkx_edges(G, pos, 
                                 width=edge_widths,
                                 alpha=0.5, 
                                 edge_color='gray',
                                 ax=ax_network)
            
            # Simplified labels (only for top nodes by centrality)
            if not centrality_metrics.empty:
                top_nodes = centrality_metrics.head(5)['college'].tolist()
                node_labels = {node: node if node in top_nodes else "" for node in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color='white',
                                      ax=ax_network)
            
            ax_network.set_title('College Collaboration Network', fontsize=14, fontweight='bold')
            ax_network.set_axis_off()
        except ImportError:
            ax_network.text(0.5, 0.5, "NetworkX required for network visualization", 
                          ha='center', va='center', fontsize=12)
            ax_network.set_axis_off()
    else:
        ax_network = fig.add_subplot(gs[1:, :2])
        ax_network.text(0.5, 0.5, "No network data available", ha='center', va='center', fontsize=12)
        ax_network.set_axis_off()
    
    # Top collaborators bar chart (top-right)
    if not centrality_metrics.empty:
        ax_top = fig.add_subplot(gs[1, 2:])
        
        # Get top 5 colleges by centrality
        top_colleges = centrality_metrics.head(5)
        
        # Create horizontal bar chart
        bars = ax_top.barh(top_colleges['college'], top_colleges['degree_centrality'], 
                         color=SDSU_BLUE, alpha=0.8)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax_top.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                      f"{width:.2f}", va='center', fontsize=8)
        
        ax_top.set_xlabel('Degree Centrality', fontsize=10)
        ax_top.set_title('Top Colleges by Network Centrality', fontsize=12, fontweight='bold')
        ax_top.grid(axis='x', linestyle='--', alpha=0.3)
    else:
        ax_top = fig.add_subplot(gs[1, 2:])
        ax_top.text(0.5, 0.5, "No centrality metrics available", ha='center', va='center', fontsize=12)
        ax_top.set_axis_off()
    
    # Role distribution (bottom-right, first row)
    if not role_distribution.empty:
        ax_roles = fig.add_subplot(gs[2, 2])
        
        # Get top 5 colleges by total projects
        top_role_colleges = role_distribution.sort_values('total_projects', ascending=False).head(5)
        
        # Create stacked bar chart
        bottom = np.zeros(len(top_role_colleges))
        
        # PI projects
        ax_roles.barh(top_role_colleges['college'], top_role_colleges['pi_percentage'], 
                    left=bottom, color=SDSU_BLUE, alpha=0.8, label='PI %')
        
        # Co-PI projects
        bottom += top_role_colleges['pi_percentage']
        ax_roles.barh(top_role_colleges['college'], top_role_colleges['co_pi_percentage'], 
                    left=bottom, color=SDSU_YELLOW, alpha=0.8, label='Co-PI %')
        
        ax_roles.set_xlim(0, 100)
        ax_roles.set_xlabel('Percentage', fontsize=10)
        ax_roles.set_title('PI vs. Co-PI Role Distribution', fontsize=12, fontweight='bold')
        ax_roles.legend(loc='lower right', fontsize=8)
    else:
        ax_roles = fig.add_subplot(gs[2, 2])
        ax_roles.text(0.5, 0.5, "No role distribution data", ha='center', va='center', fontsize=12)
        ax_roles.set_axis_off()
    
    # Diversity metrics (bottom-right, second row)
    if not diversity_metrics.empty:
        ax_diversity = fig.add_subplot(gs[2, 3])
        
        # Get top 5 colleges by diversity
        top_diverse = diversity_metrics.sort_values('collaboration_diversity', ascending=False).head(5)
        
        # Create horizontal bar chart
        bars = ax_diversity.barh(top_diverse['college'], top_diverse['collaboration_diversity'], 
                               color=SDSU_BLUE, alpha=0.8)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax_diversity.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                            f"{width:.2f}", va='center', fontsize=8)
        
        ax_diversity.set_xlim(0, 1)
        ax_diversity.set_xlabel('Diversity Index', fontsize=10)
        ax_diversity.set_title('Collaboration Diversity', fontsize=12, fontweight='bold')
        ax_diversity.grid(axis='x', linestyle='--', alpha=0.3)
    else:
        ax_diversity = fig.add_subplot(gs[2, 3])
        ax_diversity.text(0.5, 0.5, "No diversity metrics available", ha='center', va='center', fontsize=12)
        ax_diversity.set_axis_off()
    
    # Set overall title
    fig.suptitle('College Collaboration Network Analysis Dashboard', fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def create_interactive_college_matrix_graph(collaboration_matrix: pd.DataFrame, alt_ready: bool = False):
    """
    Create an interactive college collaboration matrix visualization using Altair.
    
    Args:
        collaboration_matrix: DataFrame containing the pairwise college collaboration counts
        alt_ready: Flag indicating whether the data is already prepared for Altair
        
    Returns:
        Altair Chart object if available, otherwise None
    """
    try:
        import altair as alt
        
        if collaboration_matrix.empty:
            return None
        
        if not alt_ready:
            # Reshape the matrix to long format for Altair
            matrix_data = []
            for i, college1 in enumerate(collaboration_matrix.index):
                for j, college2 in enumerate(collaboration_matrix.columns):
                    if i != j:  # Skip diagonal (self-collaborations)
                        value = collaboration_matrix.loc[college1, college2]
                        if value > 0:  # Only include non-zero values
                            matrix_data.append({
                                'College 1': college1,
                                'College 2': college2,
                                'Collaborations': int(value)
                            })
            
            # Convert to DataFrame
            if not matrix_data:
                return None
            
            matrix_df = pd.DataFrame(matrix_data)
        else:
            matrix_df = collaboration_matrix
        
        # Define SDSU colors
        SDSU_BLUE = "#0033A0"
        
        # Create heatmap with Altair
        chart = alt.Chart(matrix_df).mark_rect().encode(
            x=alt.X('College 1:N', title='College 1'),
            y=alt.Y('College 2:N', title='College 2'),
            color=alt.Color('Collaborations:Q', scale=alt.Scale(scheme='blues')),
            tooltip=['College 1', 'College 2', 'Collaborations']
        ).properties(
            width=600,
            height=400,
            title='College Collaboration Matrix'
        )
        
        # Add text for collaboration counts
        text = alt.Chart(matrix_df).mark_text(baseline='middle').encode(
            x=alt.X('College 1:N'),
            y=alt.Y('College 2:N'),
            text=alt.Text('Collaborations:Q'),
            color=alt.condition(
                alt.datum.Collaborations > 3,
                alt.value('white'),
                alt.value('black')
            )
        )
        
        return chart + text
    
    except ImportError:
        # Return None if Altair is not available
        return None

def create_interactive_network_graph(collaboration_matrix: pd.DataFrame, centrality_metrics: pd.DataFrame = None):
    """
    Create an interactive network graph visualization using Pyvis.
    
    This function creates an interactive network graph that resembles those in applications
    like Obsidian. Users can zoom, pan, drag nodes, and hover over elements to see details.
    
    Args:
        collaboration_matrix: DataFrame containing the pairwise college collaboration counts
        centrality_metrics: Optional DataFrame with centrality metrics to size and color nodes
        
    Returns:
        HTML file path of the generated interactive network or None if dependencies are missing
    """
    try:
        import networkx as nx
        from pyvis.network import Network
        import tempfile
        import os
        
        if collaboration_matrix.empty:
            return None
        
        # Create a NetworkX graph
        G = nx.Graph()
        
        # Add nodes (colleges)
        for college in collaboration_matrix.index:
            G.add_node(college)
        
        # Add edges (collaborations)
        for i, college1 in enumerate(collaboration_matrix.index):
            for j, college2 in enumerate(collaboration_matrix.columns):
                if i != j:  # Skip self-collaborations
                    weight = collaboration_matrix.loc[college1, college2]
                    if weight > 0:
                        G.add_edge(college1, college2, weight=int(weight), title=f"{weight} collaborations")
        
        # Define SDSU colors
        SDSU_BLUE = "#0033A0"
        SDSU_YELLOW = "#FFD100"
        
        # Create pyvis network
        net = Network(height="800px", width="100%", notebook=False, bgcolor="#ffffff")
        
        # Configure physics for better interaction
        net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=200, spring_strength=0.05, damping=0.09)
        
        # Take the NetworkX graph and convert it
        net.from_nx(G)
        
        # Set node sizes and colors based on centrality metrics if provided
        if centrality_metrics is not None and not centrality_metrics.empty:
            # Create a dictionary for quick lookup
            centrality_dict = {}
            for _, row in centrality_metrics.iterrows():
                college = row['college']
                centrality_dict[college] = {
                    'degree': row.get('degree_centrality', 0),
                    'betweenness': row.get('betweenness_centrality', 0),
                    'eigenvector': row.get('eigenvector_centrality', 0),
                    'total_collaborations': row.get('total_collaborations', 0)
                }
            
            # Update nodes with centrality information
            for node in net.nodes:
                college = node['id']
                if college in centrality_dict:
                    metrics = centrality_dict[college]
                    degree = metrics['degree']
                    
                    # Scale node size based on degree centrality
                    size = 15 + (degree * 85)  # Min 15, scales up with centrality
                    color = SDSU_BLUE
                    
                    # Set node properties
                    node['size'] = size
                    node['color'] = color
                    node['title'] = (f"<b>{college}</b><br>"
                                    f"Degree Centrality: {metrics['degree']:.3f}<br>"
                                    f"Betweenness Centrality: {metrics['betweenness']:.3f}<br>"
                                    f"Eigenvector Centrality: {metrics['eigenvector']:.3f}<br>"
                                    f"Total Collaborations: {metrics['total_collaborations']}")
        
        # Update edge properties for better visualization
        for edge in net.edges:
            weight = edge.get('weight', 1)
            # Scale edge width based on weight
            width = 1 + (weight * 0.5)  # Min 1, scales up with weight
            edge['width'] = width
            edge['title'] = f"{weight} collaborations"
            edge['color'] = {'color': '#999999', 'opacity': 0.8}
        
        # Configure other visualization options
        net.toggle_physics(True)
        net.show_buttons(filter_=['physics'])
        
        # Create a temporary file to save the HTML
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp:
            temp_path = temp.name
            
        # Save the interactive network to the temp file
        net.save_graph(temp_path)
        
        return temp_path
    
    except ImportError as e:
        print(f"Warning: Could not create interactive network graph. Missing dependency: {e}")
        return None 