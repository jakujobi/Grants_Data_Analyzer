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