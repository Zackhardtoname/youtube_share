import matplotlib
import pandas as pd

matplotlib.use('Agg')  # Use non-interactive backend
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os
import json
from scipy.stats import chi2_contingency
from scipy import stats
import numpy as np
from datetime import datetime

# Constants
EXCLUDED_TOPICS = {"Array", "String", "Tree"}
FAANG_COMPANIES = ["Meta", "Apple", "Amazon", "Netflix", "Google"]
AUTO_OPEN_HTML = False  # Set to True to automatically open HTML files in browser
VIZ_CONFIG = {
    'height': 1200,
    'font_size': 14,
    'margin': dict(l=150, r=100, t=150, b=150),
    'combined_height': 1400,
    'combined_margin': dict(l=300, r=100, t=200, b=150)
}

def filter_topics(topics_str):
    """Filter and clean topic strings"""
    topics = [topic.strip() for topic in topics_str.split(',')]
    return [topic for topic in topics if topic not in EXCLUDED_TOPICS]

def get_display_period(time_period):
    """Clean up time period for display"""
    display_period = time_period.replace('.csv', '').strip()
    if display_period.startswith(('1. ', '2. ', '3. ', '4. ', '5. ')):
        display_period = display_period[3:]
    return display_period


def load_topic_mapping():
    """Load frontendQuestionId to topicTags mapping from slug_data.csv"""
    try:
        slug_df = pd.read_csv('../stats/slug_data.csv')
        topic_mapping = {}
        
        for _, row in slug_df.iterrows():
            question_id = str(row['frontendQuestionId'])
            topic_tags = row['topicTags']
            
            if pd.notna(topic_tags) and topic_tags:
                # Split by semicolon and clean up
                topics = [topic.strip() for topic in topic_tags.split(';') if topic.strip()]
                topic_mapping[question_id] = topics
            
        print(f"Loaded topic mapping for {len(topic_mapping)} problems")
        return topic_mapping
    except Exception as e:
        print(f"Error loading topic mapping: {e}")
        return {}


def compare_distributions(dist1, dist2, name1, name2):
    """Compare two topic count distributions using statistical tests"""
    
    # Get all unique topics from both distributions
    all_topics = set(dist1.index) | set(dist2.index)
    
    # Create aligned arrays with 0 for missing topics
    aligned_dist1 = []
    aligned_dist2 = []
    
    for topic in sorted(all_topics):
        aligned_dist1.append(dist1.get(topic, 0))
        aligned_dist2.append(dist2.get(topic, 0))
    
    aligned_dist1 = np.array(aligned_dist1)
    aligned_dist2 = np.array(aligned_dist2)
    
    # Create contingency table for chi-square test
    contingency_table = np.array([aligned_dist1, aligned_dist2])
    
    results = {
        'comparison': f"{name1} vs {name2}",
        'topics_compared': len(all_topics),
        'total_problems_1': aligned_dist1.sum(),
        'total_problems_2': aligned_dist2.sum()
    }
    
    # Chi-square test for independence
    try:
        chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)
        results['chi2_statistic'] = chi2_stat
        results['chi2_p_value'] = chi2_p
        results['chi2_dof'] = dof
        results['chi2_significant'] = chi2_p < 0.05
    except Exception as e:
        results['chi2_error'] = str(e)
    
    # G-test (log-likelihood ratio test) - more appropriate for categorical data
    try:
        # Calculate expected frequencies under null hypothesis of equal proportions
        total1, total2 = aligned_dist1.sum(), aligned_dist2.sum()
        total_combined = total1 + total2
        
        if total_combined > 0:
            # G-statistic calculation
            g_stat = 0
            valid_categories = 0
            
            for i in range(len(aligned_dist1)):
                o1, o2 = aligned_dist1[i], aligned_dist2[i]
                if o1 + o2 > 0:  # Only consider categories with observations
                    expected_total = o1 + o2
                    e1 = expected_total * total1 / total_combined
                    e2 = expected_total * total2 / total_combined
                    
                    if o1 > 0:
                        g_stat += 2 * o1 * np.log(o1 / e1)
                    if o2 > 0:
                        g_stat += 2 * o2 * np.log(o2 / e2)
                    valid_categories += 1
            
            # Calculate p-value using chi-square distribution
            dof = valid_categories - 1
            if dof > 0:
                g_p = 1 - stats.chi2.cdf(g_stat, dof)
                results['g_statistic'] = g_stat
                results['g_p_value'] = g_p
                results['g_dof'] = dof
                results['g_significant'] = g_p < 0.05
            else:
                results['g_error'] = "Insufficient degrees of freedom"
        else:
            results['g_error'] = "No data to compare"
    except Exception as e:
        results['g_error'] = str(e)
    
    # Calculate effect size (Cramér's V for chi-square)
    try:
        n = contingency_table.sum()
        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
        results['cramers_v'] = cramers_v
    except:
        results['cramers_v'] = None
    
    # Top differences in topic proportions
    diff_analysis = []
    for topic in sorted(all_topics):
        count1 = dist1.get(topic, 0)
        count2 = dist2.get(topic, 0)
        prop1_topic = count1 / results['total_problems_1'] if results['total_problems_1'] > 0 else 0
        prop2_topic = count2 / results['total_problems_2'] if results['total_problems_2'] > 0 else 0
        diff = prop1_topic - prop2_topic
        
        diff_analysis.append({
            'topic': topic,
            f'{name1}_count': count1,
            f'{name2}_count': count2,
            f'{name1}_proportion': prop1_topic,
            f'{name2}_proportion': prop2_topic,
            'difference': diff,
            'abs_difference': abs(diff)
        })
    
    # Sort by absolute difference
    diff_analysis.sort(key=lambda x: x['abs_difference'], reverse=True)
    results['top_differences'] = diff_analysis[:10]
    
    return results


def save_statistical_results(results_list, filename="../stats/statistical_comparison_results.txt"):
    """Save statistical comparison results to a text file"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("STATISTICAL COMPARISON OF TOPIC DISTRIBUTIONS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SUMMARY:\n")
        f.write("This analysis compares the topic distributions between different problem sets using:\n")
        f.write("1. Chi-square test of independence - tests if distributions are significantly different\n")
        f.write("2. G-test (log-likelihood ratio) - alternative test for categorical data independence\n")
        f.write("3. Cramer's V - measures effect size for chi-square test (0 = no association, 1 = perfect association)\n")
        f.write("4. Top topic differences - shows which topics differ most between distributions\n\n")
        
        for i, result in enumerate(results_list, 1):
            f.write(f"COMPARISON {i}: {result['comparison']}\n")
            f.write("-" * 50 + "\n")
            
            f.write(f"Total problems compared:\n")
            f.write(f"  - {result['comparison'].split(' vs ')[0]}: {result['total_problems_1']:,}\n")
            f.write(f"  - {result['comparison'].split(' vs ')[1]}: {result['total_problems_2']:,}\n")
            f.write(f"  - Topics analyzed: {result['topics_compared']}\n\n")
            
            # Chi-square test results
            if 'chi2_statistic' in result:
                f.write(f"CHI-SQUARE TEST RESULTS:\n")
                f.write(f"  Chi-square statistic: {result['chi2_statistic']:.4f}\n")
                f.write(f"  P-value: {result['chi2_p_value']:.2e}\n")
                f.write(f"  Degrees of freedom: {result['chi2_dof']}\n")
                f.write(f"  Significant difference: {'YES' if result['chi2_significant'] else 'NO'} (alpha = 0.05)\n")
                if result['cramers_v'] is not None:
                    f.write(f"  Cramer's V (effect size): {result['cramers_v']:.4f}\n")
                f.write(f"  Interpretation: {'The distributions are significantly different' if result['chi2_significant'] else 'No significant difference found'}\n\n")
            else:
                f.write(f"CHI-SQUARE TEST: Error - {result.get('chi2_error', 'Unknown error')}\n\n")
            
            # G-test results
            if 'g_statistic' in result:
                f.write(f"G-TEST (LOG-LIKELIHOOD RATIO) RESULTS:\n")
                f.write(f"  G statistic: {result['g_statistic']:.4f}\n")
                f.write(f"  P-value: {result['g_p_value']:.2e}\n")
                f.write(f"  Degrees of freedom: {result['g_dof']}\n")
                f.write(f"  Significant difference: {'YES' if result['g_significant'] else 'NO'} (alpha = 0.05)\n")
                f.write(f"  Interpretation: {'The distributions are significantly different' if result['g_significant'] else 'No significant difference found'}\n\n")
            else:
                f.write(f"G-TEST: Error - {result.get('g_error', 'Unknown error')}\n\n")
            
            # Top differences
            f.write(f"TOP 10 TOPIC DIFFERENCES (by proportion):\n")
            f.write(f"{'Rank':<4} {'Topic':<20} {result['comparison'].split(' vs ')[0]:<15} {result['comparison'].split(' vs ')[1]:<15} {'Difference':<12}\n")
            f.write("-" * 80 + "\n")
            
            for rank, diff in enumerate(result['top_differences'], 1):
                name1 = result['comparison'].split(' vs ')[0]
                name2 = result['comparison'].split(' vs ')[1]
                prop1 = diff[f'{name1}_proportion']
                prop2 = diff[f'{name2}_proportion']
                
                f.write(f"{rank:<4} {diff['topic']:<20} {prop1:<15.3f} {prop2:<15.3f} {diff['difference']:+.3f}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
        
        f.write("NOTES:\n")
        f.write("- A significant chi-square test (p < 0.05) indicates the topic distributions are different\n")
        f.write("- A significant G-test (p < 0.05) confirms the distributions are different (alternative to chi-square)\n")
        f.write("- G-test is generally more powerful than chi-square for detecting differences in categorical data\n")
        f.write("- Positive differences mean the first dataset has higher proportion of that topic\n")
        f.write("- Negative differences mean the second dataset has higher proportion of that topic\n")
        f.write("- Cramer's V interpretation: 0.1=small, 0.3=medium, 0.5=large effect size\n")
    
    print(f"Statistical results saved to: {filename}")


def create_interactive_visualizations(topic_counts, topic_percentages, company, display_period):
    """Create interactive visualizations using Plotly"""
    
    # Create subplots with vertical layout (2 rows, 1 column)
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Topic Distribution (Pie)', 'Top Topics (Bar)'),
        specs=[
            [{"type": "pie"}],
            [{"type": "bar"}]
        ],
        vertical_spacing=0.15
    )
    
    # 1. Pie chart for top 10 topics
    top_10_counts = topic_counts.head(10)
    fig.add_trace(
        go.Pie(
            labels=top_10_counts.index,
            values=top_10_counts.values,
            name="Topic Distribution",
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
            textinfo='label+percent'
        ),
        row=1, col=1
    )
    
    # 2. Vertical bar chart for top 10 topics
    top_10_percentages = topic_percentages.head(10)
    fig.add_trace(
        go.Bar(
            x=top_10_counts.index,
            y=top_10_counts.values,
            name='Top Topics',
            marker_color='steelblue',
            text=[f'{count}<br>({perc}%)' for count, perc in zip(top_10_counts.values, top_10_percentages.values)],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{customdata}<extra></extra>',
            customdata=[f'{p}%' for p in top_10_percentages.values]
        ),
        row=2, col=1
    )
    
    # Update layout
    title = f'{company} - {display_period} - Interactive Topic Analysis'
    if company == "FAANG_Combined":
        title = f'FAANG Companies - {display_period} - Topic Analysis'
    elif company == "All_Companies_Combined":
        title = f'All Companies - {display_period} - Topic Analysis'
    
    fig.update_layout(
        title=title,
        title_x=0.5,
        height=VIZ_CONFIG['height'],
        showlegend=False,
        font=dict(size=VIZ_CONFIG['font_size']),
        margin=VIZ_CONFIG['margin']
    )
    
    # Update x-axis for bar chart
    fig.update_xaxes(title_text="Topics", row=2, col=1, tickangle=45)
    
    # Update y-axis for bar chart
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    # Save and open HTML
    html_filename = f'../stats/{company}_{display_period.replace(" ", "_")}_interactive_analysis.html'
    fig.write_html(html_filename)
    print(f"Saved interactive visualization: {html_filename}")
    if AUTO_OPEN_HTML:
        webbrowser.open('file://' + os.path.abspath(html_filename))



def analyze_companies_unified(company_filter_list, analysis_name, time_period="3. Six Months.csv"):
    """Unified function to analyze companies by loading data as needed"""
    
    companies_dir = '../stats/leetcode-company-wise-problems/'
    if not os.path.exists(companies_dir):
        print(f"Companies directory not found: {companies_dir}")
        return
    
    display_period = get_display_period(time_period)
    
    # Determine which companies to load
    if company_filter_list:
        # Load specific companies
        companies_to_load = company_filter_list
        print(f"\nAnalyzing specific companies: {', '.join(companies_to_load)}")
    else:
        # Load all companies
        companies_to_load = [name for name in os.listdir(companies_dir) 
                            if os.path.isdir(os.path.join(companies_dir, name)) 
                            and not name.startswith('.')]
        print(f"\nAnalyzing all companies ({len(companies_to_load)} total)")
    
    # Load data for specified companies
    all_dataframes = []
    successful_companies = []
    
    for i, company in enumerate(companies_to_load, 1):
        try:
            if len(companies_to_load) > 10:  # Show progress for large loads
                print(f"\rLoading {i}/{len(companies_to_load)}: {company}", end="", flush=True)
            
            file_path = f'{companies_dir}{company}/{time_period}'
            if not os.path.exists(file_path):
                continue
                
            df = pd.read_csv(file_path)
            
            # Check if there's a Topics column
            if 'Topics' not in df.columns:
                continue
            
            # Add company column
            df['Company'] = company
            all_dataframes.append(df)
            successful_companies.append(company)
            
        except Exception as e:
            continue
    
    if len(companies_to_load) > 10:
        print(f"\nSuccessfully loaded data from {len(successful_companies)} companies")
    
    if not all_dataframes:
        print(f"No data found for specified companies: {company_filter_list}")
        return
    
    # Combine all dataframes
    filtered_df = pd.concat(all_dataframes, ignore_index=True)
    companies_found = successful_companies
    
    print(f"Dataset shape for {analysis_name} ({display_period}): {filtered_df.shape}")
    
    # Extract and filter topics
    all_topics = []
    for topics_str in filtered_df['Topics'].dropna():
        filtered_topics = filter_topics(topics_str)
        all_topics.extend(filtered_topics)
    
    if not all_topics:
        print("No topics found after filtering")
        return
    
    topic_counts = pd.Series(all_topics).value_counts()
    total_questions = len(filtered_df)
    total_topics = len(all_topics)
    topic_percentages = (topic_counts / total_topics * 100).round(1)
    
    print(f"\nMost popular Topics for {analysis_name} ({display_period}):")
    print(f"Total questions analyzed: {total_questions}")
    print("Top 15 Topics with counts and percentages:")
    for topic, count in topic_counts.head(15).items():
        percentage = topic_percentages[topic]
        print(f"{topic}: {count} ({percentage}%)")
    
    # Show company statistics for multi-company analyses
    if len(companies_found) > 1:
        company_question_counts = filtered_df['Company'].value_counts()
        print(f"\nTop 10 companies by number of questions:")
        for company, count in company_question_counts.head(10).items():
            print(f"{company}: {count} questions")
    
    # Create visualization
    create_interactive_visualizations(topic_counts.head(15), 
                                    topic_percentages.head(15), 
                                    analysis_name, display_period)
    
    return topic_counts, topic_percentages


def analyze_neetcode_150():
    """Analyze NeetCode 150 problems with topics from slug_data.csv"""
    
    # Load topic mapping
    topic_mapping = load_topic_mapping()
    if not topic_mapping:
        print("Failed to load topic mapping, skipping NeetCode 150 analysis")
        return
    
    # Load NeetCode 150 problems
    try:
        with open('../stats/neetcode_150.txt', 'r') as f:
            lines = f.readlines()
        print(f"Loaded {len(lines)} lines from NeetCode 150 file")
    except Exception as e:
        print(f"Error loading NeetCode 150 file: {e}")
        return
    
    # Parse problem IDs from the file
    neetcode_problem_ids = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove tab and extract problem ID from lines like "\t1. Two Sum 56.3 % Easy"  
        content = line.lstrip('\t')
        parts = content.split('.')
        if len(parts) >= 2 and parts[0].strip().isdigit():
            problem_id = parts[0].strip()
            neetcode_problem_ids.append(problem_id)
    
    print(f"Parsed {len(neetcode_problem_ids)} problem IDs from NeetCode 150")
    
    # Extract topics for NeetCode 150 problems
    all_topics = []
    problems_with_topics = 0
    problems_without_topics = 0
    
    for problem_id in neetcode_problem_ids:
        if problem_id in topic_mapping:
            topics = topic_mapping[problem_id]
            filtered_topics = [topic for topic in topics if topic not in EXCLUDED_TOPICS]
            all_topics.extend(filtered_topics)
            problems_with_topics += 1
        else:
            problems_without_topics += 1
    
    print(f"Problems with topics: {problems_with_topics}")
    print(f"Problems without topics (missing from slug_data.csv): {problems_without_topics}")
    
    if not all_topics:
        print("No topics found for NeetCode 150 problems after filtering")
        return
    
    # Calculate statistics
    topic_counts = pd.Series(all_topics).value_counts()
    total_questions = problems_with_topics
    total_topics = len(all_topics)
    topic_percentages = (topic_counts / total_topics * 100).round(1)
    
    print(f"\nNeetCode 150 Topic Analysis:")
    print(f"Total problems analyzed: {total_questions}")
    print("Top 15 Topics with counts and percentages:")
    for topic, count in topic_counts.head(15).items():
        percentage = topic_percentages[topic]
        print(f"{topic}: {count} ({percentage}%)")
    
    # Create visualization
    create_interactive_visualizations(topic_counts.head(15), 
                                    topic_percentages.head(15), 
                                    "NeetCode_150_Curated_List", "All_Time")
    
    return topic_counts, topic_percentages


def analyze_local_problems():
    """Analyze local problems from local_problems.json with topics from slug_data.csv"""
    
    # Load topic mapping
    topic_mapping = load_topic_mapping()
    if not topic_mapping:
        print("Failed to load topic mapping, skipping local problems analysis")
        return
    
    # Load local problems
    try:
        with open('../stats/local_problems.json', 'r') as f:
            local_problems = json.load(f)
        print(f"Loaded {len(local_problems)} local problems")
    except Exception as e:
        print(f"Error loading local problems: {e}")
        return
    
    # Extract topics for local problems
    all_topics = []
    problems_with_topics = 0
    problems_without_topics = 0
    
    for problem_id, filename in local_problems.items():
        if problem_id in topic_mapping:
            topics = topic_mapping[problem_id]
            filtered_topics = [topic for topic in topics if topic not in EXCLUDED_TOPICS]
            all_topics.extend(filtered_topics)
            problems_with_topics += 1
        else:
            problems_without_topics += 1
    
    print(f"Problems with topics: {problems_with_topics}")
    print(f"Problems without topics (missing from slug_data.csv): {problems_without_topics}")
    
    if not all_topics:
        print("No topics found for local problems after filtering")
        return
    
    # Calculate statistics
    topic_counts = pd.Series(all_topics).value_counts()
    total_questions = problems_with_topics
    total_topics = len(all_topics)
    topic_percentages = (topic_counts / total_topics * 100).round(1)
    
    print(f"\nLocal Problems Topic Analysis:")
    print(f"Total problems analyzed: {total_questions}")
    print("Top 15 Topics with counts and percentages:")
    for topic, count in topic_counts.head(15).items():
        percentage = topic_percentages[topic]
        print(f"{topic}: {count} ({percentage}%)")
    
    # Create visualization
    create_interactive_visualizations(topic_counts.head(15), 
                                    topic_percentages.head(15), 
                                    "Local_Problems", "All_Time")
    
    return topic_counts, topic_percentages


if __name__ == "__main__":
    time_period = "3. Six Months.csv"
    
    print("Starting comprehensive company analysis...")
    print("This will generate:")
    print("1. Individual FAANG companies (5 HTML files)")
    print("2. FAANG combined analysis (1 HTML file)")
    print("3. All companies combined analysis (1 HTML file)")
    print("4. Local problems analysis (1 HTML file)")
    print("5. NeetCode 150 analysis (1 HTML file)")
    print("6. Statistical comparison analysis (1 text file)")
    print("=" * 80)
    
    # Store topic distributions for statistical comparison
    distributions = {}
    
    # Analyze individual FAANG companies
    print("PHASE 1: INDIVIDUAL FAANG COMPANIES ANALYSIS")
    for company in FAANG_COMPANIES:
        print(f"\n{'-'*20} {company} {'-'*20}")
        analyze_companies_unified([company], company, time_period)
    
    # Analyze FAANG combined
    print("\n" + "=" * 80)
    print("PHASE 2: FAANG COMBINED ANALYSIS")
    faang_results = analyze_companies_unified(FAANG_COMPANIES, "FAANG_Combined", time_period)
    if faang_results:
        distributions['FAANG_Combined'] = faang_results[0]
    
    # Analyze all companies combined
    print("\n" + "=" * 80)
    print("PHASE 3: ALL COMPANIES COMBINED ANALYSIS")
    all_companies_results = analyze_companies_unified(None, "All_Companies_Combined", time_period)
    if all_companies_results:
        distributions['All_Companies_Combined'] = all_companies_results[0]
    
    # Analyze local problems
    print("\n" + "=" * 80)
    print("PHASE 4: LOCAL PROBLEMS ANALYSIS")
    local_results = analyze_local_problems()
    if local_results:
        distributions['Local_Problems'] = local_results[0]
    
    # Analyze NeetCode 150
    print("\n" + "=" * 80)
    print("PHASE 5: NEETCODE 150 ANALYSIS")
    neetcode_results = analyze_neetcode_150()
    if neetcode_results:
        distributions['NeetCode_150'] = neetcode_results[0]
    
    # Perform statistical comparisons
    print("\n" + "=" * 80)
    print("PHASE 6: STATISTICAL COMPARISON ANALYSIS")
    
    if len(distributions) >= 2:
        comparison_results = []
        
        # Compare Local vs All Companies
        if 'Local_Problems' in distributions and 'All_Companies_Combined' in distributions:
            print("Comparing Local Problems vs All Companies...")
            result1 = compare_distributions(
                distributions['Local_Problems'], 
                distributions['All_Companies_Combined'],
                'Local_Problems', 
                'All_Companies_Combined'
            )
            comparison_results.append(result1)
        
        # Compare Local vs FAANG
        if 'Local_Problems' in distributions and 'FAANG_Combined' in distributions:
            print("Comparing Local Problems vs FAANG Combined...")
            result2 = compare_distributions(
                distributions['Local_Problems'], 
                distributions['FAANG_Combined'],
                'Local_Problems', 
                'FAANG_Combined'
            )
            comparison_results.append(result2)
        
        # Compare NeetCode 150 vs All Companies
        if 'NeetCode_150' in distributions and 'All_Companies_Combined' in distributions:
            print("Comparing NeetCode 150 vs All Companies...")
            result3 = compare_distributions(
                distributions['NeetCode_150'], 
                distributions['All_Companies_Combined'],
                'NeetCode_150', 
                'All_Companies_Combined'
            )
            comparison_results.append(result3)
        
        # Compare NeetCode 150 vs FAANG
        if 'NeetCode_150' in distributions and 'FAANG_Combined' in distributions:
            print("Comparing NeetCode 150 vs FAANG Combined...")
            result4 = compare_distributions(
                distributions['NeetCode_150'], 
                distributions['FAANG_Combined'],
                'NeetCode_150', 
                'FAANG_Combined'
            )
            comparison_results.append(result4)
        
        # Save results
        if comparison_results:
            save_statistical_results(comparison_results)
            print(f"Completed {len(comparison_results)} statistical comparisons")
        else:
            print("No valid comparisons could be performed")
    else:
        print("Insufficient data for statistical comparisons")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("Generated files:")
    print("- 5 individual FAANG company HTML files")
    print("- 1 FAANG combined HTML file") 
    print("- 1 All companies combined HTML file")
    print("- 1 Local problems HTML file")
    print("- 1 NeetCode 150 HTML file")
    print("- 1 Statistical comparison results text file")
    print("Total: 10 files")