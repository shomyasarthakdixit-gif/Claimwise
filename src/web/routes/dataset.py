from flask_login import login_required
from flask import Blueprint, render_template, request, send_file, abort, current_app
from src.web.services.dataset_service import (
    get_dataset_info, 
    get_dataset_page, 
    get_dataset_schema, 
    get_dataset_statistics,
    get_dataset_filepath,
    ALLOWED_DATASETS
)
import os

dataset_bp = Blueprint('dataset', __name__)

@dataset_bp.route('/dataset')
@login_required
def explorer():
    dataset = request.args.get('dataset', 'train')
    if dataset not in ALLOWED_DATASETS:
        dataset = 'train'
        
    try:
        page = int(request.args.get('page', 1))
    except ValueError:
        page = 1
        
    try:
        page_size = int(request.args.get('page_size', 25))
    except ValueError:
        page_size = 25
        
    if page_size not in [25, 50, 100, 250]:
        page_size = 25
        
    search = request.args.get('search', '').strip()
    sort = request.args.get('sort', '')
    order = request.args.get('order', 'asc')
    
    if order not in ['asc', 'desc']:
        order = 'asc'
        
    filter_column = request.args.get('filter_column', '')
    filter_operator = request.args.get('filter_operator', '')
    filter_value = request.args.get('filter_value', '')

    try:
        info = get_dataset_info(dataset)
        schema = get_dataset_schema(dataset)
        stats = get_dataset_statistics(dataset)
        
        page_result = get_dataset_page(
            dataset_name=dataset, 
            page=page, 
            page_size=page_size, 
            search=search, 
            sort_col=sort, 
            sort_order=order,
            filter_column=filter_column,
            filter_operator=filter_operator,
            filter_value=filter_value
        )
        
        return render_template(
            'dataset.html',
            current_dataset=dataset,
            datasets=ALLOWED_DATASETS.keys(),
            info=info,
            schema=schema,
            stats=stats,
            table_data=page_result['data'],
            columns=page_result['columns'],
            total_rows=page_result['total_rows'],
            total_pages=page_result['total_pages'],
            current_page=page_result['current_page'],
            page_size=page_size,
            search=search,
            sort=sort,
            order=order,
            filter_column=filter_column,
            filter_operator=filter_operator,
            filter_value=filter_value
        )
    except Exception as e:
        current_app.logger.error(f"Dataset explorer error: {e}")
        abort(500)

@dataset_bp.route('/dataset/download')
@login_required
def download():
    dataset = request.args.get('dataset', 'train')
    filepath = get_dataset_filepath(dataset)
    
    if not filepath or not os.path.exists(filepath):
        abort(404, "Requested dataset file not found.")
        
    return send_file(filepath, as_attachment=True)

@dataset_bp.route('/dataset/eda')
@login_required
def eda():
    dataset = request.args.get('dataset', 'train')
    return render_template('eda.html', current_dataset=dataset)

@dataset_bp.route('/dataset/eda_generate', methods=['POST'])
@login_required
def eda_generate():
    dataset = request.form.get('dataset', 'train')
    graph_type = request.form.get('graph_type')
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from src.web.services.dataset_service import _get_dataframe
    import uuid
    
    df = _get_dataframe(dataset)
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    filename = f"eda_{graph_type.replace(' ', '_')}_{uuid.uuid4().hex[:6]}.png"
    filepath = os.path.join(current_app.config['BASE_DIR'], 'Output', 'plot', filename)
    
    plt.figure(figsize=(8, 6))
    if 'Histogram' in graph_type and num_cols:
        sns.histplot(df[num_cols[0]].dropna(), kde=True)
        plt.title(f'Histogram of {num_cols[0]}')
    elif 'Scatter' in graph_type and len(num_cols) >= 2:
        sns.scatterplot(x=df[num_cols[0]], y=df[num_cols[1]])
        plt.title(f'Scatter of {num_cols[0]} vs {num_cols[1]}')
    elif 'Correlation' in graph_type and len(num_cols) >= 2:
        sns.heatmap(df[num_cols[:5]].corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
    else:
        plt.text(0.5, 0.5, f"Placeholder for {graph_type}", ha='center')
        
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    
    return {'url': f"/dataset/plot/{filename}"}

@dataset_bp.route('/dataset/plot/<path:filename>')
@login_required
def serve_plot(filename):
    filepath = os.path.join(current_app.config['BASE_DIR'], 'Output', 'plot', filename)
    if not os.path.exists(filepath):
        abort(404)
    return send_file(filepath)
