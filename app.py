from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import numpy as np
from claude import StructureAnalyzer
import os
import uuid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, origins=["http://127.0.0.1:5500"])

# Store sessions in memory (use a proper database in production)
SESSIONS = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/create_session', methods=['POST'])
def create_session():
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        'analyzer': StructureAnalyzer(),
        'nodes': {},
        'members': {},
        'loads': {}
    }
    return jsonify({'session_id': session_id})

@app.route('/api/add_node', methods=['POST'])
def add_node():
    try:
        data = request.json
        session_id = data.get('session_id')
        if not session_id or session_id not in SESSIONS:
            return jsonify({'error': 'Invalid session ID'}), 400
        
        session = SESSIONS[session_id]
        analyzer = session['analyzer']
        name = data.get('name')
        
        try:
            x = float(data.get('x'))
            y = float(data.get('y'))
        except (TypeError, ValueError):
            return jsonify({'error': 'Invalid coordinates'}), 400
        
        is_support = data.get('is_support', False)
        support_type = data.get('support_type', None)
        
        # Add to analyzer FIRST
        node = analyzer.add_node(name, x, y, is_support, support_type)
        
        # Then add to session storage
        session['nodes'][name] = node.__dict__
        
        print(f"Analyzer nodes after addition: {analyzer.nodes.keys()}")  # Debug log
        return jsonify({'success': True, 'node_name': name})
    except Exception as e:
        print(f"Error in add_node: {str(e)}")  # Log the error
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/add_member', methods=['POST'])
def add_member():
    data = request.json
    session_id = data.get('session_id')
    if not session_id or session_id not in SESSIONS:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    session = SESSIONS[session_id]
    analyzer = session['analyzer']
    name = data.get('name')
    start_node = data.get('start_node')
    end_node = data.get('end_node')
    try:
        EI = float(data.get('EI'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid EI value'}), 400
    
    if start_node not in analyzer.nodes or end_node not in analyzer.nodes:
        return jsonify({'error': 'Start or end node not found'}), 400
    
    member = analyzer.add_member(name, analyzer.nodes[start_node], analyzer.nodes[end_node], EI)
    session['members'][name] = member.__dict__
    
    return jsonify({'success': True, 'member_name': name})

@app.route('/api/add_load', methods=['POST'])
def add_load():
    data = request.json
    session_id = data.get('session_id')
    if not session_id or session_id not in SESSIONS:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    session = SESSIONS[session_id]
    analyzer = session['analyzer']
    member_name = data.get('member')
    load_type = data.get('load_type')
    try:
        magnitude = float(data.get('magnitude'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid magnitude value'}), 400
    location = data.get('location')
    
    if member_name not in analyzer.members:
        return jsonify({'error': f'Member {member_name} not found'}), 400
    
    member = analyzer.members[member_name]
    load = analyzer.add_load(member, load_type, magnitude, location)
    
    load_id = str(uuid.uuid4())
    session['loads'][load_id] = load.__dict__
    
    return jsonify({'success': True, 'load_id': load_id})

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        print("Received data:", data)  # Debug log
        session_id = data.get('session_id')
        if not session_id or session_id not in SESSIONS:
            print("Invalid session ID")  # Debug log
            return jsonify({'error': 'Invalid session ID'}), 400
        
        session = SESSIONS[session_id]
        analyzer = session['analyzer']
        
        if not session['members']:
            print("No members defined in structure")  # Debug log
            return jsonify({'error': 'No members defined in structure'}), 400
        
        print("Starting analysis...")  # Debug log
        success = analyzer.analyze()
        if not success:
            print("Analysis failed")  # Debug log
            return jsonify({'error': 'Analysis failed'}), 500
        
        print("Analysis completed successfully")  # Debug log
        results = {
            'unknowns': analyzer.unknowns,
            'moments': {m.name: {'start': analyzer.results.get(f"M_{m.name}_start"), 'end': analyzer.results.get(f"M_{m.name}_end")} for m in analyzer.members.values()},
            'shears': {m.name: {'start': analyzer.results.get(f"V_{m.name}_start"), 'end': analyzer.results.get(f"V_{m.name}_end")} for m in analyzer.members.values()},
        }
        
        print("Generating visualization...")  # Debug log
        buf = BytesIO()
        analyzer.plot_results()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return jsonify({'success': True, 'results': results, 'visualization': img_base64})
    except Exception as e:
        print(f"Error in analyze: {str(e)}")  # Log the error
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/get_structure', methods=['POST'])
def get_structure():
    data = request.json
    session_id = data.get('session_id')

    if session_id not in SESSIONS:
        return jsonify({'error': 'Invalid session ID'}), 400

    session = SESSIONS[session_id]

    return jsonify({
        'nodes': session.get('nodes', {}),
        'members': session.get('members', {}),
        'loads': session.get('loads', {})
    })


@app.route('/api/clear_session', methods=['POST'])
def clear_session():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in SESSIONS:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    # Reset the session with a new StructureAnalyzer
    SESSIONS[session_id] = {
        'analyzer': StructureAnalyzer(),
        'nodes': {},
        'members': {},
        'loads': {}
    }
    
    return jsonify({'success': True})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
