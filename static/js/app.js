// Global variables
let sessionId = null;
let nodes = {};
let members = {};
let loads = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Create a new session
    createSession();
    
    // Setup event listeners
    document.getElementById('nodeForm').addEventListener('submit', addNode);
    document.getElementById('memberForm').addEventListener('submit', addMember);
    document.getElementById('loadForm').addEventListener('submit', addLoad);
    document.getElementById('isSupport').addEventListener('change', toggleSupportType);
    document.getElementById('loadType').addEventListener('change', toggleLoadLocation);
    document.getElementById('analyzeBtn').addEventListener('click', analyzeStructure);
    document.getElementById('clearBtn').addEventListener('click', clearStructure);
});

// Create a new session
function createSession() {
    const swayCase = document.getElementById('swayCase').value;

    fetch('http://127.0.0.1:5000/api/create_session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sway_case: swayCase })  // Send sway_case to backend
    })
    .then(response => response.json())
    .then(data => {
        sessionId = data.session_id;
        console.log('Session created with ID:', sessionId);
    })
    .catch(error => {
        console.error('Error creating session:', error);
        showAlert('error', 'Failed to initialize application. Please refresh the page.');
    });
}

// Toggle support type dropdown based on checkbox
function toggleSupportType() {
    const isSupport = document.getElementById('isSupport').checked;
    const supportType = document.getElementById('supportType');
    
    supportType.disabled = !isSupport;
    
    if (!isSupport) {
        supportType.value = '';
    }
}

document.getElementById('isSupport').onchange = function() {
    const supportType = document.getElementById('supportType');
    supportType.disabled = !this.checked;
};

// Toggle load location field based on load type
function toggleLoadLocation() {
    const loadType = document.getElementById('loadType').value;
    const locationField = document.getElementById('locationField');
    
    // Show location for these load types
    const needsLocation = ['point_x'];
    if (needsLocation.includes(loadType)) {
        locationField.style.display = 'block';
        document.getElementById('loadLocation').required = true;
    } else {
        locationField.style.display = 'none';
        document.getElementById('loadLocation').required = false;
        document.getElementById('loadLocation').value = '';
    }
}


function updateStructureDisplay() {
    if (!sessionId) {
        showAlert('error', 'Session not initialized. Please refresh the page.');
        return;
    }

    fetch('http://127.0.0.1:5000/api/get_structure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
    })
    .then(response => response.json())
    .then(data => {
        nodes = data.nodes || {};
        members = data.members || {};
        loads = data.loads || {};

        console.log('Current nodes:', nodes); // Log the current nodes
        updateNodesList();
        updateMembersList();
        updateLoadsList();
    })
    .catch(error => showAlert('error', 'Failed to fetch structure data.'));
}


// Add a node to the structure
function addNode(event) {
    event.preventDefault();
    
    const data = {
        session_id: sessionId,
        name: document.getElementById('nodeName').value,
        x: parseFloat(document.getElementById('nodeX').value),
        y: parseFloat(document.getElementById('nodeY').value),
        is_support: document.getElementById('isSupport').checked,
        support_type: document.getElementById('isSupport').checked ? 
                    document.getElementById('supportType').value : null
    };

    fetch('http://127.0.0.1:5000/api/add_node', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) throw new Error(data.error);
        console.log('Node added:', data.node_name); // Log the added node
        updateStructureDisplay();
        showAlert('success', 'Node added!');
        event.target.reset();
    })
    .catch(error => showAlert('error', error.message));
}

// Add a member to the structure
function addMember(event) {
    event.preventDefault();
    
    const data = {
        session_id: sessionId,
        name: document.getElementById('memberName').value,
        start_node: document.getElementById('startNode').value,
        end_node: document.getElementById('endNode').value,
        EI: parseFloat(document.getElementById('EI').value)
    };

    console.log('Session ID:', sessionId);
    console.log('Data being sent:', data);

    fetch('http://127.0.0.1:5000/api/add_member', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) throw new Error(data.error);
        updateStructureDisplay();
        showAlert('success', 'Member added!');
        event.target.reset();
    })
    .catch(error => {
        console.error('Error adding member:', error);
        showAlert('error', error.message);
    });
}

// Add a load to the structure
function addLoad(event) {
    event.preventDefault();

    const loadType = document.getElementById('loadType').value;
    const magnitude = parseFloat(document.getElementById('loadMagnitude').value);
    const location = parseFloat(document.getElementById('loadLocation').value);

    // Validate magnitude
    if (isNaN(magnitude)) {
        showAlert('error', 'Magnitude must be a number.');
        return;
    }

    // Validate location for point loads
    if (['point_x'].includes(loadType) && isNaN(location)) {
        showAlert('error', 'Location must be specified for point loads.');
        return;
    }

    // Proceed with the request
    const data = {
        session_id: sessionId,
        member: document.getElementById('loadMember').value,
        load_type: loadType,
        magnitude: magnitude,
        location: location,
        is_lateral: document.getElementById('isLateralCheckbox').checked
    };

    fetch('http://127.0.0.1:5000/api/add_load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) throw new Error(data.error);
        updateStructureDisplay();
        showAlert('success', 'Load added!');
        event.target.reset();
    })
    .catch(error => showAlert('error', error.message));
}

// Analyze the structure
function analyzeStructure() {
    // Check if we have enough data to analyze
    if (Object.keys(nodes).length < 2) {
        showAlert('error', 'At least two nodes are required for analysis.');
        return;
    }
    
    if (Object.keys(members).length < 1) {
        showAlert('error', 'At least one member is required for analysis.');
        return;
    }
    
    // Send the structure data to the server for analysis
    fetch('http://127.0.0.1:5000/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) throw new Error(data.error);
        document.getElementById('visualization').src = `data:image/png;base64,${data.visualization}`;
        displayResults(data.results); // Call the displayResults function
    })
    .catch(error => showAlert('error', error.message));
}

// Display analysis results
function displayResults(results) {
    const resultsContainer = document.getElementById('resultsContainer');
    resultsContainer.innerHTML = ''; // Clear previous results

    if (!results) {
        resultsContainer.innerHTML = '<div class="alert alert-warning">No results available.</div>';
        return;
    }

    // Display unknown rotations and displacements
    const unknownsDiv = document.createElement('div');
    unknownsDiv.className = 'mb-4';
    unknownsDiv.innerHTML = `
        <h6>Unknown Rotations and Displacements</h6>
        <ul class="list-group">
            ${Object.entries(results.unknowns).map(([key, value]) => `
                <li class="list-group-item">${key} = ${value.toFixed(4)}</li>
            `).join('')}
        </ul>
    `;
    resultsContainer.appendChild(unknownsDiv);

    // Display member end moments
    const momentsDiv = document.createElement('div');
    momentsDiv.className = 'mb-4';
    momentsDiv.innerHTML = `
        <h6>Member End Moments</h6>
        <ul class="list-group">
            ${Object.entries(results.moments).map(([member, moments]) => `
                <li class="list-group-item">
                    <strong>${member}</strong>:
                    <ul>
                        <li>Start Moment: ${moments.start !== undefined ? moments.start.toFixed(4) : 'N/A'} kNm</li>
                        <li>End Moment: ${moments.end !== undefined ? moments.end.toFixed(4) : 'N/A'} kNm</li>
                    </ul>
                </li>
            `).join('')}
        </ul>
    `;
    resultsContainer.appendChild(momentsDiv);

    // Display member shear forces
    const shearsDiv = document.createElement('div');
    shearsDiv.className = 'mb-4';
    shearsDiv.innerHTML = `
        <h6>Member Shear Forces</h6>
        <ul class="list-group">
            ${Object.entries(results.shears).map(([member, shears]) => `
                <li class="list-group-item">
                    <strong>${member}</strong>:
                    <ul>
                        <li>Start Shear: ${shears.start !== undefined ? shears.start.toFixed(4) : 'N/A'} kN</li>
                        <li>End Shear: ${shears.end !== undefined ? shears.end.toFixed(4) : 'N/A'} kN</li>
                    </ul>
                </li>
            `).join('')}
        </ul>
    `;
    resultsContainer.appendChild(shearsDiv);
}

// Clear the structure
function clearStructure() {
    // Clear data
    nodes = {};
    members = {};
    loads = {};
    
    // Clear UI
    document.getElementById('nodesList').innerHTML = '';
    document.getElementById('membersList').innerHTML = '';
    document.getElementById('loadsList').innerHTML = '';
    document.getElementById('visualization').src = '';
    document.getElementById('resultsContainer').innerHTML = 
        '<div class="alert alert-info">Define your structure and click "Analyze Structure" to see results.</div>';
    
    // Reset forms
    document.getElementById('nodeForm').reset();
    document.getElementById('memberForm').reset();
    document.getElementById('loadForm').reset();
    document.getElementById('supportType').disabled = true;
    
    // Clear dropdowns
    updateNodeSelections();
    updateMemberSelections();
    
    showAlert('info', 'Structure cleared.');
}



// Helper functions

// Update the nodes list display
function updateNodesList() {
    const nodesList = document.getElementById('nodesList');
    nodesList.innerHTML = '';

    for (const [name, node] of Object.entries(nodes)) {
        const nodeItem = document.createElement('li');
        nodeItem.className = 'list-group-item';
        nodeItem.textContent = `${name} (${node.x}, ${node.y}) ${node.is_support ? 'Support: ' + node.support_type : ''}`;
        nodesList.appendChild(nodeItem);
    }

    updateNodeSelections();  // Update dropdown selections dynamically
}


// Update the members list display
function updateMembersList() {
    const membersList = document.getElementById('membersList');
    membersList.innerHTML = '';

    for (const [name, member] of Object.entries(members)) {
        const memberItem = document.createElement('li');
        memberItem.className = 'list-group-item';

        // Access the member properties explicitly
        const startNodeName = member.start_node.name || member.start_node; // Handle both object and string
        const endNodeName = member.end_node.name || member.end_node; // Handle both object and string
        const EI = member.EI;

        // Format the member information
        memberItem.textContent = `${name}: ${startNodeName} to ${endNodeName}, EI=${EI}`;
        membersList.appendChild(memberItem);
    }

    updateMemberSelections();  // Ensure dropdowns are updated dynamically
}


// Update the loads list display
function updateLoadsList() {
    const loadsList = document.getElementById('loadsList');
    loadsList.innerHTML = '';
    
    for (const [id, load] of Object.entries(loads)) {
        const loadItem = document.createElement('li');
        loadItem.className = 'list-group-item';
        
        const memberName = load.member?.name ||  // Handle object
                          load.member ||         // Handle string
                          'Unknown Member';

        let loadText = `Member ${load.member}: `;
        switch(load.load_type) {
            case 'uniform':
                loadText += `Full-span UDL ${load.magnitude} kN/m`;
                break;
            case 'point':
                loadText += `Center point load ${load.magnitude} kN`;
                break;
            case 'point_x':
                loadText += `Offset point load ${load.magnitude} kN at ${load.location}m`;
                break;
            case 'double_point':
                loadText += `Two equal point loads ${load.magnitude} kN`;
                break;
            case 'half_uniform':
                loadText += `Half-span UDL ${load.magnitude} kN/m`;
                break;
            case 'moment':
                loadText += `Moment ${load.magnitude} kNm at ${load.location}m`;
                break;
        }
        
        loadItem.textContent = loadText;
        loadsList.appendChild(loadItem);
    }
}

// Update node selections in dropdowns
function updateNodeSelections() {
    const startNodeSelect = document.getElementById('startNode');
    const endNodeSelect = document.getElementById('endNode');
    
    // Save current selections
    const startNodeValue = startNodeSelect.value;
    const endNodeValue = endNodeSelect.value;
    
    // Clear options except the first one
    while (startNodeSelect.options.length > 1) {
        startNodeSelect.remove(1);
    }
    
    while (endNodeSelect.options.length > 1) {
        endNodeSelect.remove(1);
    }
    
    // Add new options
    for (const name of Object.keys(nodes)) {
        const startOption = document.createElement('option');
        startOption.value = name;
        startOption.textContent = name;
        startNodeSelect.appendChild(startOption);
        
        const endOption = document.createElement('option');
        endOption.value = name;
        endOption.textContent = name;
        endNodeSelect.appendChild(endOption);
    }
    
    // Restore previous selections if possible
    if (startNodeValue && Object.keys(nodes).includes(startNodeValue)) {
        startNodeSelect.value = startNodeValue;
    }
    
    if (endNodeValue && Object.keys(nodes).includes(endNodeValue)) {
        endNodeSelect.value = endNodeValue;
    }
}

// Update member selections in load form
function updateMemberSelections() {
    const memberSelect = document.getElementById('loadMember');
    
    // Save current selection
    const memberValue = memberSelect.value;
    
    // Clear options except the first one
    while (memberSelect.options.length > 1) {
        memberSelect.remove(1);
    }
    
    // Add new options
    for (const name of Object.keys(members)) {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        memberSelect.appendChild(option);
    }
    
    // Restore previous selection if possible
    if (memberValue && Object.keys(members).includes(memberValue)) {
        memberSelect.value = memberValue;
    }
}

// Calculate distance between two nodes
function calculateLength(node1, node2) {
    const dx = node2.x - node1.x;
    const dy = node2.y - node1.y;
    return Math.sqrt(dx * dx + dy * dy);
}

// Show alert message
function showAlert(type, message) {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.setAttribute('role', 'alert');
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Find a good place to show the alert
    const container = document.querySelector('main .container') || document.querySelector('main');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.classList.remove('show');
        setTimeout(() => alertDiv.remove(), 150);
    }, 5000);
}


