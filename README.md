Structural Analysis Tool using Slope Deflection Method

This project is a web-based application for the analysis of 2D beams and frames. It provides an interactive user interface to define structural geometry, supports, and loads. The analysis is performed on the backend using the Slope Deflection Method, and the resulting Shear Force Diagrams (SFD) and Bending Moment Diagrams (BMD) are visualized for the user.

Features

  - Interactive UI: Simple and intuitive web interface built with Bootstrap for defining structures.
  - Structure Definition: Easily add nodes, members, and supports (fixed, pinned, roller).
  - Versatile Loading: Apply various load types, including:
      - Uniformly Distributed Loads (UDL)
      - Point loads (at center or any offset)
      - Multiple equal point loads
      - Applied moments
  - Sway and Non-Sway Analysis: Support for analyzing both sway and non-sway frames.
  - Comprehensive Results: Calculates and displays member end moments, shear forces, and unknown joint displacements/rotations.
  - Visual Feedback: Automatically generates and displays the structure, Shear Force Diagram, and Bending Moment Diagram.

Technology Stack

  - Backend: Python with Flask for the web server and API endpoints.
  - Frontend: HTML, JavaScript, and Bootstrap 5 for a responsive user interface.
  - Analysis Engine:
      - NumPy: For efficient numerical computations.
      - SymPy: For symbolic mathematics to solve the system of equilibrium equations.
      - Matplotlib: For generating plots of the SFD and BMD.

Project Structure

.
├── app.py              # Main Flask application, API routes
├── claude.py           # Core structural analysis engine (Slope Deflection Method)
├── templates
│   └── index.html      # Frontend HTML structure
└── static
    └── js
        └── app.js      # Frontend JavaScript for UI logic and API calls

  - `app.py`: The heart of the backend. It serves the `index.html` page and provides API endpoints for creating a session, adding nodes, members, loads, and triggering the analysis.
  - `claude.py`: Contains all the engineering logic. The `StructureAnalyzer` class models the structure and implements the Slope Deflection Method to compute results.
  - `index.html`: The single-page interface for the entire application.
  - `app.js`: Manages the state of the UI, handles form submissions, and communicates with the Flask backend via `fetch` requests.

Setup and Installation

Follow these steps to run the project on your local machine.

Prerequisites:

  - Python 3.x
  - `pip` (Python package installer)

1. Clone the Repository (or download the files)**

If this were a Git repository, you would clone it. For now, just place all the project files (`app.py`, `claude.py`, `index.html`, `app.js`) in a single project folder with the structure mentioned above.

2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# Navigate to your project directory
cd path/to/your/project

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install Dependencies

Create a file named `requirements.txt` in your project's root directory and add the following lines:

`requirements.txt`

```
Flask
Flask-CORS
numpy
matplotlib
sympy
```

Now, install these packages using pip:

```bash
pip install -r requirements.txt
```

4. Run the Application

With your virtual environment still active, run the Flask application:

```bash
python app.py
```

You should see output indicating that the server is running, similar to this:

```
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
```

5. Access the Tool

Open your web browser and navigate to:
[http://127.0.0.1:5000](https://www.google.com/search?q=http://127.0.0.1:5000)

How to Use

1.  Define Nodes: In the "Nodes" section, specify a name, X/Y coordinates for each joint. Check "Is Support" and select a `Support Type` for supports.
2.  Define Members: In the "Members" section, connect the nodes you created. Provide a name and the Flexural Rigidity (`EI`) for each member.
3.  Apply Loads: In the "Loads" section, select a member, a load type, and its magnitude. For offset loads or moments, a location field will appear.
4.  Select Sway Case: Choose the appropriate sway condition for your frame.
5.  Analyze: Click the Analyze Structure button.
6.  View Results: The visualization pane will update with the structure diagram, SFD, and BMD. The results card will show detailed values for moments, shears, and unknown displacements.
